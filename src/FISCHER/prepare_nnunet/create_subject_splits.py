from pathlib import Path
import json
import re

import pandas as pd
from sklearn.model_selection import KFold


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "data" / "nnunet" / "nnUNet_preprocessed" / "Dataset502_TibiaFischer"

MANIFEST_PATH = (PROJECT_ROOT / "data" / "nnunet" / "nnUNet_raw"
                 / "Dataset502_TibiaFischer" / "fischer_dataset502_manifest.csv")

OUTPUT_PATH = DATASET_DIR / "splits_final.json"

N_SPLITS = 5
RANDOM_SEED = 502

EXPECTED_CASES = 44
EXPECTED_SUBJECTS = 23


# ============================================================
# FUNCIONES
# ============================================================

def find_case_id_column(df: pd.DataFrame) -> str:
    """
    Busca automáticamente la columna que contiene identificadores
    del tipo:
        fischer_006_L
        fischer_z057_R
    """
    pattern = re.compile(r"^fischer_(.+)_[LR]$")

    for col in df.columns:
        values = df[col].dropna().astype(str)

        if not values.empty and values.map(
            lambda x: bool(pattern.match(x))
        ).all():
            return col

    raise ValueError(
        "No se encontró en el manifest una columna con case IDs "
        "del tipo 'fischer_006_L' o 'fischer_z057_R'.\n"
        f"Columnas disponibles: {list(df.columns)}"
    )


def extract_subject(case_id: str) -> str:
    """
    Extrae el identificador del sujeto desde el case ID.

    Ejemplos:
        fischer_006_L  -> 006
        fischer_z057_R -> z057
    """
    match = re.match(r"^fischer_(.+)_[LR]$", case_id)

    if match is None:
        raise ValueError(
            f"Case ID no reconocido: {case_id}"
        )

    return match.group(1)


def get_subjects_from_cases(case_ids):
    """
    Obtiene el conjunto de sujetos presentes en una lista
    de case IDs.
    """
    return {
        extract_subject(case_id)
        for case_id in case_ids
    }


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Leer manifest definitivo de Dataset502
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No existe el manifest:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    case_col = find_case_id_column(df)

    cases = sorted(
        df[case_col]
        .astype(str)
        .unique()
    )

    print("=" * 70)
    print("LECTURA DEL DATASET")
    print("=" * 70)

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Columna de casos detectada: {case_col}")
    print(f"Casos encontrados: {len(cases)}")

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"Se esperaban {EXPECTED_CASES} tibias/casos, "
            f"pero se encontraron {len(cases)}."
        )

    # --------------------------------------------------------
    # 2. Agrupar tibias por sujeto
    # --------------------------------------------------------

    subject_to_cases = {}

    for case_id in cases:

        subject = extract_subject(case_id)

        subject_to_cases.setdefault(
            subject,
            []
        ).append(case_id)

    subjects = sorted(
        subject_to_cases.keys()
    )

    print(f"Sujetos encontrados: {len(subjects)}")

    if len(subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Se esperaban {EXPECTED_SUBJECTS} sujetos, "
            f"pero se encontraron {len(subjects)}."
        )

    print("\nCasos por sujeto:")

    for subject in subjects:

        subject_cases = sorted(
            subject_to_cases[subject]
        )

        print(
            f"  {subject:>4}: "
            + ", ".join(subject_cases)
        )

    # --------------------------------------------------------
    # 3. Generar 5 folds A NIVEL DE SUJETO
    # --------------------------------------------------------

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    splits = []

    print("\n" + "=" * 70)
    print("GENERACIÓN DE FOLDS")
    print("=" * 70)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(subjects)
    ):

        train_subjects = [
            subjects[i]
            for i in train_idx
        ]

        val_subjects = [
            subjects[i]
            for i in val_idx
        ]

        train_cases = sorted(
            case_id
            for subject in train_subjects
            for case_id in subject_to_cases[subject]
        )

        val_cases = sorted(
            case_id
            for subject in val_subjects
            for case_id in subject_to_cases[subject]
        )

        # ----------------------------------------------------
        # QA 1
        # Ningún sujeto puede aparecer simultáneamente
        # en entrenamiento y validación.
        # ----------------------------------------------------

        subject_overlap = (
            set(train_subjects)
            & set(val_subjects)
        )

        if subject_overlap:
            raise RuntimeError(
                f"FOLD {fold_idx}: fuga entre train/val. "
                f"Sujetos repetidos: "
                f"{sorted(subject_overlap)}"
            )

        # ----------------------------------------------------
        # QA 2
        # Ningún caso puede aparecer simultáneamente
        # en entrenamiento y validación.
        # ----------------------------------------------------

        case_overlap = (
            set(train_cases)
            & set(val_cases)
        )

        if case_overlap:
            raise RuntimeError(
                f"FOLD {fold_idx}: casos repetidos "
                f"entre train/val: "
                f"{sorted(case_overlap)}"
            )

        # ----------------------------------------------------
        # QA 3
        # Train + val deben cubrir exactamente los 44 casos.
        # ----------------------------------------------------

        combined = (
            set(train_cases)
            | set(val_cases)
        )

        expected = set(cases)

        if combined != expected:

            missing = expected - combined
            extra = combined - expected

            raise RuntimeError(
                f"FOLD {fold_idx}: cobertura incorrecta.\n"
                f"Faltantes: {sorted(missing)}\n"
                f"Extra: {sorted(extra)}"
            )

        # ----------------------------------------------------
        # Guardar fold
        # ----------------------------------------------------

        splits.append(
            {
                "train": train_cases,
                "val": val_cases,
            }
        )

        # ----------------------------------------------------
        # Resumen del fold
        # ----------------------------------------------------

        print(f"\nFOLD {fold_idx}")

        print(
            f"  Train: "
            f"{len(train_subjects)} sujetos / "
            f"{len(train_cases)} tibias"
        )

        print(
            f"  Val:   "
            f"{len(val_subjects)} sujetos / "
            f"{len(val_cases)} tibias"
        )

        print(
            "  Sujetos val: "
            + ", ".join(val_subjects)
        )

        print(
            "  Casos val:   "
            + ", ".join(val_cases)
        )

    # --------------------------------------------------------
    # 4. QA GLOBAL
    # Cada sujeto debe aparecer exactamente una vez
    # como validación.
    # --------------------------------------------------------

    val_subject_counts = {
        subject: 0
        for subject in subjects
    }

    for split in splits:

        val_subjects_fold = (
            get_subjects_from_cases(
                split["val"]
            )
        )

        for subject in val_subjects_fold:
            val_subject_counts[subject] += 1

    bad_counts = {
        subject: count
        for subject, count
        in val_subject_counts.items()
        if count != 1
    }

    if bad_counts:
        raise RuntimeError(
            "No todos los sujetos aparecen "
            "exactamente una vez en validación: "
            f"{bad_counts}"
        )

    # --------------------------------------------------------
    # 5. QA GLOBAL
    # Cada caso/tibia debe aparecer exactamente una vez
    # como validación.
    # --------------------------------------------------------

    val_case_counts = {
        case_id: 0
        for case_id in cases
    }

    for split in splits:

        for case_id in split["val"]:

            if case_id not in val_case_counts:
                raise RuntimeError(
                    f"Caso inesperado en validación: "
                    f"{case_id}"
                )

            val_case_counts[case_id] += 1

    bad_case_counts = {
        case_id: count
        for case_id, count
        in val_case_counts.items()
        if count != 1
    }

    if bad_case_counts:
        raise RuntimeError(
            "No todos los casos aparecen exactamente "
            "una vez en validación: "
            f"{bad_case_counts}"
        )

    # --------------------------------------------------------
    # 6. Guardar splits_final.json
    # --------------------------------------------------------

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            splits,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 7. RESUMEN FINAL + BALANCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BALANCE FINAL DE LOS FOLDS")
    print("=" * 70)

    for fold_idx, split in enumerate(splits):

        train_subjects_fold = (
            get_subjects_from_cases(
                split["train"]
            )
        )

        val_subjects_fold = (
            get_subjects_from_cases(
                split["val"]
            )
        )

        print(
            f"Fold {fold_idx}: "
            f"train = "
            f"{len(train_subjects_fold)} sujetos / "
            f"{len(split['train'])} tibias | "
            f"val = "
            f"{len(val_subjects_fold)} sujetos / "
            f"{len(split['val'])} tibias"
        )

    print("\n" + "=" * 70)
    print("SPLIT FINAL GENERADO CORRECTAMENTE")
    print("=" * 70)

    print(f"Folds: {len(splits)}")
    print(f"Sujetos: {len(subjects)}")
    print(f"Casos: {len(cases)}")

    print(
        "Fuga por sujeto entre train/val: NO"
    )

    print(
        "Fuga por caso entre train/val: NO"
    )

    print(
        "Cada sujeto aparece exactamente "
        "una vez en validación: SÍ"
    )

    print(
        "Cada tibia aparece exactamente "
        "una vez en validación: SÍ"
    )

    print(
        f"Semilla reproducible: {RANDOM_SEED}"
    )

    print(
        f"\nArchivo generado:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()