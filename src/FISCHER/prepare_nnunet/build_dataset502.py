from pathlib import Path
import csv
import json
import shutil

import SimpleITK as sitk


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "fischer"
    / "oriented_rois_final"
)

GT_SUMMARY = SOURCE_ROOT / "fischer_gt_roi_summary.csv"

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "nnunet"
    / "nnUNet_raw"
    / "Dataset502_TibiaFischer"
)

IMAGES_TR = DATASET_ROOT / "imagesTr"
LABELS_TR = DATASET_ROOT / "labelsTr"
DATASET_JSON = DATASET_ROOT / "dataset.json"


# ============================================================
# GEOMETRÍA
# ============================================================

def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    """
    Comprueba que imagen y label ocupen exactamente
    el mismo grid físico.
    """
    return (
        a.GetSize() == b.GetSize()
        and all(
            abs(x - y) <= 1e-6
            for x, y in zip(
                a.GetSpacing(),
                b.GetSpacing(),
            )
        )
        and all(
            abs(x - y) <= 1e-5
            for x, y in zip(
                a.GetOrigin(),
                b.GetOrigin(),
            )
        )
        and all(
            abs(x - y) <= 1e-6
            for x, y in zip(
                a.GetDirection(),
                b.GetDirection(),
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONSTRUCCIÓN Dataset502_TibiaFischer")
    print("=" * 80)

    if not GT_SUMMARY.exists():
        raise FileNotFoundError(
            f"No existe:\n{GT_SUMMARY}"
        )

    # --------------------------------------------------------
    # Leer únicamente pares ya validados
    # --------------------------------------------------------

    with GT_SUMMARY.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    ok_rows = [
        r for r in rows
        if r["status"].strip().upper() == "OK"
    ]

    if len(ok_rows) != 44:
        raise RuntimeError(
            f"Se esperaban 44 pares GT validados; "
            f"encontrados={len(ok_rows)}"
        )

    # --------------------------------------------------------
    # Protección contra sobrescritura
    # --------------------------------------------------------

    if DATASET_ROOT.exists():
        raise RuntimeError(
            f"\nEl dataset ya existe:\n{DATASET_ROOT}\n\n"
            "No se sobrescribirá automáticamente."
        )

    IMAGES_TR.mkdir(
        parents=True,
        exist_ok=False,
    )

    LABELS_TR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Procesar pares
    # --------------------------------------------------------

    manifest = []

    for i, row in enumerate(
        ok_rows,
        start=1,
    ):

        case = row["case"].strip()
        side = row["side"].strip().upper()

        source_ct = Path(
            row["roi_path"]
        )

        source_gt = Path(
            row["gt_roi_path"]
        )

        sample_id = (
            f"fischer_{case}_{side}"
        )

        destination_ct = (
            IMAGES_TR
            / f"{sample_id}_0000.nii.gz"
        )

        destination_gt = (
            LABELS_TR
            / f"{sample_id}.nii.gz"
        )

        print(
            f"[{i:02d}/44] "
            f"{sample_id}: ",
            end="",
        )

        # ----------------------------------------------------
        # Existencia
        # ----------------------------------------------------

        if not source_ct.exists():
            raise FileNotFoundError(
                f"ROI CT inexistente:\n{source_ct}"
            )

        if not source_gt.exists():
            raise FileNotFoundError(
                f"GT ROI inexistente:\n{source_gt}"
            )

        # ----------------------------------------------------
        # QA final CT ↔ GT
        # ----------------------------------------------------

        ct = sitk.ReadImage(
            str(source_ct)
        )

        gt = sitk.ReadImage(
            str(source_gt)
        )

        if not same_geometry(ct, gt):
            raise RuntimeError(
                f"Geometría CT/GT diferente en "
                f"{sample_id}"
            )

        # Labels deben ser solamente 0 y 1.
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(gt)

        labels = set(
            int(x)
            for x in stats.GetLabels()
        )

        # LabelShapeStatistics no reporta background 0.
        if labels != {1}:
            raise RuntimeError(
                f"GT inválido en {sample_id}. "
                f"Labels no-background encontrados: "
                f"{sorted(labels)}"
            )

        # ----------------------------------------------------
        # Copiar SIN modificar los NIfTI
        # ----------------------------------------------------

        shutil.copy2(
            source_ct,
            destination_ct,
        )

        shutil.copy2(
            source_gt,
            destination_gt,
        )

        manifest.append(
            {
                "sample_id": sample_id,
                "case": case,
                "side": side,
                "image": str(destination_ct),
                "label": str(destination_gt),
            }
        )

        print("OK")

    # ========================================================
    # dataset.json
    # ========================================================

    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "tibia": 1
        },
        "numTraining": len(manifest),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }

    with DATASET_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataset_json,
            f,
            indent=4,
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest_path = (
        DATASET_ROOT
        / "fischer_dataset502_manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "case",
                "side",
                "image",
                "label",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest)

    # ========================================================
    # VERIFICACIÓN FINAL
    # ========================================================

    images = sorted(
        IMAGES_TR.glob("*_0000.nii.gz")
    )

    labels = sorted(
        LABELS_TR.glob("*.nii.gz")
    )

    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)

    print(
        f"Imágenes CT : {len(images)}"
    )

    print(
        f"Labels GT   : {len(labels)}"
    )

    print(
        f"numTraining : "
        f"{dataset_json['numTraining']}"
    )

    print(
        f"\nDataset:\n{DATASET_ROOT}"
    )

    print(
        f"\nManifest:\n{manifest_path}"
    )

    print(
        "\nDataset502_TibiaFischer construido correctamente."
    )


if __name__ == "__main__":
    main()