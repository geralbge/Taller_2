from pathlib import Path
import csv
import json
import re
import sys

import numpy as np
import SimpleITK as sitk


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FISCHER_ROOT = PROJECT_ROOT / "data" / "raw" / "fischer"

ROI_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "fischer"
    / "oriented_rois_final"
)

BATCH_CSV = ROI_ROOT / "fischer_oriented_roi_batch_summary.csv"

OUTPUT_CSV = ROI_ROOT / "fischer_gt_roi_summary.csv"

# Tolerancias para reconocer el patrón de header observado.
SPACING_TOL = 1e-6
ORIGIN_TOL_MM = 1e-4
DIRECTION_TOL = 1e-6

# Tolerancia numérica para comprobar inclusión de centros de voxel.
INDEX_TOL = 1e-6


# ============================================================
# BÚSQUEDA DE ARCHIVOS
# ============================================================

def find_seg(case_dir: Path) -> Path:
    candidates = list(
        case_dir.rglob("*-Shanks-Feet_Segmentation.seg.nrrd")
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Se esperaba 1 SEG en {case_dir}; "
            f"encontrados={len(candidates)}"
        )

    return candidates[0]


# ============================================================
# METADATA DE SEGMENTOS
# ============================================================

def get_tibia_label(seg: sitk.Image, side: str) -> int:
    """
    Resuelve Tibia_L / Tibia_R por Segment*_Name.
    No asume un LabelValue fijo.
    """

    target_name = f"Tibia_{side}"

    found = []

    pattern = re.compile(r"^Segment(\d+)_Name$")

    for key in seg.GetMetaDataKeys():

        match = pattern.match(key)

        if not match:
            continue

        segment_index = match.group(1)
        name = seg.GetMetaData(key)

        if name != target_name:
            continue

        label_key = f"Segment{segment_index}_LabelValue"

        if not seg.HasMetaDataKey(label_key):
            raise RuntimeError(
                f"{target_name} existe pero no tiene LabelValue"
            )

        label_value = int(seg.GetMetaData(label_key))
        found.append(label_value)

    if len(found) != 1:
        raise RuntimeError(
            f"Se esperaba exactamente un {target_name}; "
            f"encontrados={len(found)}"
        )

    return found[0]


# ============================================================
# GEOMETRÍA
# ============================================================

def geometry_equal(a: sitk.Image, b: sitk.Image) -> bool:

    return (
        a.GetSize() == b.GetSize()
        and np.allclose(
            a.GetSpacing(),
            b.GetSpacing(),
            atol=SPACING_TOL,
            rtol=0,
        )
        and np.allclose(
            a.GetOrigin(),
            b.GetOrigin(),
            atol=ORIGIN_TOL_MM,
            rtol=0,
        )
        and np.allclose(
            a.GetDirection(),
            b.GetDirection(),
            atol=DIRECTION_TOL,
            rtol=0,
        )
    )


def matches_validated_z_pattern(
    ct: sitk.Image,
    seg: sitk.Image,
) -> bool:
    """
    Reconoce SOLO el patrón sistemático observado en los
    20 casos z... auditados:

        mismo size
        mismo spacing

        CT direction = identidad
        SEG direction = diag(-1, +1, -1)

        SEG origin = (-CT_x, CT_y, -CT_z)

    Si aparece otra discrepancia, NO se armoniza.
    """

    if ct.GetSize() != seg.GetSize():
        return False

    if not np.allclose(
        ct.GetSpacing(),
        seg.GetSpacing(),
        atol=SPACING_TOL,
        rtol=0,
    ):
        return False

    ct_dir = np.asarray(
        ct.GetDirection(),
        dtype=float,
    ).reshape(3, 3)

    seg_dir = np.asarray(
        seg.GetDirection(),
        dtype=float,
    ).reshape(3, 3)

    expected_ct_dir = np.eye(3)

    expected_seg_dir = np.diag(
        [-1.0, 1.0, -1.0]
    )

    if not np.allclose(
        ct_dir,
        expected_ct_dir,
        atol=DIRECTION_TOL,
        rtol=0,
    ):
        return False

    if not np.allclose(
        seg_dir,
        expected_seg_dir,
        atol=DIRECTION_TOL,
        rtol=0,
    ):
        return False

    ct_origin = np.asarray(
        ct.GetOrigin(),
        dtype=float,
    )

    seg_origin = np.asarray(
        seg.GetOrigin(),
        dtype=float,
    )

    expected_seg_origin = np.array(
        [
            -ct_origin[0],
            ct_origin[1],
            -ct_origin[2],
        ],
        dtype=float,
    )

    return bool(
        np.allclose(
            seg_origin,
            expected_seg_origin,
            atol=ORIGIN_TOL_MM,
            rtol=0,
        )
    )


# ============================================================
# MÁSCARA BINARIA
# ============================================================

def make_binary_mask(
    seg: sitk.Image,
    label_value: int,
) -> sitk.Image:

    mask = sitk.Cast(
        seg == int(label_value),
        sitk.sitkUInt8,
    )

    voxel_count = int(
        sitk.GetArrayViewFromImage(mask).sum()
    )

    if voxel_count == 0:
        raise RuntimeError(
            f"Label {label_value} tiene 0 voxels"
        )

    return mask


def harmonize_mask_geometry(
    mask: sitk.Image,
    ct: sitk.Image,
    seg: sitk.Image,
):
    """
    Si CT y SEG ya coinciden: no cambia nada.

    Si presentan exactamente el patrón z... validado:
    conserva el array discreto de la máscara y asigna
    la geometría física del CT.

    No interpola ni resamplea en esta etapa.
    """

    if geometry_equal(ct, seg):
        return mask, "ORIGINAL_MATCH"

    if not matches_validated_z_pattern(ct, seg):
        raise RuntimeError(
            "CT/SEG presentan una discrepancia geométrica "
            "distinta del patrón Fischer z... validado. "
            "No se armonizará automáticamente."
        )

    harmonized = sitk.Image(mask)

    # Mismo size por fail-safe anterior.
    # Solo se sustituye la información espacial.
    harmonized.SetSpacing(ct.GetSpacing())
    harmonized.SetOrigin(ct.GetOrigin())
    harmonized.SetDirection(ct.GetDirection())

    return harmonized, "HEADER_HARMONIZED"


# ============================================================
# COMPROBACIÓN DE CORRESPONDENCIA CON MARKUP
# ============================================================

def read_markup_points(markup_path: Path):
    """
    Lee los dos puntos del Markup de Slicer y devuelve LPS.
    """

    with markup_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    markups = data.get("markups", [])

    if len(markups) != 1:
        raise RuntimeError(
            f"Markup inválido: se esperaba 1 markup en {markup_path}"
        )

    markup = markups[0]

    coordinate_system = str(
        markup.get("coordinateSystem", "")
    ).upper()

    points = []

    for cp in markup.get("controlPoints", []):

        # Slicer puede guardar puntos no definidos.
        position_status = str(
            cp.get("positionStatus", "defined")
        ).lower()

        if position_status not in {
            "defined",
            "preview",
        }:
            continue

        position = cp.get("position")

        if position is None or len(position) != 3:
            continue

        p = np.asarray(
            position,
            dtype=float,
        )

        if coordinate_system == "RAS":
            p = np.array(
                [-p[0], -p[1], p[2]],
                dtype=float,
            )

        elif coordinate_system == "LPS":
            pass

        else:
            raise RuntimeError(
                f"coordinateSystem no reconocido: "
                f"{coordinate_system}"
            )

        points.append(p)

    if len(points) != 2:
        raise RuntimeError(
            f"Se esperaban 2 puntos definidos; "
            f"encontrados={len(points)}"
        )

    return np.vstack(points)


def physical_points_to_indices(
    image: sitk.Image,
    points_lps: np.ndarray,
):
    indices = []

    for p in points_lps:
        idx = (
            image.TransformPhysicalPointToContinuousIndex(
                p.tolist()
            )
        )

        indices.append(
            np.asarray(idx, dtype=float)
        )

    return np.vstack(indices)


def mask_bbox_xyz(mask: sitk.Image):
    """
    Bounding box del GT en coordenadas de índice x,y,z.
    """

    arr = sitk.GetArrayViewFromImage(mask)

    zyx = np.argwhere(arr > 0)

    if len(zyx) == 0:
        raise RuntimeError("Máscara vacía")

    xyz = zyx[:, [2, 1, 0]]

    return (
        xyz.min(axis=0).astype(float),
        xyz.max(axis=0).astype(float),
    )


def distance_point_to_bbox_mm(
    point_index,
    bbox_min,
    bbox_max,
    spacing,
):
    """
    Distancia aproximada desde un punto hasta un AABB de la
    máscara, expresada en mm.

    0 mm = el punto está dentro del bounding box.
    """

    delta = np.maximum(
        np.maximum(
            bbox_min - point_index,
            point_index - bbox_max,
        ),
        0.0,
    )

    delta_mm = delta * spacing

    return float(
        np.linalg.norm(delta_mm)
    )


def validate_markup_against_target(
    target_mask: sitk.Image,
    opposite_mask: sitk.Image,
    points_lps: np.ndarray,
):
    """
    Comprobación independiente de lateralidad/correspondencia.

    No exige que P1/P2 sean voxels exactos de hueso porque los
    puntos son localizaciones aproximadas.

    Compara la proximidad del eje manual a la región espacial
    de la tibia objetivo frente a la tibia contralateral.
    """

    points_idx = physical_points_to_indices(
        target_mask,
        points_lps,
    )

    midpoint_idx = points_idx.mean(axis=0)

    target_min, target_max = mask_bbox_xyz(
        target_mask
    )

    opposite_min, opposite_max = mask_bbox_xyz(
        opposite_mask
    )

    spacing = np.asarray(
        target_mask.GetSpacing(),
        dtype=float,
    )

    target_distance = distance_point_to_bbox_mm(
        midpoint_idx,
        target_min,
        target_max,
        spacing,
    )

    opposite_distance = distance_point_to_bbox_mm(
        midpoint_idx,
        opposite_min,
        opposite_max,
        spacing,
    )

    # El midpoint debería pertenecer o estar muy próximo
    # a la región longitudinal de la tibia seleccionada.
    #
    # 30 mm NO es un criterio anatómico del proyecto:
    # es únicamente un guard de implementación suficientemente
    # permisivo para una localización manual aproximada.
    if target_distance > 30.0:
        raise RuntimeError(
            "El midpoint P1-P2 queda demasiado alejado del "
            f"bounding box de la tibia objetivo: "
            f"{target_distance:.3f} mm"
        )

    # Si el punto medio está claramente más cerca de la tibia
    # contralateral, existe riesgo de lateralidad incorrecta.
    if (
        opposite_distance + 5.0
        < target_distance
    ):
        raise RuntimeError(
            "El midpoint está más próximo a la tibia "
            "contralateral que a la tibia objetivo. "
            f"target={target_distance:.3f} mm, "
            f"opposite={opposite_distance:.3f} mm"
        )

    return target_distance, opposite_distance


# ============================================================
# RETENCIÓN FÍSICA
# ============================================================

def physical_retention(
    source_mask: sitk.Image,
    roi: sitk.Image,
):
    """
    Evalúa qué porcentaje de centros de voxel positivos del GT
    original queda físicamente dentro del grid del ROI.

    Esta es la métrica de crop/retención.
    No usa el volumen de la máscara resampleada como sustituto.
    """

    arr = sitk.GetArrayViewFromImage(
        source_mask
    )

    zyx = np.argwhere(arr > 0)

    total = len(zyx)

    if total == 0:
        raise RuntimeError(
            "Máscara GT vacía"
        )

    # SimpleITK index = x,y,z
    xyz = zyx[:, [2, 1, 0]].astype(float)

    source_spacing = np.asarray(
        source_mask.GetSpacing(),
        dtype=float,
    )

    source_origin = np.asarray(
        source_mask.GetOrigin(),
        dtype=float,
    )

    source_direction = np.asarray(
        source_mask.GetDirection(),
        dtype=float,
    ).reshape(3, 3)

    # Índice -> punto físico LPS
    physical = (
        source_origin[:, None]
        + source_direction
        @ (xyz * source_spacing).T
    ).T

    roi_origin = np.asarray(
        roi.GetOrigin(),
        dtype=float,
    )

    roi_spacing = np.asarray(
        roi.GetSpacing(),
        dtype=float,
    )

    roi_direction = np.asarray(
        roi.GetDirection(),
        dtype=float,
    ).reshape(3, 3)

    # Como direction es ortonormal:
    # physical -> continuous index ROI
    relative = (
        physical - roi_origin
    )

    roi_indices = (
        (roi_direction.T @ relative.T).T
        / roi_spacing
    )

    roi_size = np.asarray(
        roi.GetSize(),
        dtype=float,
    )

    # Un voxel fuente se considera retenido si su centro físico
    # cae dentro del dominio representado por los voxels ROI.
    inside = np.all(
        (
            roi_indices
            >= (-0.5 - INDEX_TOL)
        )
        &
        (
            roi_indices
            <= (roi_size - 0.5 + INDEX_TOL)
        ),
        axis=1,
    )

    inside_count = int(
        np.count_nonzero(inside)
    )

    outside_count = int(
        total - inside_count
    )

    retention_pct = (
        100.0 * inside_count / total
    )

    return (
        total,
        inside_count,
        outside_count,
        retention_pct,
    )


# ============================================================
# RESAMPLING GT -> ROI
# ============================================================

def resample_mask_to_roi(
    mask: sitk.Image,
    roi: sitk.Image,
):
    """
    Lleva la máscara binaria al grid exacto del ROI.

    Para etiquetas discretas se usa nearest-neighbor.
    """

    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(roi)

    resampler.SetTransform(
        sitk.Transform(3, sitk.sitkIdentity)
    )

    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor
    )

    resampler.SetDefaultPixelValue(0)

    resampler.SetOutputPixelType(
        sitk.sitkUInt8
    )

    result = resampler.Execute(mask)

    # Fail-safe: geometría final exactamente igual al ROI.
    if not geometry_equal(result, roi):
        raise RuntimeError(
            "La máscara resampleada no tiene la misma "
            "geometría que el ROI CT."
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FISCHER — PREPARACIÓN GT EN ROI ORIENTADO")
    print("=" * 80)

    if not BATCH_CSV.exists():
        raise FileNotFoundError(BATCH_CSV)

    with BATCH_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if r["status"].strip().upper() == "OK"
    ]

    if len(rows) != 44:
        raise RuntimeError(
            f"Se esperaban 44 tibias; encontradas={len(rows)}"
        )

    results = []

    for i, row in enumerate(rows, start=1):

        case = row["case"].strip()
        side = row["side"].strip().upper()

        print(
            f"[{i:02d}/44] {case} — Tibia {side}: ",
            end="",
        )

        record = {
            "case": case,
            "side": side,
            "status": "ERROR",
            "label_value": "",
            "geometry_mode": "",
            "markup_target_bbox_distance_mm": "",
            "markup_opposite_bbox_distance_mm": "",
            "gt_voxels_original": "",
            "gt_voxels_inside_roi": "",
            "gt_voxels_outside_roi": "",
            "retention_pct": "",
            "gt_voxels_resampled": "",
            "roi_path": row["roi"],
            "gt_roi_path": "",
            "error": "",
        }

        try:

            case_dir = FISCHER_ROOT / case

            ct_path = Path(row["ct"])
            roi_path = Path(row["roi"])
            markup_path = Path(row["markup"])
            seg_path = find_seg(case_dir)

            if not ct_path.exists():
                raise FileNotFoundError(ct_path)

            if not roi_path.exists():
                raise FileNotFoundError(roi_path)

            if not markup_path.exists():
                raise FileNotFoundError(markup_path)

            ct = sitk.ReadImage(str(ct_path))
            seg = sitk.ReadImage(str(seg_path))
            roi = sitk.ReadImage(str(roi_path))

            # ------------------------------------------------
            # Labels por nombre
            # ------------------------------------------------

            target_label = get_tibia_label(
                seg,
                side,
            )

            opposite_side = (
                "R" if side == "L" else "L"
            )

            opposite_label = get_tibia_label(
                seg,
                opposite_side,
            )

            # ------------------------------------------------
            # Crear máscaras binarias desde ARRAY ORIGINAL
            # ------------------------------------------------

            target_mask_raw = make_binary_mask(
                seg,
                target_label,
            )

            opposite_mask_raw = make_binary_mask(
                seg,
                opposite_label,
            )

            # ------------------------------------------------
            # Armonización geométrica controlada
            # ------------------------------------------------

            target_mask, geometry_mode = (
                harmonize_mask_geometry(
                    target_mask_raw,
                    ct,
                    seg,
                )
            )

            opposite_mask, opposite_mode = (
                harmonize_mask_geometry(
                    opposite_mask_raw,
                    ct,
                    seg,
                )
            )

            if geometry_mode != opposite_mode:
                raise RuntimeError(
                    "Modo geométrico inconsistente entre "
                    "máscara objetivo y contralateral."
                )

            # ------------------------------------------------
            # Validación independiente con P1/P2
            # ------------------------------------------------

            points_lps = read_markup_points(
                markup_path
            )

            (
                target_distance,
                opposite_distance,
            ) = validate_markup_against_target(
                target_mask,
                opposite_mask,
                points_lps,
            )

            # ------------------------------------------------
            # Retención física ANTES de resamplear
            # ------------------------------------------------

            (
                total,
                inside,
                outside,
                retention,
            ) = physical_retention(
                target_mask,
                roi,
            )

            # El diseño 120x120x525 fue seleccionado porque
            # debía retener el 100 % del GT en la validación.
            if outside != 0:
                raise RuntimeError(
                    f"ROI NO retiene todo el GT: "
                    f"{outside}/{total} voxels fuera "
                    f"({retention:.9f} % retenido)"
                )

            # ------------------------------------------------
            # Resample NN al grid del ROI
            # ------------------------------------------------

            gt_roi = resample_mask_to_roi(
                target_mask,
                roi,
            )

            gt_resampled_count = int(
                sitk.GetArrayViewFromImage(
                    gt_roi
                ).sum()
            )

            if gt_resampled_count == 0:
                raise RuntimeError(
                    "Máscara resampleada vacía"
                )

            # ------------------------------------------------
            # Guardar
            # ------------------------------------------------

            output_dir = roi_path.parent

            gt_roi_path = (
                output_dir
                / f"fischer_{case}_{side}_gt_roi.nii.gz"
            )

            sitk.WriteImage(
                gt_roi,
                str(gt_roi_path),
                True,
            )

            record.update(
                {
                    "status": "OK",
                    "label_value": target_label,
                    "geometry_mode": geometry_mode,

                    "markup_target_bbox_distance_mm":
                        target_distance,

                    "markup_opposite_bbox_distance_mm":
                        opposite_distance,

                    "gt_voxels_original":
                        total,

                    "gt_voxels_inside_roi":
                        inside,

                    "gt_voxels_outside_roi":
                        outside,

                    "retention_pct":
                        retention,

                    "gt_voxels_resampled":
                        gt_resampled_count,

                    "gt_roi_path":
                        str(gt_roi_path),

                    "error": "",
                }
            )

            print(
                f"OK | label={target_label} | "
                f"{geometry_mode} | "
                f"retención={retention:.6f}%"
            )

        except Exception as e:

            record["error"] = str(e)

            print("ERROR")
            print(f"         {e}")

        results.append(record)

    # ========================================================
    # CSV
    # ========================================================

    fieldnames = list(results[0].keys())

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # RESUMEN
    # ========================================================

    ok = [
        r for r in results
        if r["status"] == "OK"
    ]

    errors = [
        r for r in results
        if r["status"] != "OK"
    ]

    original_match = sum(
        r["geometry_mode"] == "ORIGINAL_MATCH"
        for r in ok
    )

    harmonized = sum(
        r["geometry_mode"] == "HEADER_HARMONIZED"
        for r in ok
    )

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    print(f"Tibias procesadas        : {len(results)}")
    print(f"OK                      : {len(ok)}")
    print(f"ERROR                   : {len(errors)}")
    print(f"Header original         : {original_match}")
    print(f"Header armonizado       : {harmonized}")

    if ok:
        retentions = np.asarray(
            [
                float(r["retention_pct"])
                for r in ok
            ],
            dtype=float,
        )

        print(
            f"Retención mínima        : "
            f"{retentions.min():.9f}%"
        )

        print(
            f"Retención media         : "
            f"{retentions.mean():.9f}%"
        )

    print(f"\nCSV:\n{OUTPUT_CSV}")

    if errors:

        print("\nCASOS CON ERROR:")

        for r in errors:
            print(
                f"  {r['case']} {r['side']}: "
                f"{r['error']}"
            )

        print(
            "\nPREPARACIÓN GT NO CERRADA. "
            "No continuar todavía a nnU-Net."
        )

        sys.exit(1)

    print(
        "\nPREPARACIÓN GT COMPLETADA."
    )


if __name__ == "__main__":
    main()