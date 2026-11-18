from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROI_FOV_MM = np.array([120.0, 120.0, 525.0], dtype=np.float64)

CT_DEFAULT_VALUE = -1024.0


# ============================================================
# UTILIDADES
# ============================================================

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Devuelve un vector unitario."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if n < eps:
        raise ValueError("No se puede normalizar un vector de longitud ~0.")

    return v / n


def read_two_points_from_slicer_markup(
    markup_path: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Lee exactamente dos puntos de un Markups JSON de 3D Slicer.

    Devuelve P1 y P2 en coordenadas físicas LPS [mm].

    Soporta:
        coordinateSystem = "LPS"
        coordinateSystem = "RAS"

    Si vienen en RAS, convierte:
        LPS = (-R, -A, S)
    """

    with markup_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    markups = data.get("markups", [])

    if len(markups) != 1:
        raise ValueError(
            f"Se esperaba exactamente 1 objeto Markup; encontrados: {len(markups)}"
        )

    markup = markups[0]

    coordinate_system = str(
        markup.get("coordinateSystem", data.get("coordinateSystem", ""))
    ).upper()

    control_points = markup.get("controlPoints", [])

    defined_points = []

    for cp in control_points:
        position = cp.get("position")

        if position is None:
            continue

        pos = np.asarray(position, dtype=np.float64)

        if pos.shape != (3,):
            raise ValueError(
                f"Posición inválida en control point: {position}"
            )

        defined_points.append(pos)

    if len(defined_points) != 2:
        raise ValueError(
            "El Markup debe contener exactamente 2 puntos definidos. "
            f"Encontrados: {len(defined_points)}"
        )

    p1 = defined_points[0]
    p2 = defined_points[1]

    if coordinate_system == "RAS":
        ras_to_lps = np.array([-1.0, -1.0, 1.0])
        p1 = p1 * ras_to_lps
        p2 = p2 * ras_to_lps

    elif coordinate_system != "LPS":
        raise ValueError(
            f"coordinateSystem no reconocido: {coordinate_system!r}. "
            "Se esperaba LPS o RAS."
        )

    return p1, p2, coordinate_system


# ============================================================
# BASE ORTONORMAL DEL ROI
# ============================================================

def build_roi_basis(
    p1_lps: np.ndarray,
    p2_lps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye una base ortonormal determinista.

    e3 = eje longitudinal P1 -> P2.

    Para evitar inestabilidad numérica, se escoge como vector auxiliar
    el eje global LPS menos paralelo a e3.

    La base final es dextrógira:
        e1 x e2 = e3
    """

    e3 = normalize(p2_lps - p1_lps)

    global_axes = np.eye(3, dtype=np.float64)

    # |dot| pequeño = eje menos paralelo a e3.
    alignment = np.abs(global_axes @ e3)
    auxiliary = global_axes[np.argmin(alignment)]

    # e1 perpendicular a auxiliary y e3.
    e1 = normalize(np.cross(auxiliary, e3))

    # Completa una base dextrógira.
    e2 = normalize(np.cross(e3, e1))

    # Control numérico.
    basis = np.column_stack((e1, e2, e3))

    gram = basis.T @ basis

    if not np.allclose(gram, np.eye(3), atol=1e-8):
        raise RuntimeError(
            "La base calculada no es ortonormal.\n"
            f"D^T D =\n{gram}"
        )

    if np.linalg.det(basis) < 0.0:
        raise RuntimeError(
            "La base ROI no es dextrógira."
        )

    return e1, e2, e3


# ============================================================
# SPACING EFECTIVO
# ============================================================

def get_image_direction_matrix(image: sitk.Image) -> np.ndarray:
    """
    SimpleITK devuelve direction aplanado por filas.
    Las columnas de la matriz resultante representan los ejes
    físicos de la imagen.
    """

    dim = image.GetDimension()

    if dim != 3:
        raise ValueError(
            f"Se requiere imagen 3D; dimensión recibida: {dim}"
        )

    return np.asarray(
        image.GetDirection(),
        dtype=np.float64,
    ).reshape(3, 3)


def effective_spacing_along_direction(
    u: np.ndarray,
    input_spacing: np.ndarray,
    input_direction: np.ndarray,
) -> float:
    """
    Calcula la resolución efectiva de la grilla original a lo largo
    de una dirección física unitaria u.

    s(u) = 1 / sqrt(
        sum_i [ (u dot d_i) / s_i ]^2
    )

    donde d_i son las columnas de la matriz direction.
    """

    u = normalize(u)

    terms = []

    for i in range(3):
        d_i = input_direction[:, i]
        s_i = float(input_spacing[i])

        if s_i <= 0:
            raise ValueError(
                f"Spacing no positivo detectado: {input_spacing}"
            )

        projection = float(np.dot(u, d_i))

        terms.append((projection / s_i) ** 2)

    denominator = math.sqrt(sum(terms))

    if denominator <= 0:
        raise RuntimeError(
            "No se pudo calcular el spacing efectivo."
        )

    return 1.0 / denominator


def compute_roi_spacing(
    image: sitk.Image,
    roi_direction: np.ndarray,
) -> np.ndarray:
    """
    Calcula spacing efectivo para los tres ejes físicos del ROI.
    """

    input_spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )

    input_direction = get_image_direction_matrix(image)

    roi_spacing = np.array(
        [
            effective_spacing_along_direction(
                roi_direction[:, i],
                input_spacing,
                input_direction,
            )
            for i in range(3)
        ],
        dtype=np.float64,
    )

    return roi_spacing


# ============================================================
# SIZE Y ORIGIN
# ============================================================

def odd_size_covering_fov(
    fov_mm: np.ndarray,
    spacing_mm: np.ndarray,
) -> np.ndarray:
    """
    Calcula un número impar de voxels tal que la distancia
    centro-a-centro entre primer y último voxel sea >= FOV solicitado.

        (N - 1) * spacing >= FOV

    Además, N impar garantiza un voxel central.
    """

    size = []

    for fov, spacing in zip(fov_mm, spacing_mm):
        n = int(math.ceil(float(fov) / float(spacing))) + 1

        if n % 2 == 0:
            n += 1

        size.append(n)

    return np.asarray(size, dtype=np.int64)


def compute_roi_origin(
    center_lps: np.ndarray,
    roi_direction: np.ndarray,
    roi_spacing: np.ndarray,
    roi_size: np.ndarray,
) -> np.ndarray:
    """
    Calcula origin de modo que el voxel central coincida exactamente
    con center_lps.

    PhysicalPoint(index) =
        origin + D @ (index * spacing)
    """

    center_index = (roi_size.astype(np.float64) - 1.0) / 2.0

    offset = roi_direction @ (center_index * roi_spacing)

    return center_lps - offset


# ============================================================
# RESAMPLING
# ============================================================

def resample_ct_to_roi(
    image: sitk.Image,
    roi_size: np.ndarray,
    roi_spacing: np.ndarray,
    roi_origin: np.ndarray,
    roi_direction: np.ndarray,
) -> sitk.Image:
    """
    Remuestrea el CT sobre la nueva grilla física orientada.
    """

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(
        [int(x) for x in roi_size]
    )

    resampler.SetOutputSpacing(
        [float(x) for x in roi_spacing]
    )

    resampler.SetOutputOrigin(
        [float(x) for x in roi_origin]
    )

    resampler.SetOutputDirection(
        tuple(float(x) for x in roi_direction.reshape(-1))
    )

    # Ambas imágenes viven en el mismo espacio físico LPS.
    resampler.SetTransform(
        sitk.Transform(3, sitk.sitkIdentity)
    )

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampler.SetDefaultPixelValue(
        CT_DEFAULT_VALUE
    )

    resampler.SetOutputPixelType(
        image.GetPixelID()
    )

    return resampler.Execute(image)


# ============================================================
# VERIFICACIONES
# ============================================================

def verify_center(
    roi: sitk.Image,
    expected_center_lps: np.ndarray,
    atol_mm: float = 1e-5,
) -> np.ndarray:
    """
    Verifica que el voxel central del ROI corresponda al midpoint.
    """

    size = np.asarray(roi.GetSize(), dtype=np.int64)

    center_index = ((size - 1) // 2).astype(int)

    actual_center = np.asarray(
        roi.TransformIndexToPhysicalPoint(
            [int(x) for x in center_index]
        ),
        dtype=np.float64,
    )

    error = np.linalg.norm(
        actual_center - expected_center_lps
    )

    if error > atol_mm:
        raise RuntimeError(
            "El centro físico del ROI no coincide con el midpoint.\n"
            f"Esperado: {expected_center_lps}\n"
            f"Obtenido: {actual_center}\n"
            f"Error: {error:.9f} mm"
        )

    return actual_center


# ============================================================
# METADATOS
# ============================================================

def build_metadata(
    input_path: Path,
    markup_path: Path,
    p1_lps: np.ndarray,
    p2_lps: np.ndarray,
    markup_coordinate_system: str,
    center_lps: np.ndarray,
    roi_direction: np.ndarray,
    roi_spacing: np.ndarray,
    roi_size: np.ndarray,
    roi_origin: np.ndarray,
) -> dict:

    realized_extent_mm = (
        (roi_size.astype(np.float64) - 1.0)
        * roi_spacing
    )

    return {
        "input_image": str(input_path),
        "input_markup": str(markup_path),

        "markup_coordinate_system_original": markup_coordinate_system,
        "internal_coordinate_system": "LPS",
        "units": "mm",

        "p1_lps_mm": p1_lps.tolist(),
        "p2_lps_mm": p2_lps.tolist(),

        "center_lps_mm": center_lps.tolist(),

        "requested_fov_mm": ROI_FOV_MM.tolist(),

        "roi_size_voxels": [
            int(x) for x in roi_size
        ],

        "roi_spacing_mm": roi_spacing.tolist(),

        "roi_realized_center_to_center_extent_mm":
            realized_extent_mm.tolist(),

        "roi_origin_lps_mm": roi_origin.tolist(),

        "roi_direction_matrix_columns_are_axes_lps":
            roi_direction.tolist(),

        "resampling": {
            "transform": "identity_physical_space",
            "interpolator": "linear",
            "default_value": CT_DEFAULT_VALUE,
        },
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Genera un ROI tibial orientado de "
            "120 x 120 x 525 mm a partir de dos puntos "
            "de 3D Slicer."
        )
    )

    parser.add_argument(
        "--ct",
        required=True,
        type=Path,
        help="CT de entrada (.nrrd, .nii, .nii.gz, etc.).",
    )

    parser.add_argument(
        "--markup",
        required=True,
        type=Path,
        help="Markup JSON de Slicer con exactamente P1 y P2.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Imagen ROI de salida, por ejemplo roi.nii.gz.",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "JSON de metadatos. Si se omite, se crea "
            "junto al ROI."
        ),
    )

    args = parser.parse_args()

    if not args.ct.exists():
        raise FileNotFoundError(
            f"No existe CT: {args.ct}"
        )

    if not args.markup.exists():
        raise FileNotFoundError(
            f"No existe Markup: {args.markup}"
        )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.metadata is None:
        if args.out.name.endswith(".nii.gz"):
            metadata_path = args.out.with_name(
                args.out.name[:-7] + "_roi_metadata.json"
            )
        else:
            metadata_path = args.out.with_suffix(
                ".roi_metadata.json"
            )
    else:
        metadata_path = args.metadata

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. CT
    # --------------------------------------------------------

    print("=" * 80)
    print("GENERACIÓN DE ROI TIBIAL ORIENTADO")
    print("=" * 80)

    print(f"\nCT     : {args.ct}")
    print(f"Markup : {args.markup}")
    print(f"Salida : {args.out}")

    image = sitk.ReadImage(str(args.ct))

    if image.GetDimension() != 3:
        raise ValueError(
            f"El CT debe ser 3D. Dimensión: {image.GetDimension()}"
        )

    print("\nGeometría CT:")
    print(f"  size      : {image.GetSize()}")
    print(f"  spacing   : {image.GetSpacing()}")
    print(f"  origin    : {image.GetOrigin()}")
    print(f"  direction : {image.GetDirection()}")

    # --------------------------------------------------------
    # 2. P1/P2
    # --------------------------------------------------------

    p1, p2, coord_system = read_two_points_from_slicer_markup(
        args.markup
    )

    distance = float(np.linalg.norm(p2 - p1))

    if distance < 50.0:
        raise ValueError(
            f"P1 y P2 están demasiado próximos ({distance:.3f} mm). "
            "Revisar el Markup."
        )

    center = (p1 + p2) / 2.0

    print("\nLocalización:")
    print(f"  sistema original : {coord_system}")
    print(f"  P1 LPS [mm]      : {p1}")
    print(f"  P2 LPS [mm]      : {p2}")
    print(f"  distancia        : {distance:.3f} mm")
    print(f"  midpoint LPS     : {center}")

    # --------------------------------------------------------
    # 3. Base ROI
    # --------------------------------------------------------

    e1, e2, e3 = build_roi_basis(
        p1,
        p2,
    )

    roi_direction = np.column_stack(
        (e1, e2, e3)
    )

    print("\nBase ortonormal:")
    print(f"  e1 transversal : {e1}")
    print(f"  e2 transversal : {e2}")
    print(f"  e3 longitudinal: {e3}")
    print(
        f"  det(D)         : "
        f"{np.linalg.det(roi_direction):.9f}"
    )

    # --------------------------------------------------------
    # 4. Spacing ROI
    # --------------------------------------------------------

    roi_spacing = compute_roi_spacing(
        image,
        roi_direction,
    )

    # --------------------------------------------------------
    # 5. Size
    # --------------------------------------------------------

    roi_size = odd_size_covering_fov(
        ROI_FOV_MM,
        roi_spacing,
    )

    realized_extent = (
        (roi_size.astype(np.float64) - 1.0)
        * roi_spacing
    )

    # --------------------------------------------------------
    # 6. Origin
    # --------------------------------------------------------

    roi_origin = compute_roi_origin(
        center,
        roi_direction,
        roi_spacing,
        roi_size,
    )

    print("\nGrilla ROI:")
    print(
        f"  FOV solicitado [mm] : "
        f"{ROI_FOV_MM}"
    )
    print(
        f"  spacing [mm]        : "
        f"{roi_spacing}"
    )
    print(
        f"  size [vox]          : "
        f"{roi_size}"
    )
    print(
        f"  extensión real [mm] : "
        f"{realized_extent}"
    )
    print(
        f"  origin LPS [mm]     : "
        f"{roi_origin}"
    )

    # --------------------------------------------------------
    # 7. Resample
    # --------------------------------------------------------

    print("\nRemuestreando CT...")

    roi = resample_ct_to_roi(
        image=image,
        roi_size=roi_size,
        roi_spacing=roi_spacing,
        roi_origin=roi_origin,
        roi_direction=roi_direction,
    )

    # --------------------------------------------------------
    # 8. Verificación
    # --------------------------------------------------------

    actual_center = verify_center(
        roi,
        center,
    )

    print(
        f"Centro físico verificado: "
        f"{actual_center}"
    )

    # --------------------------------------------------------
    # 9. Guardar
    # --------------------------------------------------------

    sitk.WriteImage(
        roi,
        str(args.out),
        useCompression=True,
    )

    metadata = build_metadata(
        input_path=args.ct,
        markup_path=args.markup,
        p1_lps=p1,
        p2_lps=p2,
        markup_coordinate_system=coord_system,
        center_lps=center,
        roi_direction=roi_direction,
        roi_spacing=roi_spacing,
        roi_size=roi_size,
        roi_origin=roi_origin,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nArchivos generados:")
    print(f"  ROI      : {args.out}")
    print(f"  Metadata : {metadata_path}")

    print("\nOK.")


if __name__ == "__main__":
    main()