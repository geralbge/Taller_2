"""
Núcleo del módulo de defecto óseo: post-procesamiento, forma de referencia por
interpolación y caracterización geométrica del defecto.

Convenciones
------------
- Todo se trabaja en la ROI orientada de la tibia (create_oriented_tibia_roi.py):
  array (k, j, i) con k = eje longitudinal e3 (de P1 proximal a P2 distal).
  El eje largo es el eje del array de mayor tamaño (axis 0 en estas ROIs).
- Máscaras booleanas; "rellena" = canal medular incluido (superficie externa).
- Unidades: mm, mm^3; porcentajes en %.

Forma de referencia (hueso "intacto" esperado dentro del defecto)
------------------------------------------------------------------
Interpolación basada en forma (shape-based interpolation): se toman las secciones
sanas más cercanas por fuera de los límites marcados por el usuario (promedio de
la distancia con signo en una banda de BAND_MM; si junto al clic la segmentación
está incompleta, la banda se desplaza hasta encontrar cortes con área sana,
máx. HEALTHY_SEARCH_MM, y se emite un aviso de control de calidad), y para cada
corte intermedio se interpola linealmente la función de distancia con signo; la
sección de referencia es {SDF <= 0}.

Caracterización
---------------
Dentro de los límites [k1, k2] marcados por el usuario:
  - referencia R(k) (interpolada), hueso remanente B(k) (segmentación)
  - defecto angular: la sección R(k) se divide en N_BINS sectores angulares
    alrededor de su centroide; un sector está "presente" si su contorno externo
    tiene hueso a <= COVER_TOL_MM. D(k) = píxeles de R(k) en sectores ausentes.
    (Robusto a que la segmentación sea hueca: no depende del canal medular.)
  - conectividad (vigente desde 04/11/2026, CONNECTED_DEFECT): D se restringe a su
    componente conexa 3D principal (26-vecindad). Un foco lítico separado del defecto por
    hueso conservado es una lesión adicional del huésped, no parte del defecto segmentario;
    uno confluente sí lo es. Evaluado en el benchmark fold 0 (130 casos): sin cambios en
    limpio/A/C; escenario B 16 -> 19/20 criterios correctos con el modelo ajustado.
  - corte "defectuoso" si área(D)/área(R) >= SLICE_DEFECT_FRAC
  - longitud del defecto = extensión (mm) del mayor tramo contiguo de cortes
    defectuosos (tolerando huecos < GAP_TOL_MM); nº de tramos separados por
    >= GAP_TOL_MM de hueso sano -> indicador multisegmentario
  - pérdida cortical por corte = % de sectores angulares ausentes (perímetro
    perdido); se informa máxima y media en el defecto
  - volumen del defecto = |D| * volumen de vóxel
  - criterios del proyecto: 25 <= L <= 70 mm, pérdida cortical > 50 %,
    un solo segmento, ubicación diafisaria (fuera del 10 % de cada extremo,
    como aproximación a "sin compromiso articular").

Control de calidad
------------------
  - banda sana desplazada > 2 mm o no encontrada (ver arriba)
  - densidad (si se entrega la TC): % del defecto con DENSE_HU <= HU < METAL_HU,
    excluyendo <= 1 mm del hueso segmentado y del metal. Global > 5 % o
    > 25 % en alguna ventana de 5 mm -> la segmentación dejó hueso fuera.
    Umbrales fijados con los 35 defectos virtuales del modo pred (error
    relevante = |dL| >= 3 mm, |dV| >= 10 % o |d pérdida| >= 10 pp).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

BAND_MM = 5.0             # banda sana usada a cada lado para la referencia
SLICE_DEFECT_FRAC = 0.10  # fracción de área faltante para considerar un corte defectuoso
GAP_TOL_MM = 5.0          # tramos defectuosos separados por menos de esto se unen
MIN_COMPONENT_FRAC = 0.01 # componentes < 1 % del total se descartan (ruido)
END_EXCLUSION_FRAC = 0.10 # extremos de la tibia excluidos (aprox. articular)
N_BINS = 180              # sectores angulares de 2°
COVER_TOL_MM = 2.0        # tolerancia contorno de referencia -> hueso (error de interpolación)

CRITERIA = dict(length_min_mm=25.0, length_max_mm=70.0, cortical_loss_min_pct=50.0)

# Control de calidad por densidad (requiere la TC en HU, misma grilla que la ROI).
# Idea: lo que el sistema declara "defecto" no debería ser hueso en la TC. Si una
# fracción relevante del defecto tiene densidad cortical, la segmentación dejó
# hueso fuera (sub-segmentación) y la caracterización no es confiable.
DENSE_HU = 700.0          # umbral de densidad cortical (mismo criterio que la verificación de HU de DRYAD)
METAL_HU = 2500.0         # >= esto se considera metal (clavo, tornillos) y se excluye
DENSE_EXCL_MM = 1.0       # se excluyen vóxeles a <= 1 mm del hueso segmentado o del metal (volumen parcial / blooming)
DENSE_WIN_MM = 5.0        # ventana deslizante para detectar fallas localizadas
DENSE_FRAC_MAX = 0.05     # aviso si > 5 % del volumen del defecto es denso (calibrado con defectos virtuales, modo pred)
DENSE_WIN_FRAC_MAX = 0.25 # aviso si en alguna ventana de 5 mm > 25 % es denso (calibrado con defectos virtuales, modo pred)
DENSE_WIN_MIN_MM3 = 50.0  # ventanas con menos volumen de defecto no se evalúan (denominador inestable)


# ------------------------------------------------------------------ E/S
def read_image(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def write_like(arr, ref_img, path, dtype=np.uint8):
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(arr.astype(dtype))
    out.CopyInformation(ref_img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path), useCompression=True)


def spacing_kji(img):
    return np.array(img.GetSpacing()[::-1], dtype=float)


def physical_to_k(img, point_lps):
    """Índice continuo k (eje largo de la ROI) de un punto físico LPS."""
    idx = img.TransformPhysicalPointToContinuousIndex([float(x) for x in point_lps])
    return float(idx[2])


# ------------------------------------------------------------------ máscaras
def keep_components(m, min_frac=MIN_COMPONENT_FRAC):
    """
    Conserva todas las componentes >= min_frac del total (no solo la mayor):
    con un defecto circunferencial la tibia queda en 2 fragmentos.
    """
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return m.astype(bool)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.nonzero(sizes >= min_frac * sizes.sum())[0]
    return np.isin(lab, keep)


def fill_slices(m, axis=0):
    out = m.copy()
    for k in range(m.shape[axis]):
        idx = [slice(None)] * 3; idx[axis] = k
        out[tuple(idx)] = ndi.binary_fill_holes(m[tuple(idx)])
    return out


def postprocess_bone(m):
    """Componentes relevantes + relleno de canal (2D por corte + 3D)."""
    m = keep_components(m.astype(bool))
    return ndi.binary_fill_holes(fill_slices(m))


# ------------------------------------------------------------------ referencia
def sdf2d(section, spacing_ji):
    """Distancia con signo (mm): negativa dentro, positiva fuera."""
    if not section.any():
        return None
    outside = ndi.distance_transform_edt(~section, sampling=spacing_ji)
    inside = ndi.distance_transform_edt(section, sampling=spacing_ji)
    return outside - inside


def band_sdf(mask, k_lo, k_hi, spacing_ji):
    sdfs = [sdf2d(mask[k], spacing_ji) for k in range(max(k_lo, 0), min(k_hi, mask.shape[0]))]
    sdfs = [s for s in sdfs if s is not None]
    if not sdfs:
        raise RuntimeError(f"Sin hueso en la banda de referencia k={k_lo}..{k_hi}")
    return np.mean(sdfs, axis=0)


HEALTHY_SEARCH_MM = 40.0   # búsqueda máxima de banda sana más allá de cada clic
HEALTHY_AREA_FRAC = 0.85   # área mínima de la banda respecto de la mediana de los 10 mm siguientes
HEALTHY_LOOKAHEAD_MM = 10.0
# F3 v2 (opcional; por defecto desactivado = comportamiento original). Ver set_f3_v2().
HEALTHY_AREA_MAX_FRAC = None   # área máxima de la banda respecto de la misma mediana (rechaza bandas engrosadas: involucro/callo)
CONNECTED_DEFECT = True        # (vigente desde 04/11/2026) el defecto es solo la componente 3D principal (focos líticos separados no cuentan)


def set_connected_only(on=True):
    """Solo la regla de conectividad (defecto = componente 3D principal); banda sana original."""
    global CONNECTED_DEFECT
    set_f3_v2(False)
    CONNECTED_DEFECT = bool(on)


def set_f3_v2(on=True):
    """F3 v2: banda sana con límite inferior y superior de área (0,92-1,08 de la mediana de los
    20 mm siguientes, búsqueda hasta 70 mm) + defecto = componente conexa 3D principal."""
    global HEALTHY_SEARCH_MM, HEALTHY_AREA_FRAC, HEALTHY_LOOKAHEAD_MM, HEALTHY_AREA_MAX_FRAC, CONNECTED_DEFECT
    if on:
        HEALTHY_SEARCH_MM, HEALTHY_AREA_FRAC, HEALTHY_LOOKAHEAD_MM = 70.0, 0.92, 20.0
        HEALTHY_AREA_MAX_FRAC, CONNECTED_DEFECT = 1.08, True
    else:
        HEALTHY_SEARCH_MM, HEALTHY_AREA_FRAC, HEALTHY_LOOKAHEAD_MM = 40.0, 0.85, 10.0
        HEALTHY_AREA_MAX_FRAC, CONNECTED_DEFECT = None, False


def find_healthy_band(bone_filled, k_click, direction, sp):
    """
    Busca, alejándose del clic (direction=-1 proximal, +1 distal), la primera
    banda de BAND_MM cuyo corte de menor área tenga >= HEALTHY_AREA_FRAC de la
    mediana del área en los HEALTHY_LOOKAHEAD_MM siguientes (más alejados del
    defecto); máx. HEALTHY_SEARCH_MM. Protege la referencia de una segmentación incompleta
    junto al defecto. Devuelve (k_lo, k_hi_excl, desplazamiento_mm, encontrada).
    """
    band = max(1, int(round(BAND_MM / sp[0])))
    search = int(round(HEALTHY_SEARCH_MM / sp[0]))
    look = int(round(HEALTHY_LOOKAHEAD_MM / sp[0]))
    nk = bone_filled.shape[0]
    area = bone_filled.sum(axis=(1, 2)).astype(float)
    for shift in range(0, search + 1):
        if direction < 0:
            lo, hi = k_click - shift - band, k_click - shift
            beyond = area[max(lo - look, 0):lo]
        else:
            lo, hi = k_click + 1 + shift, k_click + 1 + shift + band
            beyond = area[hi:min(hi + look, nk)]
        if lo < 0 or hi > nk or beyond.size == 0:
            break
        med = np.median(beyond)
        a = area[lo:hi]
        if HEALTHY_AREA_MAX_FRAC is None:
            ok = a.min() >= HEALTHY_AREA_FRAC * med
        else:
            ok = med > 0 and a.min() > 0 and a.min() >= HEALTHY_AREA_FRAC * med and a.max() <= HEALTHY_AREA_MAX_FRAC * med
        if ok:
            return lo, hi, shift * sp[0], True
    lo, hi = (k_click - band, k_click) if direction < 0 else (k_click + 1, k_click + 1 + band)
    return lo, hi, 0.0, False


def interpolated_reference(bone_filled, k1, k2, sp, info=None):
    """
    Forma de referencia entre k1 y k2 (inclusive) por interpolación de SDF entre
    la banda sana proximal más cercana a k1 y la distal más cercana a k2
    (ver find_healthy_band).
    """
    sp_ji = sp[1:]
    p_lo, p_hi, p_shift, p_ok = find_healthy_band(bone_filled, k1, -1, sp)
    d_lo, d_hi, d_shift, d_ok = find_healthy_band(bone_filled, k2, +1, sp)
    s_p = band_sdf(bone_filled, p_lo, p_hi, sp_ji)
    s_d = band_sdf(bone_filled, d_lo, d_hi, sp_ji)
    ref = np.zeros_like(bone_filled, dtype=bool)
    span = d_lo - p_hi                      # cortes entre ambas bandas
    for k in range(k1, k2 + 1):
        t = (k - p_hi + 0.5) / span
        ref[k] = ((1 - t) * s_p + t * s_d) <= 0
    if info is not None:
        info.update(ref_band_shift_prox_mm=p_shift, ref_band_shift_dist_mm=d_shift,
                    ref_band_found_prox=p_ok, ref_band_found_dist=d_ok)
    return ref


# ------------------------------------------------------------------ medición
def runs(flags):
    """Tramos contiguos de True -> lista de (inicio, fin) inclusive."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def merge_runs(rs, gap):
    if not rs:
        return rs
    merged = [list(rs[0])]
    for a, b in rs[1:]:
        if a - merged[-1][1] - 1 < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]


def angular_defect_slice(ref_s, bone_s, sp_ji, n_bins=N_BINS):
    """
    Devuelve (máscara de defecto en el corte, % de perímetro perdido).
    """
    if not ref_s.any():
        return np.zeros_like(ref_s), np.nan
    cj, ci = ndi.center_of_mass(ref_s)
    jj, ii = np.nonzero(ref_s)
    ang = np.arctan2((jj - cj) * sp_ji[0], (ii - ci) * sp_ji[1])
    bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    contour = ref_s & ~ndi.binary_erosion(ref_s)
    if bone_s.any():
        dist_bone = ndi.distance_transform_edt(~bone_s, sampling=sp_ji)
        covered = contour & (dist_bone <= COVER_TOL_MM)
    else:
        covered = np.zeros_like(contour)
    on_contour = contour[jj, ii]
    cov = covered[jj, ii]
    present = np.full(n_bins, np.nan)
    for b in np.unique(bins[on_contour]):
        sel = on_contour & (bins == b)
        present[b] = 1.0 if cov[sel].any() else 0.0
    if np.isnan(present).all():
        return np.zeros_like(ref_s), np.nan
    idx = np.nonzero(~np.isnan(present))[0]          # rellenar bins sin contorno (vecino circular)
    for b in np.nonzero(np.isnan(present))[0]:
        dist = np.minimum(np.abs(idx - b), n_bins - np.abs(idx - b))
        present[b] = present[idx[np.argmin(dist)]]
    missing = present[bins] == 0
    d = np.zeros_like(ref_s)
    d[jj[missing], ii[missing]] = True
    return d, 100.0 * float((present == 0).mean())


def density_qc(defect, ct, bone, sp):
    """
    Fracción del defecto con densidad cortical en la TC (DENSE_HU <= HU < METAL_HU),
    excluyendo vóxeles a <= DENSE_EXCL_MM del hueso segmentado o del metal.
    Trabaja en el recorte (bounding box) del defecto para no calcular la
    transformada de distancia en toda la ROI.
    """
    out = dict(dense_frac_pct=0.0, dense_frac_win_max_pct=0.0, dense_volume_mm3=0.0,
               dense_win_max_k=None, qc_density_warning=False)
    if not defect.any():
        return out
    idx = np.nonzero(defect)
    m = [int(np.ceil(3 * DENSE_EXCL_MM / s)) + 1 for s in sp]
    sl = tuple(slice(max(int(a.min()) - mm, 0), min(int(a.max()) + mm + 1, n))
               for a, mm, n in zip(idx, m, defect.shape))
    d, c, b = defect[sl], ct[sl].astype(np.float32), bone[sl]
    near = (b | (c >= METAL_HU))
    dist = ndi.distance_transform_edt(~near, sampling=sp)
    dense = d & (c >= DENSE_HU) & (c < METAL_HU) & (dist > DENSE_EXCL_MM)
    vox = float(np.prod(sp))
    nd = dense.sum(axis=(1, 2)).astype(float)
    nt = d.sum(axis=(1, 2)).astype(float)
    w = max(1, int(round(DENSE_WIN_MM / sp[0])))
    ker = np.ones(w)
    sd, st = np.convolve(nd, ker, "same"), np.convolve(nt, ker, "same")
    ok = st * vox >= DENSE_WIN_MIN_MM3
    frac_w = np.where(ok, sd / np.maximum(st, 1), 0.0)
    out.update(dense_frac_pct=100.0 * nd.sum() / nt.sum(),
               dense_frac_win_max_pct=100.0 * float(frac_w.max()),
               dense_volume_mm3=float(nd.sum() * vox),
               dense_win_max_k=int(sl[0].start + int(np.argmax(frac_w))) if ok.any() else None)
    out["qc_density_warning"] = bool(out["dense_frac_pct"] > 100 * DENSE_FRAC_MAX
                                     or out["dense_frac_win_max_pct"] > 100 * DENSE_WIN_FRAC_MAX)
    return out


def characterize(bone_raw, sp, k1, k2, tibia_extent=None, reference=None, ct=None):
    """
    bone_raw: segmentación (bool) en la ROI; sp: spacing (k,j,i) mm;
    k1 < k2: límites del usuario (índices enteros en el eje largo).
    reference: máscara de referencia ya calculada (opcional; p. ej. contralateral).
    ct: TC en HU en la misma grilla (opcional) -> control de calidad por densidad.
    Devuelve (resultado dict, máscara defecto, máscara referencia en [k1,k2]).
    """
    k1, k2 = int(min(k1, k2)), int(max(k1, k2))
    bone = postprocess_bone(bone_raw)
    qc = {}
    ref = interpolated_reference(bone, k1, k2, sp, qc) if reference is None else reference.copy()
    window = np.zeros(bone.shape[0], bool); window[k1:k2 + 1] = True
    ref[~window] = False
    defect = np.zeros_like(ref)
    loss_k = np.full(bone.shape[0], np.nan)
    for k in range(k1, k2 + 1):
        defect[k], loss_k[k] = angular_defect_slice(ref[k], bone[k], sp[1:])

    if CONNECTED_DEFECT and defect.any():
        lab, n = ndi.label(defect, structure=np.ones((3, 3, 3)))
        if n > 1:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            defect = lab == sz.argmax()
    area_ref = ref.sum(axis=(1, 2)).astype(float)
    area_def = defect.sum(axis=(1, 2)).astype(float)
    frac = np.divide(area_def, area_ref, out=np.zeros_like(area_def), where=area_ref > 0)
    flags = (frac >= SLICE_DEFECT_FRAC) & window
    gap = int(round(GAP_TOL_MM / sp[0]))
    segs = merge_runs(runs(flags), gap)

    res = dict(k1=k1, k2=k2, window_mm=(k2 - k1 + 1) * sp[0], n_segments=len(segs), **qc)
    if qc:
        res["qc_warning"] = bool(max(qc["ref_band_shift_prox_mm"], qc["ref_band_shift_dist_mm"]) > 2.0
                                 or not (qc["ref_band_found_prox"] and qc["ref_band_found_dist"]))
    vox = float(np.prod(sp))
    if not segs:
        res.update(defect_detected=False, length_mm=0.0, volume_mm3=0.0,
                   cortical_loss_max_pct=0.0, cortical_loss_mean_pct=0.0)
    else:
        a, b = max(segs, key=lambda s: s[1] - s[0])
        loss = loss_k[a:b + 1]
        res.update(defect_detected=True, k_start=a, k_end=b,
                   length_mm=(b - a + 1) * sp[0],
                   volume_mm3=float(defect[a:b + 1].sum() * vox),
                   volume_total_window_mm3=float(defect.sum() * vox),
                   cortical_loss_max_pct=float(np.nanmax(loss)),
                   cortical_loss_mean_pct=float(np.nanmean(loss)),
                   ref_volume_mm3=float(ref[a:b + 1].sum() * vox))
        if ct is not None:
            main = np.zeros_like(defect); main[a:b + 1] = defect[a:b + 1]
            res.update(density_qc(main, ct, bone, sp))
            res["qc_warning"] = bool(res.get("qc_warning", False) or res["qc_density_warning"])
    if tibia_extent is not None and res.get("defect_detected"):
        t0, t1 = tibia_extent
        L = t1 - t0
        lo, hi = t0 + END_EXCLUSION_FRAC * L, t1 - END_EXCLUSION_FRAC * L
        res["diaphyseal"] = bool(res["k_start"] >= lo and res["k_end"] <= hi)
    res["meets_criteria"] = bool(
        res.get("defect_detected")
        and CRITERIA["length_min_mm"] - sp[0] / 2 <= res["length_mm"] <= CRITERIA["length_max_mm"] + sp[0] / 2
        and res["cortical_loss_max_pct"] > CRITERIA["cortical_loss_min_pct"]
        and res["n_segments"] == 1
        and res.get("diaphyseal", True))
    return res, defect, ref


def tibia_extent_k(mask):
    ks = np.nonzero(mask.any(axis=(1, 2)))[0]
    return int(ks.min()), int(ks.max())


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
