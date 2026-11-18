"""
Post-proceso físico del defecto (--defect-refine). Módulo nuevo: no cambia defect_core.py.

Motivo: la caracterización declara ausente un sector angular cuando la SEGMENTACIÓN no cubre su
contorno, y entonces toda la cuña (hasta el centroide, canal incluido) entra al defecto. Si el modelo
sub-segmenta cortical existente, aparece una cuña falsa. Las reglas usan solo la TC y la forma de
referencia ya calculada por defect_core; ningún parámetro sale de las referencias manuales de pacientes.

Reglas (se pueden activar por separado)
  sector (R2)  sector presente por densidad: un sector ausente cuyo contorno de referencia tiene, a
               <= COVER_TOL_MM, vóxeles con hu_thr <= HU < METAL_HU (a > DENSE_EXCL_MM del metal)
               no está ausente: hay hueso y la segmentación lo perdió. Se quita la cuña completa.
  voxel  (R1)  el defecto es hueso ausente: se quitan los vóxeles con HU >= hu_thr.
  Después de cualquiera de ellas se conserva la componente 3D principal (26-vecindad).

Constantes: COVER_TOL_MM, METAL_HU, DENSE_EXCL_MM, N_BINS y SLICE_DEFECT_FRAC son las de defect_core.py,
fijadas antes de este trabajo. HU_THR_DEFAULT = 600 se fijó a priori y no se ajusta.
Limitación física: HU >= 600 no distingue hueso de cuerpos extraños densos (cemento, perlas, metal bajo
el umbral de metal), ni hueso huésped de fragmentos libres.

Dependencias: numpy, scipy, defect_core.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defect_core as dc  # noqa: E402

HU_THR_DEFAULT = 600.0          # fijado a priori
RULES = ("sector", "voxel", "sector+voxel")


def _bins(ref_s, sp_ji, n_bins):
    cj, ci = ndi.center_of_mass(ref_s)
    jj, ii = np.nonzero(ref_s)
    ang = np.arctan2((jj - cj) * sp_ji[0], (ii - ci) * sp_ji[1])
    return jj, ii, ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins


def _sector_keep_slice(ref_s, def_s, dense_s, sp_ji, n_bins=dc.N_BINS):
    """Vóxeles de def_s que siguen ausentes tras mirar la densidad del contorno. Mismo binning
    angular y mismo relleno de bins sin contorno que dc.angular_defect_slice."""
    if not dense_s.any():
        return def_s.copy()
    jj, ii, bins = _bins(ref_s, sp_ji, n_bins)
    contour = ref_s & ~ndi.binary_erosion(ref_s)
    covered = contour & (ndi.distance_transform_edt(~dense_s, sampling=sp_ji) <= dc.COVER_TOL_MM)
    on_contour, cov = contour[jj, ii], covered[jj, ii]
    present = np.full(n_bins, np.nan)
    for b in np.unique(bins[on_contour]):
        sel = on_contour & (bins == b)
        present[b] = 1.0 if cov[sel].any() else 0.0
    if np.isnan(present).all():
        return def_s.copy()
    idx = np.nonzero(~np.isnan(present))[0]
    for b in np.nonzero(np.isnan(present))[0]:
        dist = np.minimum(np.abs(idx - b), n_bins - np.abs(idx - b))
        present[b] = present[idx[np.argmin(dist)]]
    binmap = np.full(ref_s.shape, -1, int)
    binmap[jj, ii] = bins
    keep = def_s.copy()
    sel = def_s & (binmap >= 0)
    keep[sel] = present[binmap[sel]] == 0
    return keep


def dense_mask(ct, sp, hu_thr, sl, metal_excl_mm=dc.DENSE_EXCL_MM):
    """Densidad ósea en el recorte sl: hu_thr <= HU < METAL_HU, a > metal_excl_mm del metal."""
    c = ct[sl]
    metal = c >= dc.METAL_HU
    d = (c >= hu_thr) & ~metal
    if metal.any():
        d &= ndi.distance_transform_edt(~metal, sampling=sp) > metal_excl_mm
    return d


def host_dense(ct, sp, hu_thr, bone, k0, k1, mass_r_mm=None, pad_mm=15.0, metal_excl_mm=dc.DENSE_EXCL_MM):
    """Densidad ósea atribuible al huésped en los cortes k0..k1 (bool, mismo alto que esos cortes).
    Hueso sub-segmentado = cortical contigua al hueso segmentado. Lo denso que no toca al huésped
    (cuerpos extraños, perlas, fragmentos libres) no cuenta. mass_r_mm: además quita masas densas más
    gruesas que 2·r (espaciador de cemento, que sí toca los extremos del huésped)."""
    pad = int(np.ceil(pad_mm / sp[0]))
    a, b = max(k0 - pad, 0), min(k1 + pad + 1, ct.shape[0])
    sl = (slice(a, b), slice(None), slice(None))
    d = dense_mask(ct, sp, hu_thr, sl, metal_excl_mm)
    if mass_r_mm:
        core = ndi.distance_transform_edt(d, sampling=sp) > mass_r_mm
        if core.any():
            d &= ndi.distance_transform_edt(~core, sampling=sp) > mass_r_mm
    if bone is not None:
        lab, n = ndi.label(d | bone[sl], structure=np.ones((3, 3, 3)))
        ids = np.unique(lab[bone[sl]])
        d &= np.isin(lab, ids[ids > 0])
    return d[k0 - a:k1 - a + 1]


def main_component(m):
    if not m.any():
        return m
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return m
    s = np.bincount(lab.ravel())
    s[0] = 0
    return lab == s.argmax()


def summarize(defect, ref, sp):
    """Longitud/volumen con los mismos criterios de corte y tramo que dc.characterize."""
    area_r = ref.sum(axis=(1, 2)).astype(float)
    frac = np.divide(defect.sum(axis=(1, 2)).astype(float), area_r, out=np.zeros_like(area_r), where=area_r > 0)
    segs = dc.merge_runs(dc.runs(frac >= dc.SLICE_DEFECT_FRAC), int(round(dc.GAP_TOL_MM / sp[0])))
    out = dict(n_segments=len(segs), volume_total_mm3=float(defect.sum() * np.prod(sp)), defect_detected=bool(segs))
    if segs:
        a, b = max(segs, key=lambda s: s[1] - s[0])
        out.update(k_start=int(a), k_end=int(b), length_mm=float((b - a + 1) * sp[0]),
                   volume_mm3=float(defect[a:b + 1].sum() * np.prod(sp)))
    else:
        out.update(length_mm=0.0, volume_mm3=0.0)
    return out


def refine(defect, ct, sp, rule="sector+voxel", reference=None, hu_thr=HU_THR_DEFAULT, bone=None, mass_r_mm=None,
           metal_excl_mm=dc.DENSE_EXCL_MM):
    """
    defect: máscara del sistema; ct: TC en HU (misma grilla); sp: spacing (k,j,i); reference: forma de
    referencia del sistema (obligatoria para 'sector'); bone: hueso segmentado (opcional): activa la
    restricción a densidad contigua al huésped; mass_r_mm: quita masas densas. Devuelve (defecto, informe).
    """
    if rule not in RULES:
        raise ValueError(f"regla desconocida: {rule} (opciones {RULES})")
    if ct.shape != defect.shape:
        raise RuntimeError("La TC no está en la grilla del defecto")
    defect = defect.astype(bool)
    d = defect.copy()
    vox = float(np.prod(sp))
    info = dict(regla=rule, hu_umbral=float(hu_thr), restringe_huesped=bone is not None, radio_masa_mm=mass_r_mm, V_inicial_mm3=float(defect.sum() * vox),
                metal_hu=dc.METAL_HU, tol_contorno_mm=dc.COVER_TOL_MM, excl_metal_mm=float(metal_excl_mm))
    if "sector" in rule:
        if reference is None:
            raise ValueError("La regla 'sector' necesita la forma de referencia del sistema")
        reference = reference.astype(bool)
        ks = np.nonzero(defect.any(axis=(1, 2)))[0]
        k0, k1 = int(ks.min()), int(ks.max())
        dense = host_dense(ct, sp, hu_thr, bone, k0, k1, mass_r_mm, metal_excl_mm=metal_excl_mm)
        for i, k in enumerate(range(k0, k1 + 1)):
            if d[k].any() and reference[k].any():
                d[k] = _sector_keep_slice(reference[k], d[k], dense[i], sp[1:])
        info["V_quitado_sector_mm3"] = float((defect & ~d).sum() * vox)
    n = d.sum()
    if "voxel" in rule:
        if bone is None and mass_r_mm is None:
            d &= ct < hu_thr
        else:                                    # con restricción: solo la densidad del huésped, no el metal ni cuerpos extraños
            ks = np.nonzero(d.any(axis=(1, 2)))[0]
            k0, k1 = int(ks.min()), int(ks.max())
            d[k0:k1 + 1] &= ~host_dense(ct, sp, hu_thr, bone, k0, k1, mass_r_mm, metal_excl_mm=metal_excl_mm)
        info["V_quitado_voxel_mm3"] = float((n - d.sum()) * vox)
    n = d.sum()
    d = main_component(d)
    info["V_quitado_componentes_mm3"] = float((n - d.sum()) * vox)
    info["V_final_mm3"] = float(d.sum() * vox)
    if reference is not None:
        info["resumen"] = summarize(d, reference, sp)
    return d, info
