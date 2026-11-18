"""
Defectos óseos segmentarios VIRTUALES con verdad conocida, sobre tibias intactas
(ROI orientada; eje k = longitudinal, de proximal a distal).

Configuraciones (dentro de los criterios del proyecto):
  - longitud L = 30, 50, 70 mm
  - tipo "circ": resección circunferencial (pérdida cortical 100 %)
         "parc": resección de un sector de 240° (pérdida cortical ~67 %)
  - centro en el 45 % o 55 % de la longitud de la tibia (diáfisis)
  - "ctrl": sin defecto (control negativo; se simulan clics en la diáfisis)

Simulación en la TC: dentro del bloque [k_start, k_end] y del sector angular
(respecto del centroide de la sección), los vóxeles de la tibia rellena
(dilatada 2 vóxeles en el plano para eliminar el borde de volumen parcial) se
reemplazan por tejido blando: N(media, DE) estimadas de tejido blando
peri-óseo (3–15 mm del hueso, -100..150 HU) en los mismos cortes.

Clics simulados del usuario: 5 mm fuera de cada extremo del defecto (el
usuario encierra el defecto con un poco de hueso sano).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

LENGTHS_MM = (30, 50, 70)
TYPES = {"circ": 360.0, "parc": 240.0}
CLICK_MARGIN_MM = 5.0
CTRL_WINDOW_MM = 50.0
SECTOR_DIRECTION_DEG = 0.0
SEED = 20261020


def configs():
    out = []
    i = 0
    for L in LENGTHS_MM:
        for t in TYPES:
            out.append(dict(name=f"{t}{L}", kind=t, length_mm=float(L),
                            center_frac=0.45 if i % 2 == 0 else 0.55))
            i += 1
    out.append(dict(name="ctrl", kind="ctrl", length_mm=0.0, center_frac=0.50))
    return out


def sector_mask(shape_ji, centroid_ji, sp_ji, width_deg, direction_deg=SECTOR_DIRECTION_DEG):
    if width_deg >= 360:
        return np.ones(shape_ji, bool)
    jj, ii = np.mgrid[:shape_ji[0], :shape_ji[1]]
    ang = np.degrees(np.arctan2((jj - centroid_ji[0]) * sp_ji[0], (ii - centroid_ji[1]) * sp_ji[1]))
    d = (ang - direction_deg + 180) % 360 - 180
    return np.abs(d) <= width_deg / 2


def make_defect(intact, sp, cfg, ct=None, rng=None):
    """
    intact: tibia rellena (bool, ROI). Devuelve dict con removal (bool),
    k_start, k_end, clicks (k1,k2) y, si se da ct, la TC con el defecto.
    """
    ks = np.nonzero(intact.any(axis=(1, 2)))[0]
    t0, t1 = ks.min(), ks.max()
    center = int(round(t0 + cfg["center_frac"] * (t1 - t0)))
    margin = int(round(CLICK_MARGIN_MM / sp[0]))
    out = dict(config=cfg["name"], kind=cfg["kind"], tibia_k=(int(t0), int(t1)))

    if cfg["kind"] == "ctrl":
        half = int(round(CTRL_WINDOW_MM / sp[0] / 2))
        out.update(removal=np.zeros_like(intact), k_start=None, k_end=None,
                   clicks=(center - half, center + half), ct=ct)
        return out

    nk = int(round(cfg["length_mm"] / sp[0]))
    a = center - nk // 2
    b = a + nk - 1
    removal = np.zeros_like(intact)
    grow = ndi.generate_binary_structure(2, 1)
    for k in range(a, b + 1):
        sec = intact[k]
        if not sec.any():
            continue
        cj, ci = ndi.center_of_mass(sec)
        reg = ndi.binary_dilation(sec, structure=grow, iterations=2)
        removal[k] = reg & sector_mask(sec.shape, (cj, ci), sp[1:], TYPES[cfg["kind"]])

    out.update(removal=removal, k_start=int(a), k_end=int(b),
               clicks=(int(a - margin), int(b + margin)))

    if ct is not None:
        rng = rng or np.random.default_rng(SEED)
        ct_def = ct.copy()
        slab = intact[a:b + 1]
        dist = ndi.distance_transform_edt(~slab, sampling=(1e6, sp[1], sp[2]))  # solo en el plano
        ring = (dist >= 3) & (dist <= 15)
        vals = ct[a:b + 1][ring]
        vals = vals[(vals > -100) & (vals < 150)]
        mu, sd = (float(vals.mean()), float(vals.std())) if vals.size > 100 else (40.0, 15.0)
        n = int(removal.sum())
        ct_def[removal] = np.clip(rng.normal(mu, sd, n), -100, 150).astype(ct.dtype)
        out.update(ct=ct_def, soft_tissue_mu=mu, soft_tissue_sd=sd)
    return out
