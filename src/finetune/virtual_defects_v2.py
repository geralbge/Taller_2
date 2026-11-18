"""
Generador v2 de defectos óseos segmentarios VIRTUALES "realistas" sobre tibias
intactas (ROI orientada; array (k, j, i), k = eje longitudinal, proximal -> distal).

Sigue siendo un dato SINTÉTICO con verdad conocida. Los rangos de parámetros de
SCENARIOS son supuestos de trabajo, porque todavía no están respaldados uno a uno
con literatura. Cada rango está marcado y se ajusta al terminar la revisión de F1.

Escenarios (por estado de la TC, no por enfermedad):
  A  post-trauma / desbridamiento agudo: extremos oblicuos, espiroideos o
     conminutos, fragmentos libres, hematoma y gas, fíbula fracturada,
     fijador externo, clavo o placa.
  B  infección desbridada: extremos regularizados, esclerosis, secuestro
     opcional, espaciador de cemento opcional (Masquelet, etapa 1),
     osteopenia y líquido.
  C  no unión establecida con pérdida ósea: extremos afilados (atrófica) o
     redondeados con callo (hipertrófica), canal sellado, esclerosis,
     osteopenia, clavo intramedular con cerrojos y desalineación.
  D  resección limpia (= v1): cortes transversales planos.

Convención de etiqueta (label_policy):
  "host"     (por defecto) etiqueta = hueso huésped viable: hueso remanente,
             callo, tapón de canal y engrosamiento endóstico = 1; fragmentos
             libres, secuestro, metal y cemento = 0.
  "all_bone" fragmentos y secuestro también = 1 (metal y cemento siguen en 0).

Salidas de make_defect_v2():
  ct        TC con defecto (mismo dtype)
  label     etiqueta de entrenamiento (uint8)
  removal   hueso intacto resecado (bool), que es la verdad para la caracterización
  aux       mapa de componentes para control de calidad (uint8, ver AUX)
  clicks    (k1, k2), 5 mm fuera del extremo más alejado de cada lado
  meta      todos los parámetros muestreados (JSON serializable)

Dependencias: numpy, scipy.  Unidades: mm y HU.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

SEED = 20261103
CLICK_MARGIN_MM = 5.0
METAL_HU = 3071.0              # valor máximo observado en pat_01 por saturación (clavo); representa el artefacto, no la densidad física
AUX = dict(fragment=1, sequestrum=2, metal=3, spacer=4, callus=5, cap=6, gas=7, involucrum=8)

# ----------------------------------------------------------------------------
# Rangos por escenario.  (lo, hi) = uniforme; p_* = probabilidad.
# [SUPUESTO] = rango de trabajo pendiente de respaldo bibliográfico (F1).
# Convención de densidades (HU):
#   *_hu   = valor ABSOLUTO asignado al vóxel: callo, involucro, cemento, contenido del defecto.
#   *_dhu  = INCREMENTO sumado al valor original de cada vóxel de la misma TC:
#            sclerosis_dhu -> hueso huésped cerca del extremo (decae con la distancia);
#            fragment_dhu  -> vóxeles del fragmento, tomados de la cortical resecada;
#            secuestro     -> +150 a +300 sobre los vóxeles de cortical resecada de los que sale.
#   METAL_HU = valor máximo de la imagen (3071 = techo del rango de 12 bits), no la densidad física del metal.
# ----------------------------------------------------------------------------
SCENARIOS = {
    "A": dict(
        length_mm=(20, 80),                 # [SUPUESTO] entrenamiento; prueba 25-70
        p_partial=0.3, sector_deg=(180, 300),
        obliquity_deg=(0, 45), p_spiral=0.3, spiral_pitch_mm=(10, 40),
        jagged_mm=(1.0, 5.0),               # [SUPUESTO] amplitud de conminución
        end_shape=("flat",), taper_mm=(0, 0),
        p_fragments=0.6, n_fragments=(1, 4), fragment_shift_mm=(1, 6),
        p_wedge=0.6, n_wedge=(1, 2), wedge_len_mm=(20, 50), wedge_arc_deg=(60, 120), wedge_shift_mm=(2, 10),  # img38, img32, img43
        content="hematoma", p_gas=0.4, n_gas=(1, 6),
        p_fibula_fx=0.8,                    # img32, img43: fíbula rota a la misma altura
        implant_probs=dict(exfix=0.45, ring=0.2, nail=0.2, plate=0.1, none=0.05),
        p_sclerosis=0.0, p_cap=0.0, p_osteopenia=0.1, p_callus=0.0,
        p_sequestrum=0.0, p_spacer=0.0, p_malalign=0.4,
        angulation_deg=(0, 12), translation_mm=(0, 10),   # img38, img43
    ),
    "B": dict(
        length_mm=(20, 80),
        p_partial=0.3, sector_deg=(180, 300),
        obliquity_deg=(0, 10), p_spiral=0.0, spiral_pitch_mm=(0, 0),
        jagged_mm=(0.0, 1.0),
        end_shape=("flat",), taper_mm=(0, 0),
        p_fragments=0.0, n_fragments=(0, 0), fragment_shift_mm=(0, 0),
        content="fluid", p_gas=0.1, n_gas=(1, 3),
        p_fibula_fx=0.2,
        implant_probs=dict(exfix=0.3, ring=0.3, nail=0.2, plate=0.15, none=0.05),  # img24-25, img52
        p_sclerosis=0.7, sclerosis_len_mm=(5, 20), sclerosis_dhu=(150, 450),
        p_lysis=0.5, n_lysis=(1, 5),        # img24: esclerosis mezclada con lisis
        p_involucrum=0.3, involucrum_len_mm=(20, 60), involucrum_thick_mm=(2, 5), involucrum_hu=(600, 1200),  # img7 (Desimpel)
        p_cloaca=0.3,                       # img9 (Desimpel)
        p_cap=0.3, cap_len_mm=(2, 6),
        p_osteopenia=0.6, osteopenia_frac=(0.10, 0.30),
        p_callus=0.0,
        p_sequestrum=0.3, p_spacer=0.5, spacer_hu=(1000, 1800),  # PMMA puro 1328 ± 27 HU (Verburg et al. 2014, BMC Med Imaging 14:29); rango ampliado por marca/antibiótico/escáner [decisión de modelado]
        spacer_overlap_mm=(5, 20), spacer_margin_mm=(3, 10), p_spacer_rod=0.4,  # img20, img43, img59b
        p_malalign=0.1, angulation_deg=(0, 5), translation_mm=(0, 3),
    ),
    "C": dict(
        length_mm=(20, 80),
        p_partial=0.3, sector_deg=(180, 300),
        obliquity_deg=(0, 20), p_spiral=0.0, spiral_pitch_mm=(0, 0),
        jagged_mm=(0.0, 1.5),
        end_shape=("irregular", "irregular", "pointed", "rounded"), taper_mm=(5, 20),  # img19, img59a
        p_fragments=0.5, n_fragments=(1, 3), fragment_shift_mm=(0, 3),  # fragmentos residuales (img19; Çalışır fig 7)
        fragment_dhu=(100, 300),            # fragmentos escleróticos, más densos (Çalışır fig 7a)
        content="fibrous", p_gas=0.0, n_gas=(0, 0),
        p_fibula_fx=0.4, p_fibula_old=1.0,  # fractura antigua de fíbula con callo (Çalışır fig 8c)
        implant_probs=dict(nail=0.5, plate=0.35, exfix=0.05, ring=0.05, none=0.05),  # img19, img59a, img56
        p_broken=0.2,                       # img19: placa rota
        p_sclerosis=0.8, sclerosis_len_mm=(5, 20), sclerosis_dhu=(200, 500),
        p_cap=0.8, cap_len_mm=(2, 8),
        p_osteopenia=0.7, osteopenia_frac=(0.10, 0.30),
        p_callus=0.5, callus_len_mm=(8, 20), callus_thick_mm=(2, 6), callus_hu=(300, 1000),  # brecha en consolidación: 216-514 HU a las 8-12 sem (Grigoryan et al. 2003, Acad Radiol 10:1267, tabla 4); techo = callo maduro [decisión de modelado]
        p_sequestrum=0.0, p_spacer=0.0,
        p_malalign=0.5, angulation_deg=(0, 10), translation_mm=(0, 5),
    ),
    "D": dict(
        length_mm=(20, 80),
        p_partial=0.5, sector_deg=(150, 300),
        obliquity_deg=(0, 0), p_spiral=0.0, spiral_pitch_mm=(0, 0),
        jagged_mm=(0.0, 0.0),
        end_shape=("flat",), taper_mm=(0, 0),
        p_fragments=0.0, n_fragments=(0, 0), fragment_shift_mm=(0, 0),
        content="soft", p_gas=0.0, n_gas=(0, 0),
        p_fibula_fx=0.0,
        implant_probs=dict(none=1.0),
        p_sclerosis=0.0, p_cap=0.0, p_osteopenia=0.0, p_callus=0.0,
        p_sequestrum=0.0, p_spacer=0.0, p_malalign=0.0,
        angulation_deg=(0, 0), translation_mm=(0, 0),
    ),
}

CONTENT_HU = dict(soft=None, hematoma=(60.0, 12.0), fluid=(15.0, 10.0), fibrous=(70.0, 15.0))


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------
def _u(rng, lohi):
    lo, hi = lohi
    return float(lo) if hi == lo else float(rng.uniform(lo, hi))


def _choice(rng, probs: dict):
    keys = list(probs); p = np.array([probs[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def fill_tibia(lbl):
    """Tibia rellena (canal incluido): relleno 2D por corte + 3D."""
    m = lbl > 0
    out = np.zeros_like(m)
    for k in np.nonzero(m.any(axis=(1, 2)))[0]:
        out[k] = ndi.binary_fill_holes(m[k])
    return ndi.binary_fill_holes(out)


def centroid_line(mask):
    """Centroide (j, i) por corte, interpolado y suavizado donde falta."""
    nk = mask.shape[0]
    c = np.full((nk, 2), np.nan)
    for k in np.nonzero(mask.any(axis=(1, 2)))[0]:
        c[k] = ndi.center_of_mass(mask[k])
    ok = ~np.isnan(c[:, 0])
    for d in range(2):
        c[:, d] = np.interp(np.arange(nk), np.nonzero(ok)[0], c[ok, d])
        c[:, d] = ndi.uniform_filter1d(c[:, d], 9, mode="nearest")
    return c


def sample_params(scenario, rng, tibia_len_mm):
    s = SCENARIOS[scenario]
    p = dict(scenario=scenario)
    p["length_mm"] = _u(rng, s["length_mm"])
    p["center_frac"] = float(rng.uniform(0.30, 0.70))
    p["sector_deg"] = _u(rng, s["sector_deg"]) if rng.random() < s["p_partial"] else 360.0
    p["sector_dir_deg"] = float(rng.uniform(-180, 180))
    for end in ("prox", "dist"):
        e = dict(obliquity_deg=_u(rng, s["obliquity_deg"]),
                 oblique_dir_deg=float(rng.uniform(-180, 180)),
                 spiral_pitch_mm=_u(rng, s["spiral_pitch_mm"]) if rng.random() < s["p_spiral"] else 0.0,
                 spiral_phase_deg=float(rng.uniform(-180, 180)),
                 jagged_mm=_u(rng, s["jagged_mm"]),
                 jagged_phases=[float(x) for x in rng.uniform(0, 2 * np.pi, 7)],
                 shape=str(rng.choice(s["end_shape"])),
                 taper_mm=_u(rng, s["taper_mm"]))
        if e["shape"] in ("flat", "irregular"):
            e["taper_mm"] = 0.0
        if e["shape"] == "irregular":       # extremo escleroso e irregular de la no unión
            e["jagged_mm"] = float(rng.uniform(1.0, 3.0))
            e["obliquity_deg"] = float(rng.uniform(0, 25))
        p[end] = e
    p["sector_edge_jitter_deg"] = float(rng.uniform(0, 15)) if p["sector_deg"] < 360 else 0.0
    p["sector_edge_phases"] = [float(x) for x in rng.uniform(0, 2 * np.pi, 4)]
    p["fragments"] = int(rng.integers(s["n_fragments"][0], s["n_fragments"][1] + 1)) if rng.random() < s["p_fragments"] else 0
    p["fragment_shift_mm"] = s["fragment_shift_mm"]
    p["fragment_dhu"] = _u(rng, s["fragment_dhu"]) if "fragment_dhu" in s else 0.0
    p["wedges"] = []
    if rng.random() < s.get("p_wedge", 0.0):
        for _ in range(int(rng.integers(s["n_wedge"][0], s["n_wedge"][1] + 1))):
            p["wedges"].append(dict(len_mm=_u(rng, s["wedge_len_mm"]), arc_deg=_u(rng, s["wedge_arc_deg"]),
                                    dir_deg=float(rng.uniform(-180, 180)), pos=float(rng.uniform(0.2, 0.8)),
                                    shift_mm=_u(rng, s["wedge_shift_mm"]), shift_dir_deg=float(rng.uniform(-180, 180)),
                                    tilt_deg=float(rng.uniform(-10, 10))))
    p["lysis"] = int(rng.integers(s["n_lysis"][0], s["n_lysis"][1] + 1)) if rng.random() < s.get("p_lysis", 0) else 0
    p["involucrum"] = bool(rng.random() < s.get("p_involucrum", 0))
    if p["involucrum"]:
        p["involucrum_len_mm"] = _u(rng, s["involucrum_len_mm"]); p["involucrum_thick_mm"] = _u(rng, s["involucrum_thick_mm"])
        p["involucrum_hu"] = _u(rng, s["involucrum_hu"]); p["involucrum_side"] = str(rng.choice(["prox", "dist", "both"]))
    p["cloaca"] = bool(rng.random() < s.get("p_cloaca", 0))
    p["broken"] = bool(rng.random() < s.get("p_broken", 0))
    p["content"] = s["content"]
    p["n_gas"] = int(rng.integers(s["n_gas"][0], s["n_gas"][1] + 1)) if rng.random() < s["p_gas"] else 0
    p["fibula_fx"] = bool(rng.random() < s["p_fibula_fx"])
    p["fibula_old"] = p["fibula_fx"] and bool(rng.random() < s.get("p_fibula_old", 0))
    p["implant"] = _choice(rng, s["implant_probs"])
    p["sclerosis"] = bool(rng.random() < s["p_sclerosis"])
    if p["sclerosis"]:
        p["sclerosis_len_mm"] = _u(rng, s["sclerosis_len_mm"]); p["sclerosis_dhu"] = _u(rng, s["sclerosis_dhu"])
    p["cap"] = bool(rng.random() < s["p_cap"])
    if p["cap"]:
        p["cap_len_mm"] = _u(rng, s["cap_len_mm"])
    p["osteopenia"] = bool(rng.random() < s["p_osteopenia"])
    if p["osteopenia"]:
        p["osteopenia_frac"] = _u(rng, s.get("osteopenia_frac", (0.1, 0.3)))
        p["osteopenia_lambda_mm"] = float(rng.uniform(50, 150))
    p["callus"] = bool(rng.random() < s["p_callus"])
    if p["callus"]:
        p["callus_len_mm"] = _u(rng, s["callus_len_mm"]); p["callus_thick_mm"] = _u(rng, s["callus_thick_mm"])
        p["callus_hu"] = _u(rng, s["callus_hu"])
    p["sequestrum"] = bool(rng.random() < s["p_sequestrum"])
    p["spacer"] = bool(rng.random() < s["p_spacer"])
    if p["spacer"]:
        p["spacer_hu"] = _u(rng, s["spacer_hu"]); p["spacer_overlap_mm"] = _u(rng, s["spacer_overlap_mm"])
        p["spacer_margin_mm"] = _u(rng, s["spacer_margin_mm"])
        p["spacer_rod"] = bool(rng.random() < s.get("p_spacer_rod", 0)) and p["implant"] != "nail"
    # con clavo intramedular no se simula desalineación (el clavo recto la impide)
    p["malalign"] = bool(rng.random() < s["p_malalign"]) and p["implant"] != "nail" and not p.get("spacer", False)
    if p["malalign"]:
        p["angulation_deg"] = _u(rng, s["angulation_deg"]); p["angulation_dir_deg"] = float(rng.uniform(-180, 180))
        p["translation_mm"] = _u(rng, s["translation_mm"]); p["translation_dir_deg"] = float(rng.uniform(-180, 180))
    return p


# ----------------------------------------------------------------------------
# superficie de corte de cada extremo
# ----------------------------------------------------------------------------
def _end_offset(e, x, y, r_norm, theta, side):
    """
    Desplazamiento (mm, a lo largo de k) de la superficie de corte respecto del
    plano transversal nominal.  side = -1 extremo proximal, +1 distal.
    Positivo = hacia distal.
    """
    a = np.radians(e["oblique_dir_deg"])
    f = np.tan(np.radians(e["obliquity_deg"])) * (x * np.cos(a) + y * np.sin(a))
    if e["spiral_pitch_mm"] > 0:
        t = (theta - np.radians(e["spiral_phase_deg"]) + np.pi) % (2 * np.pi) - np.pi
        f = f + e["spiral_pitch_mm"] * t / (2 * np.pi)
    if e["jagged_mm"] > 0:
        ph = e["jagged_phases"]
        g = sum(np.cos(m * theta + ph[m - 2]) / m for m in range(2, 9))
        f = f + e["jagged_mm"] * g / 1.2
    if e["taper_mm"] > 0:
        rn = np.clip(r_norm, 0, 1)
        prof = rn if e["shape"] == "pointed" else 1 - np.sqrt(1 - rn ** 2)
        # el hueso del huésped retrocede en la periferia: el extremo queda en punta o en cúpula
        f = f + side * e["taper_mm"] * prof
    return f


def _defect_region(intact, sp, p, cl):
    """Región resecada del hueso intacto (bool) y extensión en k."""
    ks = np.nonzero(intact.any(axis=(1, 2)))[0]
    t0, t1 = ks.min(), ks.max()
    zc = (t0 + p["center_frac"] * (t1 - t0)) * sp[0]
    zp, zd = zc - p["length_mm"] / 2, zc + p["length_mm"] / 2
    reach = 60.0  # mm; cubre oblicuidad, espiral y conicidad
    k_lo = max(int((zp - reach) / sp[0]), t0)
    k_hi = min(int((zd + reach) / sp[0]) + 1, t1)
    nj, ni = intact.shape[1:]
    jj, ii = np.mgrid[:nj, :ni]
    grow = ndi.generate_binary_structure(2, 1)
    removal = np.zeros_like(intact)
    for k in range(k_lo, k_hi + 1):
        sec = intact[k]
        if not sec.any():
            continue
        y = (jj - cl[k, 0]) * sp[1]; x = (ii - cl[k, 1]) * sp[2]
        r = np.hypot(x, y); th = np.arctan2(y, x)
        R = np.sqrt(sec.sum() * sp[1] * sp[2] / np.pi)
        z = k * sp[0]
        cut_p = zp + _end_offset(p["prox"], x, y, r / R, th, -1)
        cut_d = zd + _end_offset(p["dist"], x, y, r / R, th, +1)
        inside = (z >= cut_p) & (z <= cut_d)
        if not inside.any():
            continue
        reg = ndi.binary_dilation(sec, structure=grow, iterations=2)
        if p["sector_deg"] < 360:
            d = (np.degrees(th) - p["sector_dir_deg"] + 180) % 360 - 180
            ph = p.get("sector_edge_phases", [0, 0, 0, 0]); jit = p.get("sector_edge_jitter_deg", 0.0)
            zz = k * sp[0] / 15.0
            half = p["sector_deg"] / 2 + jit * (np.sin(zz + ph[0]) * 0.6 + np.sin(2.3 * zz + ph[1]) * 0.4) * np.sign(d) \
                + jit * 0.5 * np.sin(np.radians(d) * 3 + ph[2])
            inside &= np.abs(d) <= half
        removal[k] = reg & inside
    kk = np.nonzero(removal.any(axis=(1, 2)))[0]
    return removal, (int(kk.min()), int(kk.max())) if kk.size else (None, None), (zp, zd)


# ----------------------------------------------------------------------------
# elementos
# ----------------------------------------------------------------------------
def _soft_stats(ct, intact, sp, a, b):
    slab = intact[a:b + 1]
    dist = ndi.distance_transform_edt(~slab, sampling=(1e6, sp[1], sp[2]))
    vals = ct[a:b + 1][(dist >= 3) & (dist <= 15)]
    # solo músculo (10-100 HU): la DE de la mezcla grasa+músculo (~45 HU) mide heterogeneidad de tejidos,
    # no el ruido del escáner (~15-20 HU en músculo), y dejaba el relleno más granulado que su entorno
    vals = vals[(vals > 10) & (vals < 100)]
    return (float(vals.mean()), float(vals.std())) if vals.size > 100 else (40.0, 15.0)


def _ball(shape, center, radius_mm, sp):
    c = np.asarray(center, float)
    lo = np.maximum((c - radius_mm / sp - 1).astype(int), 0)
    hi = np.minimum((c + radius_mm / sp + 2).astype(int), shape)
    g = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    d2 = sum(((g[i] - c[i]) * sp[i]) ** 2 for i in range(3))
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = d2 <= radius_mm ** 2
    return m


def _cylinder(shape, p0, direction, radius_mm, half_len_mm, sp):
    """Cilindro en coordenadas físicas (mm); p0 en índices (k, j, i), direction en mm (k, j, i)."""
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    c = np.asarray(p0, float) * sp
    ext = np.abs(d) * half_len_mm + radius_mm
    lo = np.maximum(((c - ext) / sp - 1).astype(int), 0)
    hi = np.minimum(((c + ext) / sp + 2).astype(int), shape)
    if np.any(hi <= lo):
        return np.zeros(shape, bool)
    g = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(float)
    v = np.stack([g[i] * sp[i] - c[i] for i in range(3)])
    t = np.tensordot(d, v, axes=1)
    perp2 = (v ** 2).sum(0) - t ** 2
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (perp2 <= radius_mm ** 2) & (np.abs(t) <= half_len_mm)
    return m


def _nail(shape, cl, k_from, k_to, radius_mm, sp):
    m = np.zeros(shape, bool)
    nj, ni = shape[1:]
    jj, ii = np.mgrid[:nj, :ni]
    for k in range(k_from, k_to + 1):
        m[k] = ((jj - cl[k, 0]) * sp[1]) ** 2 + ((ii - cl[k, 1]) * sp[2]) ** 2 <= radius_mm ** 2
    return m


def _transverse_screw(shape, cl, k, ang_deg, radius_mm, sp, half_len_mm=40.0):
    a = np.radians(ang_deg)
    return _cylinder(shape, (k, cl[k, 0], cl[k, 1]), (0.0, np.sin(a), np.cos(a)), radius_mm, half_len_mm, sp)


def _streaks(ct, metal, sp, rng):
    """
    Artefacto metálico aproximado en el plano: patrón angular claro/oscuro que
    decae con la distancia y una banda oscura entre objetos metálicos del mismo
    corte.  Es una aproximación de imagen, no física de reconstrucción.
    """
    out = ct.astype(np.float32)
    nj, ni = ct.shape[1:]
    jj, ii = np.mgrid[:nj, :ni]
    amp = rng.uniform(80, 250); lam = rng.uniform(15, 40); n = int(rng.integers(4, 11)); ph0 = rng.uniform(0, 2 * np.pi)
    for k in np.nonzero(metal.any(axis=(1, 2)))[0]:
        lab, nc = ndi.label(metal[k])
        cents = ndi.center_of_mass(metal[k], lab, range(1, nc + 1))
        # la intensidad del artefacto escala con el tamaño del objeto en el corte: un alambre de ~2 mm
        # produce rayas mucho más débiles que un clavo de ~10 mm (radio equivalente de referencia: 4 mm)
        areas = ndi.sum(np.ones_like(metal[k]), lab, range(1, nc + 1)) * sp[1] * sp[2]
        fac = np.clip(np.sqrt(np.asarray(areas) / np.pi) / 4.0, 0.15, 1.5)
        delta = np.zeros((nj, ni), np.float32)
        for (cj, ci), fa in zip(cents, fac):
            y = (jj - cj) * sp[1]; x = (ii - ci) * sp[2]
            r = np.hypot(x, y); th = np.arctan2(y, x)
            delta += fa * amp * np.cos(n * th + ph0) * np.exp(-r / lam)
        for a in range(len(cents)):
            for b in range(a + 1, len(cents)):
                (j0, i0), (j1, i1) = cents[a], cents[b]
                v = np.array([(j1 - j0) * sp[1], (i1 - i0) * sp[2]]); L = np.linalg.norm(v)
                if L < 1e-3:
                    continue
                v /= L
                py = (jj - j0) * sp[1]; px = (ii - i0) * sp[2]
                t = py * v[0] + px * v[1]
                dperp = np.abs(py * v[1] - px * v[0])
                band = (t > 0) & (t < L) & (dperp < 3.0)
                delta[band] -= amp * 1.2 * min(fac[a], fac[b])
        delta[metal[k]] = 0
        out[k] += delta
    return out


# ----------------------------------------------------------------------------
# generador principal
# ----------------------------------------------------------------------------
def make_defect_v2(ct, lbl, sp, params, rng=None, label_policy="host"):
    rng = rng or np.random.default_rng(SEED)
    p = params
    sp = np.asarray(sp, float)
    lbl = lbl > 0
    intact = fill_tibia(lbl)
    cl = centroid_line(intact)
    ks = np.nonzero(intact.any(axis=(1, 2)))[0]
    t0, t1 = int(ks.min()), int(ks.max())
    ctf = ct.astype(np.float32).copy()
    aux = np.zeros(ct.shape, np.uint8)
    host = lbl.copy()
    shape = ct.shape

    cort = ct[lbl]
    cort_hi = float(np.percentile(cort, 90))          # cortical densa, en HU de esta TC
    marrow = float(np.median(ct[intact & ~lbl])) if (intact & ~lbl).any() else 0.0

    # 1) región resecada con extremos irregulares
    removal, (ka, kb), (zp, zd) = _defect_region(intact, sp, p, cl)
    if ka is None:
        raise ValueError("defecto vacío")
    mu, sd = _soft_stats(ct, intact, sp, ka, kb)
    if p["content"] in ("hematoma", "fluid", "fibrous"):
        mu = CONTENT_HU[p["content"]][0]     # la media depende del tejido; el ruido es el del escáner (sd local)
    ctf[removal] = np.clip(_noise_fill(rng, removal, mu, sd), -100, 150)
    # borde difuso entre el contenido y el tejido blando vecino: sin esto queda marcado el contorno
    # exacto del hueso resecado, que en una TC real no se ve (el hematoma/fibrosis se mezcla con el músculo)
    edge = ndi.binary_dilation(removal, iterations=2) & ~ndi.binary_erosion(removal, iterations=2) & ~(intact & ~removal)
    if edge.any():
        sm = ndi.gaussian_filter(ctf, 1.2)
        ctf[edge] = sm[edge]
    host &= ~removal
    host_filled_side = intact & ~removal
    kp_end = int(np.floor(zp / sp[0])); kd_end = int(np.ceil(zd / sp[0]))

    def band_slices(k_end, length_mm, side):
        n = max(int(round(length_mm / sp[0])), 1)
        return range(k_end - n, k_end + 1) if side < 0 else range(k_end, k_end + n + 1)

    # 2) esclerosis de los extremos (+HU en el hueso remanente, decae con la distancia)
    if p["sclerosis"]:
        for k_end, side in ((kp_end, -1), (kd_end, +1)):
            for k in band_slices(k_end, p["sclerosis_len_mm"], side):
                if not (t0 <= k <= t1):
                    continue
                w = 1 - abs(k - k_end) * sp[0] / p["sclerosis_len_mm"]
                m = host[k]
                ctf[k][m] += p["sclerosis_dhu"] * max(w, 0)
                # (el engrosamiento endóstico de la versión anterior se quitó: no tiene respaldo en el
                #  catálogo y generaba un patrón dentado artificial dentro del canal)

    # 3) canal medular sellado (tapón cortical en cada extremo)
    # un extremo en punta o en cúpula (no unión) siempre queda cerrado: el tapón cubre la conicidad
    for end in ("prox", "dist"):
        if p[end]["taper_mm"] > 0 and not p["cap"]:
            p["cap"] = True; p["cap_len_mm"] = 2.0
    if p["cap"]:
        for k_end, side, end in ((kp_end, -1, "prox"), (kd_end, +1, "dist")):
            for k in band_slices(k_end, p["cap_len_mm"] + p[end]["taper_mm"], side):
                if not (t0 <= k <= t1):
                    continue
                canal = host_filled_side[k] & ~_largest_2d(host[k]) & ~removal[k]
                canal &= ndi.binary_erosion(intact[k], iterations=1)
                ctf[k][canal] = rng.normal(cort_hi, 80, int(canal.sum()))
                host[k] |= canal
                aux[k][canal] = AUX["cap"]

    # 4) callo hipertrófico: manguito perióstico que se ensancha hacia el extremo
    if p["callus"]:
        for k_end, side in ((kp_end, -1), (kd_end, +1)):
            n = int(round(p["callus_len_mm"] / sp[0]))
            over = int(round(2.0 / sp[0]))
            ks_c = list(range(k_end - n, k_end + over + 1) if side < 0 else range(k_end - over, k_end + n + 1))
            # callo limitado y asimétrico (Çalışır fig 8: "limited callus formation ... laterally"):
            # más grueso hacia un lado, en parches, no un manguito uniforme
            nzc = ndi.gaussian_filter(rng.normal(0, 1, (len(ks_c),) + shape[1:]).astype(np.float32),
                                      (3.0 / sp[0], 3.0 / sp[1], 3.0 / sp[2]))
            nzc /= max(float(nzc.std()), 1e-6)
            a_c = rng.uniform(-np.pi, np.pi)
            yy, xx = np.mgrid[:shape[1], :shape[2]]
            shell_all = np.zeros(shape, bool)
            for i_c, k in enumerate(ks_c):
                if not (t0 <= k <= t1) or not host[k].any():
                    continue
                dist_end = abs(k - k_end) * sp[0]
                th = np.arctan2((yy - cl[k, 0]) * sp[1], (xx - cl[k, 1]) * sp[2])
                lado = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(th - a_c))
                thick = p["callus_thick_mm"] * max(1 - dist_end / p["callus_len_mm"], 0.15) * lado * np.clip(1 + 0.7 * nzc[i_c], 0, 2.2)
                outside = ndi.distance_transform_edt(~ndi.binary_fill_holes(host[k]), sampling=sp[1:])
                shell = (outside > 0) & (outside <= thick) & (thick > 0.7)
                host[k] |= shell
                aux[k][shell] = AUX["callus"]
                shell_all[k] = shell
            if shell_all.any():
                ctf[shell_all] = _noise_fill(rng, shell_all, p["callus_hu"], 0.15 * p["callus_hu"], sigma_vox=1.2)

    # 4b) lisis mezclada con la esclerosis (B): focos en la cortical del huésped cerca de los extremos
    for _ in range(p.get("lysis", 0)):
        k_end = kp_end if rng.random() < 0.5 else kd_end
        off = rng.uniform(2, 25) / sp[0] * (-1 if k_end == kp_end else 1)
        kk = int(np.clip(k_end + off, t0, t1))
        pts = np.argwhere(host[kk])
        if len(pts) == 0:
            continue
        c = np.r_[kk, pts[rng.integers(len(pts))]]
        # foco lítico alargado en el eje (2-5 mm de radio, 2-3 veces más largo): la lisis de la
        # osteomielitis sigue la diáfisis; con una esfera chica casi no se veía en ningún corte
        rad = rng.uniform(2.0, 5.0); elong = rng.uniform(2.0, 3.0)
        kz = np.arange(shape[0])[:, None, None]; jy = np.arange(shape[1])[None, :, None]; ix = np.arange(shape[2])[None, None, :]
        b = ((((kz - c[0]) * sp[0]) / (rad * elong)) ** 2 + (((jy - c[1]) * sp[1]) / rad) ** 2 + (((ix - c[2]) * sp[2]) / rad) ** 2) <= 1
        b &= host
        if b.any():
            ctf[b] = _noise_fill(rng, b, 40, 15)
            host &= ~b

    # 4c) involucro (B): capa de hueso perióstico nuevo, continua, pegada a la cortical y de espesor irregular
    if p.get("involucrum"):
        sides = {"prox": [(kp_end, -1)], "dist": [(kd_end, +1)], "both": [(kp_end, -1), (kd_end, +1)]}[p["involucrum_side"]]
        n = int(round(p["involucrum_len_mm"] / sp[0]))
        gap = 0.0   # hueso perióstico nuevo: nace pegado a la cortical, no como una capa flotante
        inv_mask = np.zeros(shape, bool)
        for k_end, side in sides:
            ks_i = range(k_end - n, k_end) if side < 0 else range(k_end + 1, k_end + n + 1)
            # capa base continua pegada a la cortical (>= 1,5 mm) con engrosamientos suaves de baja frecuencia;
            # espesor acotado a [1,5 mm, 1,5 x espesor nominal] para que no aparezcan masas de 10 mm
            nzi = ndi.gaussian_filter(rng.normal(0, 1, (n + 2,) + shape[1:]).astype(np.float32),
                                      (8.0 / sp[0], 6.0 / sp[1], 6.0 / sp[2]))
            nzi /= max(float(nzi.std()), 1e-6)
            for ii_k, k in enumerate(ks_i):
                if not (t0 <= k <= t1) or not host[k].any():
                    continue
                fh = ndi.binary_fill_holes(host[k])
                dout = ndi.distance_transform_edt(~fh, sampling=sp[1:])
                w = 1 - abs(k - k_end) * sp[0] / p["involucrum_len_mm"]           # más grueso cerca del defecto
                tn = p["involucrum_thick_mm"]
                t = np.clip(tn * (0.5 + 0.5 * w) * (1 + 0.35 * nzi[ii_k]), 1.5, 1.5 * tn)   # >= 1,5 mm: 1 mm (~1 vóxel) se rompe en puntos
                shell = (dout > gap) & (dout <= gap + t)
                shell = ndi.binary_closing(shell, iterations=2) & (dout > 0)
                host[k] |= shell
                aux[k][shell] = AUX["involucrum"]
                inv_mask[k] |= shell
        if inv_mask.any():
            # hueso nuevo moteado: densidad heterogénea con pequeñas zonas más claras
            ctf[inv_mask] = _noise_fill(rng, inv_mask, p["involucrum_hu"], 0.25 * p["involucrum_hu"], sigma_vox=1.5)
        # cloaca: orificio que atraviesa cortical e involucro
    if p.get("cloaca"):
        k_end = kp_end if rng.random() < 0.5 else kd_end
        kk = int(np.clip(k_end + (rng.uniform(5, 20) / sp[0]) * (-1 if k_end == kp_end else 1), t0, t1))
        a = rng.uniform(-np.pi, np.pi)
        cyl = _cylinder(shape, (kk, cl[kk, 0], cl[kk, 1]), (0.0, np.sin(a), np.cos(a)), rng.uniform(3, 6), 30.0, sp)
        half = np.zeros(shape, bool)
        idx = np.argwhere(cyl)
        proj = (idx[:, 1] - cl[kk, 0]) * sp[1] * np.sin(a) + (idx[:, 2] - cl[kk, 1]) * sp[2] * np.cos(a)
        idx = idx[proj > 0]
        half[tuple(idx.T)] = True
        hole = half & host
        if hole.any():
            ctf[hole] = _noise_fill(rng, hole, 25, 10)
            host &= ~hole

    # 5) osteopenia por desuso (menos HU en el hueso huésped, decae con la distancia)
    if p["osteopenia"]:
        kc = (kp_end + kd_end) / 2
        kidx = np.arange(shape[0])
        dk = np.maximum(np.abs(kidx - kc) * sp[0] - p["length_mm"] / 2, 0)
        w = p["osteopenia_frac"] * np.exp(-dk / p["osteopenia_lambda_mm"])
        # solo el hueso preexistente pierde densidad: el hueso nuevo (callo, involucro, tapón) conserva
        # el valor absoluto asignado, para que su rango de HU documentado siga siendo válido
        old_bone = host & ~np.isin(aux, [AUX["callus"], AUX["involucrum"], AUX["cap"]])
        # solo sobre los vóxeles de hueso (en float32): evita volúmenes temporales float64 completos,
        # que en tibias largas (~57 M vóxeles) pedían ~0,5 GB cada uno
        kk = np.nonzero(old_bone)[0]
        ctf[old_bone] = marrow + (ctf[old_bone] - marrow) * (1 - w[kk]).astype(np.float32)

    # 5b) fíbula fracturada (solo imagen; la fíbula no está etiquetada)
    if p["fibula_fx"]:
        # trazo oblicuo (img32, img43): la brecha se desplaza a lo largo de k según la posición en el plano
        k0 = int(rng.integers(ka, kb + 1)); gap_mm = rng.uniform(2, 10); tilt = np.tan(np.radians(rng.uniform(0, 50)))
        old = p.get("fibula_old", False)
        if old:   # fractura antigua: trazo fino y callo alrededor (Çalışır fig 8c)
            gap_mm = rng.uniform(0.7, 2.0); cal_hu = rng.uniform(300, 900); cal_mm = rng.uniform(5, 12); a_cal = rng.uniform(-np.pi, np.pi)
        a = rng.uniform(-np.pi, np.pi)
        mu_s, sd_s = _soft_stats(ct, intact, sp, ka, kb)
        reach = int(round(40 / sp[0]))
        jj, ii = np.mgrid[:shape[1], :shape[2]]
        for k in range(max(k0 - reach, 0), min(k0 + reach, shape[0])):
            fib = (ct[k] > 250) & ~ndi.binary_dilation(intact[k], iterations=5)
            if not fib.any():
                continue
            fib = ndi.binary_dilation(fib, iterations=1) & ~ndi.binary_dilation(intact[k], iterations=3)
            u = ((jj - cl[k, 0]) * sp[1] * np.sin(a) + (ii - cl[k, 1]) * sp[2] * np.cos(a))
            zc = k0 * sp[0] + tilt * (u - u[fib].mean())
            if old:
                # callo fusiforme pegado a la fíbula: más grueso en el trazo, se afina a cal_mm
                rr = 2.5 * np.clip(1 - np.abs(k * sp[0] - zc) / cal_mm, 0, 1)
                dfo = ndi.distance_transform_edt(~ndi.binary_fill_holes(fib), sampling=sp[1:])
                # callo excéntrico, no un halo concéntrico: más grueso hacia un lado (a_cal)
                fj, fi = np.argwhere(fib).mean(0)
                thf = np.arctan2((jj - fj) * sp[1], (ii - fi) * sp[2])
                rr_map = rr * (0.25 + 0.75 * (0.5 + 0.5 * np.cos(thf - a_cal)))
                cal = (dfo > 0) & (dfo <= rr_map) & (rr_map > 0.6) & ~ndi.binary_dilation(intact[k], iterations=2)
                if cal.any():
                    ctf[k][cal] = np.maximum(ctf[k][cal], rng.normal(cal_hu, 60, int(cal.sum())))
            cut = fib & (np.abs(k * sp[0] - zc) <= gap_mm / 2)
            if cut.any():
                ctf[k][cut] = rng.normal(mu_s, sd_s, int(cut.sum()))

    geo = intact.copy()   # geometría para implantes y espaciador (sigue a la desalineación)
    # 6) desalineación del fragmento distal (desplazamiento en el plano que crece
    #     dentro del defecto y continúa como angulación por debajo; ángulos <= 10°)
    if p["malalign"]:
        a_t = np.radians(p["translation_dir_deg"]); a_a = np.radians(p["angulation_dir_deg"])
        tvec = p["translation_mm"] * np.array([np.sin(a_t), np.cos(a_t)])
        avec = np.tan(np.radians(p["angulation_deg"])) * np.array([np.sin(a_a), np.cos(a_a)])
        for k in range(kp_end, shape[0]):
            z = k * sp[0]
            w = np.clip((z - zp) / max(zd - zp, 1e-6), 0, 1)
            s_mm = w * tvec + max(z - zd, 0) * avec
            s_vox = s_mm / sp[1:]
            if np.abs(s_vox).max() < 0.05:
                continue
            ctf[k] = ndi.shift(ctf[k], s_vox, order=1, mode="nearest")
            host[k] = ndi.shift(host[k].astype(np.uint8), s_vox, order=0, mode="constant") > 0
            aux[k] = ndi.shift(aux[k], s_vox, order=0, mode="constant")
            geo[k] = ndi.shift(geo[k].astype(np.uint8), s_vox, order=0, mode="constant") > 0
        p["aligned"] = False
        cl = centroid_line(geo)
    else:
        p["aligned"] = True


    # 7) fragmentos libres (A) o secuestro (B): trozos de la cortical resecada
    removed_cort = lbl & removal
    frag_mask = np.zeros(shape, bool)
    n_pieces = p["fragments"]
    if n_pieces and removed_cort.sum() > 200:
        idx = np.argwhere(removed_cort)
        seeds = idx[rng.choice(len(idx), size=min(10, len(idx)), replace=False)]
        # Voronoi simple: asignar cada vóxel a la semilla más cercana (en mm)
        d2 = np.stack([(((idx - s) * sp) ** 2).sum(1) for s in seeds])
        owner = d2.argmin(0)
        order = rng.permutation(len(seeds))
        made = 0
        for s_i in order:
            if made >= n_pieces:
                break
            vox = idx[owner == s_i]
            vol = len(vox) * sp.prod()
            if vol < 30:
                continue
            is_seq = False
            if is_seq:
                shift_mm = rng.uniform(0, 1, 3)
            else:
                ang = rng.uniform(-np.pi, np.pi)
                rad = _u(rng, p["fragment_shift_mm"])
                shift_mm = np.array([rng.uniform(-3, 3), rad * np.sin(ang), rad * np.cos(ang)])
            sh = np.round(shift_mm / sp).astype(int)
            dst = vox + sh
            ok = np.all((dst >= 0) & (dst < shape), axis=1)
            vox, dst = vox[ok], dst[ok]
            vals = ct[tuple(vox.T)].astype(np.float32)
            if not is_seq and p.get("fragment_dhu", 0):
                vals = vals + p["fragment_dhu"]
            if is_seq:
                vals = vals + rng.uniform(150, 300)          # hueso necrótico, más denso
                ring = ndi.binary_dilation(_points(shape, dst), iterations=2) & ~_points(shape, dst)
                ring &= ~host & ~intact
                ctf[ring] = rng.normal(25, 10, int(ring.sum()))  # halo de tejido de granulación
            ctf[tuple(dst.T)] = vals
            frag_mask[tuple(dst.T)] = True
            aux[tuple(dst.T)] = AUX["sequestrum"] if is_seq else AUX["fragment"]
            made += 1
    # 7b) fragmentos grandes en cuña / ala de mariposa (A): trozo de la cortical resecada con forma de
    #     rombo (ancho angular máximo al centro, cero en las puntas), desplazado hacia afuera (img38, img32, img43)
    for w_ in p.get("wedges", []):
        nkw = int(round(w_["len_mm"] / sp[0]))
        kc_w = int(ka + w_["pos"] * (kb - ka)); k0w = max(kc_w - nkw // 2, ka); k1w = min(k0w + nkw, kb)
        piece = np.zeros(shape, bool)
        for k in range(k0w, k1w + 1):
            if not removed_cort[k].any():
                continue
            frac = 1 - abs((k - (k0w + k1w) / 2) / max((k1w - k0w) / 2, 1))
            y = (np.mgrid[:shape[1], :shape[2]][0] - cl[k, 0]) * sp[1]; x = (np.mgrid[:shape[1], :shape[2]][1] - cl[k, 1]) * sp[2]
            d = (np.degrees(np.arctan2(y, x)) - w_["dir_deg"] + 180) % 360 - 180
            piece[k] = removed_cort[k] & (np.abs(d) <= w_["arc_deg"] / 2 * frac)
        if piece.sum() * sp.prod() < 100:
            continue
        vox = np.argwhere(piece)
        a = np.radians(w_["dir_deg"] + rng.uniform(-30, 30))
        shift_mm = np.array([rng.uniform(-5, 5), w_["shift_mm"] * np.sin(a), w_["shift_mm"] * np.cos(a)])
        # pequeña inclinación del fragmento: desplazamiento lateral que crece a lo largo de k
        tilt = np.tan(np.radians(w_["tilt_deg"])) * (vox[:, 0] - vox[:, 0].mean()) * sp[0]
        dst = np.round((vox * sp + shift_mm + np.c_[np.zeros(len(vox)), tilt * np.sin(a), tilt * np.cos(a)]) / sp).astype(int)
        ok = np.all((dst >= 0) & (dst < shape), axis=1)
        vox, dst = vox[ok], dst[ok]
        ctf[tuple(dst.T)] = ct[tuple(vox.T)]
        frag_mask[tuple(dst.T)] = True
        aux[tuple(dst.T)] = AUX["fragment"]
    # 7c) secuestro (B): segmento de cortical necrótica del HUÉSPED (no del hueco desbridado) en el
    #     muñón adyacente a un extremo del defecto, dentro de la zona del involucro si existe. Queda en su
    #     sitio, más denso que la cortical viva (+150-300 HU) y aislado del hueso vivo por un halo de
    #     tejido de granulación de ~1,5 mm (Desimpel: más denso que el hueso vivo y separado de él;
    #     Solomin fig 16a: extremos necróticos). Etiqueta 0 (no es hueso huésped viable).
    if p["sequestrum"]:
        L = rng.uniform(10, 25); arc = rng.uniform(60, 140); dir_ = rng.uniform(-180, 180)
        side = p.get("involucrum_side", "both") if p.get("involucrum") else "both"
        near_prox = (rng.random() < 0.5) if side == "both" else side == "prox"
        nk = int(round(L / sp[0])); off = int(round(rng.uniform(2, 6) / sp[0]))
        if near_prox:
            k1q = kp_end - off; k0q = k1q - nk
        else:
            k0q = kd_end + off; k1q = k0q + nk
        k0q, k1q = max(k0q, t0), min(k1q, t1)
        piece = np.zeros(shape, bool)
        jy = np.arange(shape[1])[:, None]; ix = np.arange(shape[2])[None, :]
        for k in range(k0q, k1q + 1):
            if not lbl[k].any() or removal[k].any():
                continue
            dsurf = ndi.distance_transform_edt(intact[k], sampling=sp[1:])   # profundidad desde el periostio
            d = (np.degrees(np.arctan2((jy - cl[k, 0]) * sp[1], (ix - cl[k, 1]) * sp[2])) - dir_ + 180) % 360 - 180
            piece[k] = lbl[k] & host[k] & (dsurf <= 4.0) & (np.abs(d) <= arc / 2)
        if piece.sum() * sp.prod() >= 100:
            seq = piece
            halo = ndi.binary_dilation(seq, iterations=2) & ~seq & ~frag_mask   # también lo separa del involucro
            ctf[halo] = _noise_fill(rng, halo, 25, 10)
            host &= ~halo
            aux[halo] = 0
            ctf[seq] = ct[seq].astype(np.float32) + rng.uniform(150, 300)
            frag_mask |= seq
            aux[seq] = AUX["sequestrum"]
            p["sequestrum_detail"] = dict(len_mm=round(L, 1), arc_deg=round(arc, 1),
                                          near="prox" if near_prox else "dist", k_range=[int(k0q), int(k1q)])
    host &= ~frag_mask

    # 8) gas (fractura abierta) dentro del defecto o en su borde
    for _ in range(p["n_gas"]):
        pts = np.argwhere(removal)
        c = pts[rng.integers(len(pts))]
        g = _ball(shape, c, rng.uniform(1, 3), sp) & ~host
        ctf[g] = rng.uniform(-900, -500)
        aux[g] = AUX["gas"]

    # 9) espaciador de cemento (Masquelet, etapa 1)
    spacer = np.zeros(shape, bool)
    if p["spacer"]:
        ov = int(round(p["spacer_overlap_mm"] / sp[0]))
        k0s, k1s = max(ka - ov, t0), min(kb + ov, t1)
        # cuerpo moldeado a mano: el espesor radial del cemento (margen) se mide desde el contorno
        # perióstico de la tibia intacta en cada corte (sigue su forma triangular), con una irregularidad
        # suave de baja frecuencia (orden angular 2-3 que varía lento a lo largo del eje); dentro del
        # defecto el espesor queda acotado al rango 3-10 mm del margen definido
        nkS = k1s - k0s + 1
        zz = np.linspace(0, 1, nkS)
        ph = rng.uniform(0, 2 * np.pi, 4); am = rng.uniform(0.05, 0.15, 2)
        jy = np.arange(shape[1])[:, None]; ix = np.arange(shape[2])[None, :]
        for k in range(k0s, k1s + 1):
            if not geo[k].any():
                continue
            gk = ndi.binary_fill_holes(geo[k])
            dper = ndi.distance_transform_edt(~gk, sampling=sp[1:])
            th = np.arctan2((jy - cl[k, 0]) * sp[1], (ix - cl[k, 1]) * sp[2])
            z = zz[k - k0s]
            wob = 1 + sum(am[i] * np.cos((i + 2) * th + ph[i] + 2 * np.pi * z * (0.5 + 0.3 * i) + ph[i + 2]) for i in range(2))
            # el margen disminuye linealmente hasta el 30 % en la zona de sobreposición con los muñones
            fk = 1.0 if ka <= k <= kb else max(1 - (min(abs(k - ka), abs(k - kb)) * sp[0]) / max(p["spacer_overlap_mm"], 1), 0.3)
            m = p["spacer_margin_mm"] * fk * wob
            if ka <= k <= kb:
                m = np.clip(m, 3.0, 10.0)
            ref = (dper <= m) | gk
            spacer[k] = ref & ~ndi.binary_fill_holes(host[k]) if k < ka or k > kb else ref & ~host[k]
        spacer &= ~frag_mask
        # cemento con antibiótico: textura granular/moteada, no homogénea (Medicine 2023 fig 1C; Simultaneous fig 2H)
        ctf[spacer] = _noise_fill(rng, spacer, p["spacer_hu"], max(60.0, 0.15 * p["spacer_hu"]), sigma_vox=1.0)
        aux[spacer] = AUX["spacer"]

    # 11) implantes
    metal = np.zeros(shape, bool)
    imp = p["implant"]
    p_imp = dict(type=imp)
    if imp == "nail":
        rad = rng.uniform(4.5, 6.0)
        k_from = t0 + int(0.05 * (t1 - t0)); k_to = t1 - int(0.05 * (t1 - t0))
        metal |= _nail(shape, cl, k_from, k_to, rad, sp)
        screws = []
        for k_s in (k_from + int(rng.uniform(15, 40) / sp[0]), k_from + int(rng.uniform(45, 70) / sp[0]),
                    k_to - int(rng.uniform(15, 40) / sp[0]), k_to - int(rng.uniform(45, 70) / sp[0])):
            if t0 <= k_s <= t1 and not (ka - 10 <= k_s <= kb + 10):
                a = rng.uniform(-180, 180)
                metal |= _transverse_screw(shape, cl, k_s, a, rng.uniform(2.0, 2.5), sp) & _near(geo, sp, 3.0)
                screws.append(int(k_s))
        p_imp.update(radius_mm=round(rad, 2), k_from=int(k_from), k_to=int(k_to), screws_k=screws)
    elif imp == "plate":
        phi = rng.uniform(-180, 180); width = rng.uniform(11, 16); thick = rng.uniform(3, 5)
        ext = rng.uniform(40, 80)
        k_from = max(int((zp - ext) / sp[0]), t0); k_to = min(int((zd + ext) / sp[0]), t1)
        nj, ni = shape[1:]; jj, ii = np.mgrid[:nj, :ni]
        for k in range(k_from, k_to + 1):
            ref = geo[k]
            if not ref.any():
                continue
            dout = ndi.distance_transform_edt(~ref, sampling=sp[1:])
            y = (jj - cl[k, 0]) * sp[1]; x = (ii - cl[k, 1]) * sp[2]
            R = np.sqrt(ref.sum() * sp[1] * sp[2] / np.pi)
            th = np.degrees(np.arctan2(y, x))
            half_ang = np.degrees(width / 2 / (R + 1))
            ang_ok = np.abs((th - phi + 180) % 360 - 180) <= half_ang
            metal[k] |= (dout > 0.5) & (dout <= 0.5 + thick) & ang_ok
        screws = []
        a = np.radians(phi)
        for k_s in np.arange(k_from + 5 / sp[0], k_to, rng.uniform(12, 18) / sp[0]).astype(int):
            if ka - 8 / sp[0] <= k_s <= kb + 8 / sp[0]:
                continue
            p0 = (k_s, cl[k_s, 0], cl[k_s, 1])
            metal |= _cylinder(shape, p0, (0.0, np.sin(a), np.cos(a)), rng.uniform(1.75, 2.25), 25.0, sp) & (_near(geo, sp, 2.0) | _near(metal, sp, 1.0))
            screws.append(int(k_s))
        p_imp.update(dir_deg=round(phi, 1), width_mm=round(width, 1), thick_mm=round(thick, 1),
                     k_from=int(k_from), k_to=int(k_to), screws_k=screws)
    elif imp == "ring":
        # fijador circular (Ilizarov/TrueLok): alambres finos transfixiantes cruzados + 1 pin por anillo
        body = ct > -500
        wires = []
        for zc_, sgn in ((zp, -1), (zd, +1)):
            for d in (rng.uniform(15, 35), rng.uniform(50, 80)):
                k_s = int((zc_ + sgn * d) / sp[0])
                if not (t0 <= k_s <= t1):
                    continue
                a0 = rng.uniform(-180, 180)
                for da in (0.0, rng.uniform(50, 70)):
                    a = np.radians(a0 + da)
                    k_w = min(max(k_s + int(round(rng.uniform(-3, 3) / sp[0])), t0), t1)
                    p0 = (k_w, cl[k_w, 0], cl[k_w, 1])
                    metal |= _cylinder(shape, p0, (0.0, np.sin(a), np.cos(a)), rng.uniform(0.75, 1.0), 120.0, sp) & body
                    wires.append(int(k_w))
        p_imp.update(wires_k=wires)
    elif imp == "exfix":
        # los pins se colocan por la cara subcutánea anteromedial de la tibia (corredor seguro); se toma
        # como esa cara la dirección de menor espesor de partes blandas entre el periostio y la piel, que
        # no depende de si la pierna es izquierda o derecha
        def _safe_dir(k):
            body = ndi.binary_fill_holes(ct[k] > -300)
            gk = ndi.binary_fill_holes(geo[k])
            best, best_a = 1e9, 0.0
            for ang in np.arange(-180, 180, 5):
                ar = np.radians(ang); rs = np.arange(0, 120, 0.5)
                jj = np.round(cl[k, 0] + rs * np.sin(ar) / sp[1]).astype(int); ii = np.round(cl[k, 1] + rs * np.cos(ar) / sp[2]).astype(int)
                ok = (jj >= 0) & (jj < shape[1]) & (ii >= 0) & (ii < shape[2])
                jj, ii, rr = jj[ok], ii[ok], rs[ok]
                if not gk[jj, ii].any():
                    continue
                r_b = rr[np.nonzero(gk[jj, ii])[0].max()]
                out = np.nonzero(~body[jj, ii] & (rr > r_b))[0]
                # si el recorte de la ROI corta la pierna, el borde de la imagen cuenta como piel (cota inferior)
                r_skin = rr[out[0]] if out.size else rr[-1]
                if r_skin - r_b < best:
                    best, best_a = r_skin - r_b, float(ang)
            return best_a
        pins = []
        for zc_, sgn in ((zp, -1), (zd, +1)):
            base = zc_ + sgn * rng.uniform(25, 50)
            for d in (0.0, sgn * rng.uniform(20, 40)):
                k_s = int((base + d) / sp[0])
                if not (t0 <= k_s <= t1):
                    continue
                a = np.radians(_safe_dir(k_s) + rng.uniform(-25, 25))
                p0 = (k_s, cl[k_s, 0], cl[k_s, 1])
                # pin bicortical que sale hacia la piel por un lado (semiespacio)
                cyl = _cylinder(shape, p0, (0.0, np.sin(a), np.cos(a)), rng.uniform(2.5, 3.0), 80.0, sp)
                kk, jj_, ii_ = np.nonzero(cyl)
                proj = (jj_ - cl[k_s, 0]) * sp[1] * np.sin(a) + (ii_ - cl[k_s, 1]) * sp[2] * np.cos(a)
                R = np.sqrt(max(geo[k_s].sum(), 1) * sp[1] * sp[2] / np.pi)
                keep = proj >= -(R + 2)
                metal[kk[keep], jj_[keep], ii_[keep]] = True
                pins.append(int(k_s))
        p_imp.update(pins_k=pins)
    if p.get("spacer_rod") and spacer.any():
        # varilla de refuerzo dentro del espaciador (img20)
        ks_ = np.nonzero(spacer.any(axis=(1, 2)))[0]
        ext = int(round(rng.uniform(10, 30) / sp[0]))
        metal |= _nail(shape, cl, max(ks_.min() - ext, t0), min(ks_.max() + ext, t1), rng.uniform(1.5, 3.0), sp)
        p_imp["spacer_rod"] = True
    if p.get("broken") and imp in ("plate", "nail") and metal.any():
        # material roto a la altura del defecto (img19)
        kbk = int(rng.integers(ka, kb + 1)); g = max(int(round(rng.uniform(1.5, 4) / sp[0])), 1)
        metal[kbk:kbk + g] = False
        p_imp["broken_k"] = kbk
    if metal.any():
        bloom = ndi.binary_dilation(metal, iterations=1) & ~metal
        ctf = _streaks(ctf, metal, sp, rng)
        ctf[bloom] = np.maximum(ctf[bloom], rng.uniform(1500, 2500))
        ctf[metal] = METAL_HU
        aux[metal] = AUX["metal"]
        host &= ~metal
        frag_mask &= ~metal
    p["implant_detail"] = p_imp

    # etiqueta
    label = host.copy()
    if label_policy == "all_bone":
        label |= frag_mask
    label &= aux != AUX["metal"]
    label &= aux != AUX["spacer"]

    # clics: 5 mm fuera del extremo más alejado de la resección
    m = int(round(CLICK_MARGIN_MM / sp[0]))
    clicks = (int(max(ka - m, t0)), int(min(kb + m, t1)))
    info = np.iinfo(ct.dtype) if np.issubdtype(ct.dtype, np.integer) else None
    # rango del escáner: las bandas oscuras del artefacto de metal no bajan del mínimo de la TC (-1024 en
    # pat_01 y Fischer) ni nada sube del techo de 12 bits (METAL_HU); sin esto aparecían valores de -6000 HU
    ctf = np.clip(ctf, float(ct.min()), max(float(ct.max()), METAL_HU))
    ct_out = np.clip(np.round(ctf), info.min, info.max).astype(ct.dtype) if info else ctf.astype(ct.dtype)
    p["solomin"] = solomin_descriptor(p, intact, sp, (ka + kb) // 2)
    p.update(k_start=int(ka), k_end=int(kb), nominal_z_mm=(round(zp, 2), round(zd, 2)), clicks_k=clicks,
             tibia_k=(t0, t1), content_hu=(round(mu, 1), round(sd, 1)), label_policy=label_policy)
    return dict(ct=ct_out, label=label.astype(np.uint8), removal=removal, aux=aux, clicks=clicks, meta=p)


def solomin_descriptor(p, intact, sp, k_mid):
    """Descriptor morfológico según la ULBDC de Solomin et al. (2026), solo diáfisis.

    La ULBDC describe la integridad ósea (Bi), el contacto entre segmentos, el acortamiento (Sh) y la
    diástasis (D). La extensión circunferencial del defecto es una variable independiente del generador
    (sector_deg) y se registra aparte. La traducción a un tipo ULBDC solo vale bajo dos condiciones del
    generador: no hay acortamiento (Sh = 0) y, en un defecto parcial, la cortical remanente es continua.
      - pérdida circunferencial -> segmentos sin contacto -> Bi interrumpida con diástasis -> C3, D = BDs-l
      - pérdida parcial con cortical remanente continua -> Bi parcialmente íntegra (Solomin fig. 1b)
        -> defecto cortical con integridad conservada -> tipo A (no B: B exige Bi interrumpida)
    Bd = diámetro equivalente de la tibia en el centro del defecto (sqrt(4A/pi)); Solomin propone 30 mm.
    C1 (sin brecha) y C2 (contacto parcial tras acortamiento agudo) quedan fuera del alcance: el
    generador no simula acortamiento ni angulación correctiva.
    """
    area = float(ndi.binary_fill_holes(intact[k_mid]).sum() * sp[1] * sp[2])
    bd = float(np.sqrt(4 * area / np.pi)) if area > 0 else 30.0
    r = p["length_mm"] / bd
    sg = 1 if r < 1 else 2 if r < 2 else 3 if r < 4 else 4
    circ = p["sector_deg"] >= 360
    out = dict(extension="circunferencial" if circ else "parcial", sector_deg=round(float(p["sector_deg"]), 1),
               bd_mm=round(bd, 1), length_bd=round(r, 2), subgroup_len=sg, shortening_mm=0.0)
    if circ:
        out.update(type="ULBDC-C3", code=f"ULBDC C3.{sg}.{sg}", bone_integrity="interrumpida", diastasis_mm=round(p["length_mm"], 1))
    else:
        fr = p["sector_deg"] / 360
        out.update(type="ULBDC-A", code="ULBDC A (defecto cortical)", bone_integrity="parcialmente íntegra",
                   width_third=1 if fr < 1 / 3 else 2 if fr < 2 / 3 else 3, diastasis_mm=0.0)
    return out


def _points(shape, pts):
    m = np.zeros(shape, bool)
    m[tuple(pts.T)] = True
    return m


def _near(mask, sp, mm):
    """Vóxeles a <= mm (en el plano) de la máscara."""
    out = np.zeros_like(mask, dtype=bool)
    it = max(int(round(mm / sp[1])), 1)
    for k in np.nonzero(mask.any(axis=(1, 2)))[0]:
        out[k] = ndi.binary_dilation(mask[k], iterations=it)
    return out


def _noise_fill(rng, mask, mu, sd, sigma_vox=0.8):
    """Ruido gaussiano con correlación espacial leve (más parecido al ruido de TC que el ruido blanco)."""
    idx = np.argwhere(mask)
    lo = idx.min(0); hi = idx.max(0) + 1
    n = rng.normal(0, 1, tuple(hi - lo)).astype(np.float32)
    n = ndi.gaussian_filter(n, sigma_vox)
    n /= max(float(n.std()), 1e-6)
    return mu + sd * n[tuple((idx - lo).T)]


def _largest_2d(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()
