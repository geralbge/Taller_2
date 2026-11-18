"""
Generación paramétrica de andamios porosos TPMS dentro de la envolvente del
defecto y métricas de arquitectura porosa / verificación geométrica.

Representación implícita
------------------------
Se trabaja en la grilla de la ROI orientada de la tibia (k = eje largo), a una
resolución fina h (mm). El andamio es el conjunto {f <= 0} con

    f = max( d_tpms , d_env )                (intersección suave por máximo)

  d_env  = distancia con signo (mm) a la envolvente del defecto (+ holgura)
           (función de distancia de la máscara del defecto, interpolada)
  d_tpms = distancia aproximada (mm) a la pared TPMS:
           sheet   : (|G| - w) / |∇G|      (pared de espesor ~t alrededor de G = 0)
           network : (G - w) / |∇G|        (sólido en un laberinto; poros en el otro)

  Giroide: G = sin X cos Y + sin Y cos Z + sin Z cos X,  X = 2π x / a  (a = celda, mm)

La superficie se extrae con marching cubes y se suaviza con Taubin (mismo
criterio que la reconstrucción de la tibia). La celda se alinea con el eje
largo de la tibia (útil para la compresión axial del FEA).

Métricas (Capítulo II, ecs. 8–12)
---------------------------------
  porosidad P = 1 - V_sólido / V_envolvente
  ocupación PO = V_envolvente / V_defecto
  coincidencia geométrica CG = |E ∩ D| / |E ∪ D|  (envolvente del andamio vs defecto)
  distancia superficial promedio envolvente <-> defecto
  tamaño de poro: diámetro de la mayor esfera inscrita en los canales
      (2 × máximos locales de la transformada de distancia del espacio poroso)
  espesor de pared: ídem sobre el sólido
  componentes desconectadas (se eliminan fragmentos < MIN_FRAGMENT_FRAC del volumen)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

TPMS = {
    "gyroid": lambda X, Y, Z: np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X),
    "diamond": lambda X, Y, Z: (np.sin(X) * np.sin(Y) * np.sin(Z) + np.sin(X) * np.cos(Y) * np.cos(Z)
                                + np.cos(X) * np.sin(Y) * np.cos(Z) + np.cos(X) * np.cos(Y) * np.sin(Z)),
    "primitive": lambda X, Y, Z: np.cos(X) + np.cos(Y) + np.cos(Z),
}
MIN_FRAGMENT_FRAC = 0.001   # fragmentos sólidos < 0,1 % del volumen se eliminan (no imprimibles / no mallables)
KEEP_LARGEST_ONLY = True    # (05/11/2026) se conserva solo la mayor componente sólida (26-conectividad): una pieza
                            # unida al resto solo por aristas/vértices del defecto no transmite carga ni es fabricable
SOLID_REMOVED_WARN_FRAC = 0.05  # aviso de QC si al quedarse con la mayor componente se quita > 5 % del sólido


# ------------------------------------------------------------- celda unitaria
def _unit_cell(kind, n=96):
    t = (np.arange(n) + 0.5) / n * 2 * np.pi
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    G = TPMS[kind](X, Y, Z)
    gx, gy, gz = np.gradient(G, t[1] - t[0])
    gn = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return G, gn


def grad_scale(kind):
    """|∇G| medio sobre la superficie G≈0, en unidades de fase (multiplicar por 2π/a para 1/mm)."""
    G, gn = _unit_cell(kind)
    band = np.abs(G) < 0.05
    return float(gn[band].mean())


def porosity_of(kind, mode, w):
    G, _ = _unit_cell(kind)
    solid = (np.abs(G) <= w) if mode == "sheet" else (G <= w)
    return 1.0 - float(solid.mean())


def w_for_porosity(kind, mode, porosity):
    lo, hi = (0.0, 3.0) if mode == "sheet" else (-3.0, 3.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        p = porosity_of(kind, mode, mid)
        # sheet: más w -> menos porosidad; network: más w -> menos porosidad
        if p > porosity:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def w_for_wall(kind, mode, cell_mm, wall_mm):
    """Espesor de pared t (mm) -> umbral w (aprox. lineal cerca de G=0). Solo para 'sheet'."""
    if mode != "sheet":
        raise ValueError("El espesor de pared solo define el modo 'sheet'; para 'network' use --porosity")
    return 0.5 * wall_mm * grad_scale(kind) * 2 * np.pi / cell_mm


# ------------------------------------------------------------- campos
def signed_distance(mask, spacing):
    """Distancia con signo (mm): negativa dentro. Precisión ~ medio vóxel."""
    inside = ndi.distance_transform_edt(mask, sampling=spacing)
    outside = ndi.distance_transform_edt(~mask, sampling=spacing)
    return outside - inside


def _resample_to_fine(arr, sp, h, n_out=None):
    """Remuestrea (trilineal) un campo de la ROI a paso ~h conservando los extremos.
    Si se da n_out, fuerza ese tamaño (misma grilla que un campo ya construido).
    Devuelve el campo fino y el paso real por eje (k,j,i)."""
    n_in = np.array(arr.shape)
    if n_out is None:
        n_out = np.maximum(np.round((n_in - 1) * np.asarray(sp) / h).astype(int) + 1, 2)
    n_out = np.asarray(n_out, int)
    # Con grid_mode=False, scipy deduce el tamaño de salida como round(n_in·zoom) y
    # alinea los extremos según ese tamaño. Se pasa zoom = n_out/n_in para que la
    # salida tenga exactamente n_out muestras con extremos alineados (antes se
    # recortaba una salida más grande, lo que estiraba el campo ~1 %).
    out = ndi.zoom(arr.astype(np.float32), n_out / n_in, order=1, grid_mode=False, mode="nearest")
    if tuple(out.shape) != tuple(n_out):
        raise RuntimeError(f"remuestreo: forma {out.shape} != {tuple(n_out)}")
    h_axes = (n_in - 1) * np.asarray(sp) / (np.array(out.shape) - 1)
    return out, h_axes


def _crop_box(defect, sp, margin_mm):
    idx = np.nonzero(defect)
    pad = [int(np.ceil((margin_mm + 2) / s)) + 1 for s in sp]
    lo = [max(int(a.min()) - p, 0) for a, p in zip(idx, pad)]
    hi = [min(int(a.max()) + p + 1, n) for a, p, n in zip(idx, pad, defect.shape)]
    return lo, hi


def build_scaffold_field(defect, sp, h=0.12, cell_mm=3.0, kind="gyroid", mode="sheet",
                         w=0.5, clearance_mm=0.0, margin_mm=1.0):
    """
    defect: máscara bool (k,j,i) en la ROI; sp: spacing (k,j,i) mm.
    Devuelve dict con f (campo implícito, float32, grilla fina), env (SDF de la
    envolvente con holgura), h_axes (paso real por eje) y origin_idx (índice
    continuo de la ROI donde empieza la grilla fina).
    """
    lo, hi = _crop_box(defect, sp, margin_mm)
    crop = defect[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sdf = ndi.gaussian_filter(signed_distance(crop, sp), 0.5)      # suaviza escalones de vóxel
    env, h_axes = _resample_to_fine(sdf, sp, h)
    del sdf
    env += np.float32(clearance_mm)

    s = 2 * np.pi / cell_mm
    ax = [np.arange(n, dtype=np.float32) * np.float32(hh * s) for n, hh in zip(env.shape, h_axes)]
    K, J, I = ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]
    G = TPMS[kind](K, J, I).astype(np.float32)
    gs = np.float32(grad_scale(kind) * s)                          # |∇G| medio en 1/mm
    if mode == "sheet":
        np.abs(G, out=G)
    G -= np.float32(w)
    G /= gs                                                        # distancia aprox. (mm) a la pared
    np.maximum(G, env, out=G)
    return dict(f=G, env=env, h_axes=h_axes, h=float(np.mean(h_axes)),
                params=dict(kind=kind, mode=mode, w=float(w), cell_mm=float(cell_mm)),
                origin_idx=np.array(lo, float), lo=lo, hi=hi)


def lattice_cube(params, h, n_cells=4):
    """Retícula TPMS (bool) en un cubo de n_cells celdas y máscara del núcleo central (2 celdas)."""
    a = params["cell_mm"]; n = int(np.ceil(n_cells * a / h))
    s = 2 * np.pi / a
    x = (np.arange(n, dtype=np.float32) * np.float32(h * s))
    G = TPMS[params["kind"]](x[:, None, None], x[None, :, None], x[None, None, :]).astype(np.float32)
    if params["mode"] == "sheet":
        np.abs(G, out=G)
    lat = G <= params["w"]
    c = np.zeros(n, bool); c[n // 4: n - n // 4] = True
    core = c[:, None, None] & c[None, :, None] & c[None, None, :]
    return lat, core


def solid_mask(field):
    return field["f"] <= 0


def clean_fragments(solid, min_frac=MIN_FRAGMENT_FRAC, keep_largest=None):
    """Componentes conexas del sólido con 26-conectividad (vecindad 3x3x3).
    Se eliminan las < min_frac del volumen (ruido). Si keep_largest (por defecto KEEP_LARGEST_ONLY),
    además se conserva solo la mayor: n_relevantes_antes cuenta las componentes >= min_frac antes de ese paso,
    vox_quitados_mayor / frac_quitada_mayor lo que se quitó por quedarse con la mayor, y aviso_qc se activa
    si se quita > SOLID_REMOVED_WARN_FRAC del sólido en total."""
    keep_largest = KEEP_LARGEST_ONLY if keep_largest is None else keep_largest
    lab, n = ndi.label(solid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return solid, dict(n_components=int(n), n_kept=int(n), n_relevantes_antes=int(n), removed_components=0,
                           removed_frac=0.0, vox_quitados_mayor=0, frac_quitada_mayor=0.0, aviso_qc=False)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_frac * sizes.sum()
    n_rel = int(keep.sum())
    vox_big = 0
    if keep_largest and n_rel > 1:
        big = int(np.argmax(sizes)) + 1
        vox_big = int(sizes[keep[1:]].sum() - sizes[big - 1])
        keep[:] = False
        keep[big] = True
    out = keep[lab]
    removed_frac = float(1 - out.sum() / solid.sum())
    return out, dict(n_components=int(n), n_kept=int(keep.sum()), n_relevantes_antes=n_rel,
                     removed_components=int(n - keep.sum()), removed_frac=removed_frac,
                     vox_quitados_mayor=vox_big, frac_quitada_mayor=float(vox_big / solid.sum()),
                     aviso_qc=bool(removed_frac > SOLID_REMOVED_WARN_FRAC))


def apply_mask_to_field(field, solid):
    """Fuerza f > 0 donde se eliminaron fragmentos (para que marching cubes no los genere)."""
    f = field["f"].copy()
    removed = (f <= 0) & ~solid
    f[removed] = np.float32(field["h"])
    return f


# ------------------------------------------------------------- métricas
def local_thickness(mask, h, max_bins=24):
    """
    Espesor local (Hildebrand & Rüegsegger, 1997; definición de BoneJ): para cada
    vóxel, diámetro de la mayor esfera contenida en 'mask' que lo incluye.
    Implementación por bandas de radio sobre la cresta de la transformada de
    distancia. Devuelve un mapa (mm) con NaN fuera de 'mask'.
    Corrección de discretización: diámetro = 2·d − h/2 (error máx. ±h/2).
    """
    hm = float(np.mean(h))
    d = ndi.distance_transform_edt(mask, sampling=h).astype(np.float32)
    ridge = mask & (d >= ndi.maximum_filter(d, size=3))
    rv = np.unique(np.round(d[ridge] / hm).astype(int))
    rv = rv[rv > 0][::-1]
    if rv.size > max_bins:
        rv = np.unique(np.linspace(rv.min(), rv.max(), max_bins).round().astype(int))[::-1]
    th = np.full(mask.shape, np.nan, np.float32)
    for r in rv:
        seeds = ridge & (np.round(d / hm) >= r)
        if not seeds.any():
            continue
        cover = (ndi.distance_transform_edt(~seeds, sampling=h) <= (r + 0.5) * hm) & mask & np.isnan(th)
        th[cover] = max(2 * r * hm - 0.5 * hm, hm)
    th[mask & np.isnan(th)] = hm
    return th


def feature_sizes(mask, h, eval_mask=None):
    """
    Estadísticos del espesor local (mm) de 'mask', evaluados en 'eval_mask'
    (por defecto, todo 'mask'): media, mediana, p10, p90, máx.
    """
    ev = mask if eval_mask is None else (mask & eval_mask)
    if not ev.any():
        return dict(mean=float("nan"), median=float("nan"), p10=float("nan"), p90=float("nan"),
                    max=float("nan"))
    th = local_thickness(mask, h)
    v = th[ev]
    return dict(mean=float(v.mean()), median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)), max=float(v.max()))


def jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def architecture_metrics(field, solid, cell_mm, defect_fine=None):
    h = tuple(field["h_axes"])
    env = field["env"] <= 0
    vox = float(np.prod(h))
    v_env = float(env.sum() * vox)
    v_sol = float((solid & env).sum() * vox)
    m = dict(V_envolvente_mm3=v_env, V_solido_mm3=v_sol,
             porosidad_pct=100.0 * (1 - v_sol / v_env) if v_env else float("nan"))
    # Poro y pared: la retícula es periódica, así que se miden en un cubo de
    # 4 × 4 × 4 celdas de la retícula SIN recortar (evaluando el núcleo, lejos de
    # los bordes del cubo). Mismo resultado que dentro de la envolvente, con
    # mucha menos memoria.
    lat, core = lattice_cube(field["params"], float(np.mean(h)))
    hh = (float(np.mean(h)),) * 3
    m["poro_mm"] = feature_sizes(~lat, hh, core)
    m["pared_mm"] = feature_sizes(lat, hh, core)
    if defect_fine is not None:
        v_def = float(defect_fine.sum() * vox)
        m["V_defecto_mm3"] = v_def
        m["ocupacion_pct"] = 100.0 * v_env / v_def if v_def else float("nan")
        m["coincidencia_geometrica"] = jaccard(env, defect_fine)
    return m


def defect_on_fine_grid(defect, sp, field, offset_mm=0.0):
    """Máscara {SDF + offset <= 0} de 'defect' en la grilla fina del andamio
    (offset = holgura; 0 = el defecto mismo)."""
    lo, hi = field["lo"], field["hi"]
    crop = defect[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sdf = ndi.gaussian_filter(signed_distance(crop, sp), 0.5)
    # misma grilla exacta que el campo (no se recalcula el tamaño desde h medio,
    # que podía dar 1 vóxel menos en algún eje)
    fine, _ = _resample_to_fine(sdf, sp, field["h"], n_out=field["f"].shape)
    return (fine + np.float32(offset_mm)) <= 0


def field_to_mesh(f, field, sp, img_origin, img_direction):
    """
    Marching cubes sobre el campo implícito (nivel 0) -> vértices LPS (mm), caras.
    La grilla fina (k,j,i) empieza en el índice continuo origin_idx de la ROI.
    """
    from skimage.measure import marching_cubes
    pad = np.pad(f, 1, mode="constant", constant_values=np.float32(1.0))
    v, faces, _, _ = marching_cubes(pad, level=0.0, spacing=tuple(field["h_axes"]))
    v -= np.asarray(field["h_axes"])                                  # deshace el padding
    idx_kji = field["origin_idx"] + v / np.asarray(sp)                # índice continuo en la ROI
    D = np.asarray(img_direction, float).reshape(3, 3)
    sp_xyz = np.asarray(sp, float)[::-1]
    phys = np.asarray(img_origin, float) + (D @ (idx_kji[:, ::-1] * sp_xyz).T).T
    faces = faces.astype(np.int64)
    vol = _signed_volume(phys, faces)
    if vol < 0:
        faces = faces[:, ::-1].copy()
    return phys, faces


def _signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
