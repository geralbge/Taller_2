"""
F4. Parámetros del andamio: envolvente, gradiente radial, material y fabricabilidad.

Extiende scaffold_core.py SIN cambiar su comportamiento por defecto. Todo lo de
este módulo es opcional: si no se activa nada, generate_scaffold.py produce
exactamente el mismo andamio que antes (mismos 30 defectos virtuales).

Envolvente
----------
  holgura (clearance_mm)  ya existía en scaffold_core: SDF + c. c > 0 deja un
                          espacio con el hueso; c < 0 es interferencia (press-fit).
  tapón intramedular      extensión del andamio dentro del canal medular de cada
                          extremo huésped, de longitud L (mm), con holgura radial.
                          El canal se obtiene de la segmentación CRUDA (la
                          convención Fischer excluye el canal: canal = relleno 2D
                          del corte − hueso). Solo entra en la parte del canal que
                          queda frente a la cara del defecto.
  canal para clavo        cilindro (eje + radio) que se resta del andamio, para un
                          clavo intramedular presente en la TC (detección por HU de
                          metal) o de diámetro indicado. Un tapón y un clavo pueden
                          coexistir: el tapón queda anular alrededor del clavo.

Arquitectura
------------
  gradiente radial        (solo 'sheet') espesor de pared t(p) según la profundidad
                          radial p (mm), medida en cada corte ⟂ al eje desde el
                          contorno externo de la envolvente (no desde las caras extremas):
                          t_ext para p <= p1, interpolación lineal hasta t_int en
                          p2, t_int más adentro (más denso en la zona cortical).
  interconectividad       % del volumen poroso conectado con el exterior de la
                          envolvente (conectividad 6, criterio conservador);
                          poros cerrados = el resto.
  superficie específica   área de la malla del andamio / volumen de la envolvente
                          (mm^-1) y / volumen sólido.

Fabricabilidad (sobre el sólido REAL recortado por la envolvente, no sobre la
retícula ideal): espesor local de pared y de poro (Hildebrand & Rüegsegger, 1997)
en tres tramos (extremo proximal, centro, extremo distal); fracción del volumen
sólido INTERIOR (a > 1 mm de la superficie) con pared < mínimo del material;
percentil 5 del poro frente al mínimo; número de componentes sólidas (debe ser 1).
La banda de 1 mm junto a la superficie se reporta aparte: allí el recorte deja
aristas en cuña cuyo espesor local tiende a 0 por definición.

Coordenadas: todo trabaja en la grilla (k, j, i) de la ROI orientada, k = eje largo,
spacing sp (mm) en el mismo orden. La grilla fina del andamio empieza en el índice
continuo field['origin_idx'] de la ROI con paso field['h_axes'] (mm).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import scaffold_core as sc

# --------------------------------------------------------------------- material
# Valores de diseño. Cada número lleva su fuente; "None" = no definido (no inventar).
MATERIALS = {
    "Ti6Al4V_LPBF": dict(
        nombre="Ti-6Al-4V ELI fabricado por fusión láser en lecho de polvo (LPBF)",
        # E: 108-115 GPa en tres fuentes LPBF: Vrancken et al. (2012) 109,2 (sin tratar) y 114,7 (850 C/2 h);
        # Shah et al. (2024) 108 (sin tratar, J. Mater. Res. Technol. 33:1756); Luo et al. (2023) 108,09.
        E_MPa=110000.0,
        nu=0.32,                 # Tevet et al. (2022), Materials 15:638, SLM sin tratar, ultrasonido: media de 6 valores nu_ij = 0,323 (0,303-0,342); Luo et al. (2023) usa 0,30
        densidad_g_cm3=4.43,     # Luo et al. (2023)
        fluencia_MPa=938.0,      # menor valor medido entre fuentes LPBF: Shah et al. (2024) 938; Vrancken et al. (2012) 955 (tratado) y 1110 (sin tratar)
        pared_min_mm=0.20,       # McGregor et al. (2021), Addit. Manuf.: espesores fabricados por PBF 200–1669 µm
        poro_min_mm=0.30,        # Karageorgiou & Kaplan (2005): poro > 300 µm (criterio biológico, ya en el Cap. II)
    ),
    "PCL_bTCP_extrusion": dict(
        nombre="PCL/β-TCP 80:20 por extrusión de material (alternativa biodegradable)",
        # Material macizo (filamento, tracción): Bruyas et al. (2018), J. Mater. Res.,
        # doi:10.1557/jmr.2018.112. Con 20 % de β-TCP: E = 355 MPa, resistencia 12,4 MPa.
        E_MPa=355.0,
        nu=0.41,                 # Lu et al. (2014), J. Appl. Biomater. Funct. Mater. 12:145, medido con galgas: 0,442 PCL y 0,408 PCL/HA 80:20 (análogo con fosfato de calcio al 20 %)
        densidad_g_cm3=None,     # no verificado (no se usa en el FEA estático)
        fluencia_MPa=12.4,       # Bruyas et al. (2018): resistencia a tracción "at zero slope", 20 % β-TCP
        pared_min_mm=0.40,       # Bruyas et al. (2018): boquilla de 500 µm, struts de 350–400 µm; se toma el extremo superior
        poro_min_mm=0.30,        # Karageorgiou & Kaplan (2005), igual que el titanio
        # FEA por etapas: factores sobre las propiedades iniciales. Se miden sobre el andamio
        # (propiedad aparente); aplicarlos al material macizo es un supuesto del proyecto.
        etapas_degradacion=[
            dict(etapa="t0", meses=0, factor_E=1.00, factor_fluencia=1.00, fuente="propiedades iniciales"),
            dict(etapa="6m_optimista", meses=6, factor_E=round(12.0 / 16.9, 3), factor_fluencia=None,
                 fuente="Lam et al. (2009), J Biomed Mater Res A 90:906-919, PCL/TCP in vivo (conejo): rigidez 16,9 -> 12,0 MPa; sin dato de fluencia"),
            dict(etapa="6m_pesimista", meses=6, factor_E=round(7.6 / 23.1, 3), factor_fluencia=round(4.01 / 6.38, 3),
                 fuente="Yeo et al. (2008), J Biomed Mater Res A 84:208-218, PCL-TCP 80:20 in vivo (rata), 24 semanas: módulo 23,1 -> 7,6 MPa; resistencia 6,38 -> 4,01 MPa"),
        ],
    ),
}

FAB_CRITERIA = dict(
    frac_pared_delgada_max=0.01,   # criterio del proyecto (propuesto): <= 1 % del volumen sólido bajo el mínimo
    poro_percentil=5,              # se compara el p5 del poro con el mínimo
    componentes_solidas_max=1,
)


# --------------------------------------------------------------------- utilidades
def k_range(mask):
    ks = np.nonzero(mask.any(axis=(1, 2)))[0]
    return int(ks.min()), int(ks.max())


def _largest_2d(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    s = np.bincount(lab.ravel())[1:]
    return lab == (1 + int(np.argmax(s)))


def canal_slice(bone2d):
    """Canal medular de un corte ⟂ al eje: relleno 2D del hueso − hueso (mayor componente)."""
    if not bone2d.any():
        return np.zeros_like(bone2d)
    c = ndi.binary_fill_holes(bone2d) & ~bone2d
    return _largest_2d(c) if c.any() else c


def fine_coords_mm(field, sp, axis):
    """Coordenada (mm, sistema k,j,i de la ROI) de cada muestra de la grilla fina en un eje."""
    n = field["f"].shape[axis]
    idx = field["origin_idx"][axis] + np.arange(n) * field["h_axes"][axis] / sp[axis]
    return (idx * sp[axis]).astype(np.float32)


# --------------------------------------------------------------------- tapón intramedular
def intramedullary_plugs(defect, bone_raw, sp, length_mm, clearance_mm=0.0):
    """
    Máscara (grilla ROI) de los tapones intramedulares en ambos extremos del defecto.
    bone_raw: segmentación cruda (bool) con el canal SIN rellenar.
    Solo se conserva la parte del canal que queda frente a la cara extrema del defecto.
    """
    plug = np.zeros_like(defect, dtype=bool)
    if length_mm <= 0:
        return plug, dict(activo=False)
    k0, k1 = k_range(defect)
    n = int(np.ceil(length_mm / sp[0]))
    info = dict(activo=True, longitud_mm=float(length_mm), holgura_radial_mm=float(clearance_mm))
    for side, face, ks in (("proximal", k0, range(k0 - 1, k0 - n - 1, -1)),
                           ("distal", k1, range(k1 + 1, k1 + n + 1))):
        face_proj = defect[face]
        diam, nk = [], 0
        for k in ks:
            if k < 0 or k >= defect.shape[0]:
                break
            c = canal_slice(bone_raw[k])
            if clearance_mm > 0 and c.any():
                c &= ndi.distance_transform_edt(c, sampling=sp[1:]) > clearance_mm
            c &= face_proj
            if not c.any():
                break                         # el canal se cerró o no enfrenta al defecto
            plug[k] = c
            nk += 1
            diam.append(2 * np.sqrt(c.sum() * sp[1] * sp[2] / np.pi))
        info[side] = dict(longitud_real_mm=float(nk * sp[0]),
                          diametro_equivalente_mm=float(np.median(diam)) if diam else 0.0)
    return plug, info


# --------------------------------------------------------------------- clavo intramedular
def detect_nail(ct, defect, sp, hu_metal=2000.0, margin_mm=15.0):
    """
    Eje y radio del clavo intramedular a partir de la TC de la ROI (HU).
    Toma la componente metálica más grande que cruza el tramo del defecto (± margen),
    ajusta una recta (mínimos cuadrados j(k), i(k)) a los centroides por corte y usa
    la mediana del radio equivalente. El blooming del metal agranda el radio: queda
    del lado seguro (canal algo más ancho).
    Devuelve None si no hay metal.
    """
    k0, k1 = k_range(defect)
    m = int(np.ceil(margin_mm / sp[0]))
    a, b = max(k0 - m, 0), min(k1 + m + 1, ct.shape[0])
    metal = ct[a:b] >= hu_metal
    if not metal.any():
        return None
    lab, n = ndi.label(metal, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    comp = lab == (1 + int(np.argmax(sizes)))
    ks, cj, ci, r = [], [], [], []
    for kk in range(comp.shape[0]):
        s = ndi.binary_fill_holes(comp[kk])          # clavos canulados: cuenta el hueco central
        if s.sum() < 3:
            continue
        jj, ii = np.nonzero(s)
        ks.append((a + kk) * sp[0]); cj.append(jj.mean() * sp[1]); ci.append(ii.mean() * sp[2])
        r.append(np.sqrt(s.sum() * sp[1] * sp[2] / np.pi))
    if len(ks) < 5:
        return None
    ks, cj, ci, r = map(np.asarray, (ks, cj, ci, r))
    # Cortes con tornillos de bloqueo (transversales al clavo): área mucho mayor y
    # centroide desplazado. Se excluyen (área > 1,3 × mediana) y se reajusta la recta
    # descartando residuos > 1 mm, en dos pasadas.
    keep = r <= 1.3 * np.median(r)
    for _ in range(2):
        pj, pi_ = np.polyfit(ks[keep], cj[keep], 1), np.polyfit(ks[keep], ci[keep], 1)
        resid = np.sqrt((cj - np.polyval(pj, ks)) ** 2 + (ci - np.polyval(pi_, ks)) ** 2)
        keep = keep & (resid <= 1.0)
    d = np.array([1.0, pj[0], pi_[0]]); d /= np.linalg.norm(d)
    p0 = np.array([0.0, pj[1], pi_[1]])
    return dict(p0_mm=p0.tolist(), dir=d.tolist(), radio_mm=float(np.median(r[keep])),
                n_cortes=int(len(ks)), n_cortes_usados=int(keep.sum()),
                residuo_eje_max_mm=float(resid[keep].max()), hu_metal=float(hu_metal),
                tramo_k_mm=[float(ks.min()), float(ks.max())])


def apply_channel(field, sp, nail, clearance_mm=0.5, diameter_mm=None):
    """Resta del andamio (y de la envolvente) un cilindro de radio r + holgura alrededor del eje del clavo."""
    R = np.float32((diameter_mm / 2 if diameter_mm else nail["radio_mm"]) + clearance_mm)
    p0 = np.asarray(nail["p0_mm"], np.float32); d = np.asarray(nail["dir"], np.float32)
    zk, zj, zi = (fine_coords_mm(field, sp, ax) for ax in range(3))
    for kk in range(field["f"].shape[0]):           # por corte: memoria acotada
        vk = zk[kk] - p0[0]
        vj = (zj - p0[1])[:, None]; vi = (zi - p0[2])[None, :]
        t = vk * d[0] + vj * d[1] + vi * d[2]
        dist = np.sqrt(np.maximum(vk ** 2 + vj ** 2 + vi ** 2 - t ** 2, 0))
        cut = R - dist                                # > 0 dentro del canal
        np.maximum(field["f"][kk], cut, out=field["f"][kk])
        np.maximum(field["env"][kk], cut, out=field["env"][kk])
    return dict(radio_canal_mm=float(R), holgura_mm=float(clearance_mm),
                diametro_clavo_mm=float(diameter_mm) if diameter_mm else float(2 * nail["radio_mm"]),
                diametro_origen="indicado" if diameter_mm else "detectado en TC")


# --------------------------------------------------------------------- campo con gradiente
def build_field(defect, sp, h, cell_mm, kind, mode, w, clearance_mm=0.0, grad=None, margin_mm=1.0):
    """
    Igual que sc.build_scaffold_field; si grad = dict(t_ext, t_int, p1, p2) (mm) y modo
    'sheet', el espesor de pared varía con la profundidad desde la superficie externa.
    """
    if grad is None:
        return sc.build_scaffold_field(defect, sp, h=h, cell_mm=cell_mm, kind=kind, mode=mode, w=w,
                                       clearance_mm=clearance_mm, margin_mm=margin_mm)
    if mode != "sheet":
        raise ValueError("El gradiente radial está definido solo para el modo 'sheet'")
    lo, hi = sc._crop_box(defect, sp, margin_mm)
    crop = defect[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sdf = ndi.gaussian_filter(sc.signed_distance(crop, sp), 0.5)
    env, h_axes = sc._resample_to_fine(sdf, sp, h)
    del sdf
    env += np.float32(clearance_mm)
    s = 2 * np.pi / cell_mm
    ax = [np.arange(n, dtype=np.float32) * np.float32(hh * s) for n, hh in zip(env.shape, h_axes)]
    G = sc.TPMS[kind](ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]).astype(np.float32)
    gs = np.float32(sc.grad_scale(kind) * s)
    depth = radial_depth(env <= 0, h_axes)
    t_ext, t_int, p1, p2 = (np.float32(grad[k]) for k in ("t_ext", "t_int", "p1", "p2"))
    frac = np.clip((depth - p1) / max(p2 - p1, 1e-6), 0, 1)
    t = t_ext + (t_int - t_ext) * frac
    np.abs(G, out=G)
    G /= gs
    G -= 0.5 * t                                     # (|G| - w)/gs con w = t·gs/2
    np.maximum(G, env, out=G)
    return dict(f=G, env=env, h_axes=h_axes, h=float(np.mean(h_axes)),
                params=dict(kind=kind, mode=mode, w=float(w), cell_mm=float(cell_mm), gradiente=dict(grad)),
                origin_idx=np.array(lo, float), lo=lo, hi=hi)


def radial_depth(env_mask, h_axes):
    """Profundidad radial (mm): distancia en el plano del corte (⟂ eje largo) al contorno
    externo de la envolvente. No cuenta las caras extremas, así el gradiente es radial."""
    out = np.zeros(env_mask.shape, np.float32)
    for kk in range(env_mask.shape[0]):
        m = ndi.binary_fill_holes(env_mask[kk])
        if m.any():
            out[kk] = ndi.distance_transform_edt(m, sampling=h_axes[1:])
    return out


# --------------------------------------------------------------------- métricas F4
def interconnectivity(field, solid):
    env = field["env"] <= 0
    pore = env & ~solid
    if not pore.any():
        return dict(interconectividad_pct=float("nan"))
    lab, n = ndi.label(pore)                          # conectividad 6
    border = env & ~ndi.binary_erosion(env)
    open_ids = np.unique(lab[border & pore])
    open_ids = open_ids[open_ids > 0]
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    v_open = sizes[open_ids].sum()
    return dict(interconectividad_pct=100.0 * float(v_open) / float(pore.sum()),
                poros_cerrados_n=int(n - open_ids.size),
                poros_cerrados_vol_pct=100.0 * float(pore.sum() - v_open) / float(pore.sum()))


def mesh_area(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _ridge_seeds(d, h):
    """Centros candidatos: vóxeles cuya distancia no crece a razón ~1 mm/mm hacia ningún vecino 26.
    A paso fino la cresta rara vez es un máximo local estricto (que es lo que usa sc.local_thickness,
    pensado para la grilla gruesa), por eso se usa la pendiente. Sobran semillas sin error: toda
    esfera inscrita es válida; solo cuestan tiempo."""
    h = np.asarray(h, float)
    flat = np.ones(d.shape, bool)
    for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
              (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)):
        o = np.array(o)
        step = float(np.linalg.norm(o * h))
        shifted = np.roll(d, tuple(-o), axis=(0, 1, 2))     # d(x + o)
        flat &= (shifted - d) / step < 0.9                   # subida hacia x + o
        shifted = np.roll(d, tuple(o), axis=(0, 1, 2))       # d(x - o)
        flat &= (shifted - d) / step < 0.9                   # subida hacia x - o
    return flat


def local_thickness_fine(mask, h, max_bins=24):
    """Espesor local (misma definición de sc.local_thickness: diámetro de la mayor esfera inscrita que
    contiene al vóxel), con dos cambios para la grilla fina: semillas por pendiente (_ridge_seeds) y
    radio en bandas de h/2, con diámetro = 2·d sin corrección. En láminas oblicuas de 0,30-0,53 mm y
    cilindros de 0,75 y 1,35 mm (espesor conocido, h = 0,025) 2·d queda entre -0,02 y 0 mm de la verdad,
    y en láminas alineadas con la grilla, a ±0,03; 2·d - h/2 y 2·d - h sesgan más hacia abajo. Si hay más de max_bins bandas se reparten entre el radio mínimo y el máximo,
    y la cuantización del diámetro pasa a (r_máx − r_mín)·2/max_bins. Devuelve mapa (mm), NaN fuera."""
    hm = float(np.mean(h))
    d = ndi.distance_transform_edt(mask, sampling=h).astype(np.float32)
    seeds = mask & _ridge_seeds(d, h)
    step = hm / 2
    rmax, rmin = float(d[seeds].max()), float(d[seeds].min())
    nb = int(np.ceil((rmax - rmin) / step)) + 1
    edges = np.linspace(rmax, rmin, min(nb, max_bins))
    th = np.full(mask.shape, np.nan, np.float32)
    todo = mask.copy()
    for r in edges:
        s = seeds & (d >= r)
        if not s.any():
            continue
        cover = todo & (ndi.distance_transform_edt(~s, sampling=h) <= r)
        th[cover] = max(2 * r, hm)
        todo &= ~cover
        if not todo.any():
            break
    th[todo] = np.maximum(2 * d[todo], hm)
    return th


def _block_field(field, solid, lo, hi, ratio):
    """Reconstruye el andamio en la caja de la grilla gruesa [lo, hi) a paso h/ratio.
    Mismo campo que el andamio real: fase TPMS anclada a la grilla gruesa (las muestras comunes
    coinciden), mismo umbral, y envolvente (con el canal del clavo ya restado) interpolada
    trilinealmente como en _resample_to_fine. Filtro de una pieza: se conservan las componentes
    finas que tocan el sólido grueso ya filtrado por sc.clean_fragments."""
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    n_c = np.array([b - a for a, b in zip(lo, hi)])
    n_f = (n_c - 1) * ratio + 1
    zf = n_f / n_c
    env = ndi.zoom(field["env"][sl], zf, order=1, grid_mode=False, mode="nearest")
    p = field["params"]
    s = 2 * np.pi / p["cell_mm"]
    ax = [((a + np.arange(m, dtype=np.float64) / ratio) * hh * s).astype(np.float32)
          for a, m, hh in zip(lo, n_f, field["h_axes"])]
    G = sc.TPMS[p["kind"]](ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]).astype(np.float32)
    gs = np.float32(sc.grad_scale(p["kind"]) * s)
    if "gradiente" in p:
        g = p["gradiente"]
        depth = ndi.zoom(radial_depth(field["env"] <= 0, field["h_axes"])[sl], zf, order=1, grid_mode=False, mode="nearest")
        frac = np.clip((depth - g["p1"]) / max(g["p2"] - g["p1"], 1e-6), 0, 1)
        f = np.abs(G) / gs - 0.5 * (g["t_ext"] + (g["t_int"] - g["t_ext"]) * frac)
    elif p["mode"] == "sheet":
        f = (np.abs(G) - np.float32(p["w"])) / gs
    else:
        f = (G - np.float32(p["w"])) / gs
    S = np.maximum(f, env) <= 0
    lab, nl = ndi.label(S, structure=np.ones((3, 3, 3)))
    if nl > 1:
        rr = (slice(None, None, ratio),) * 3
        ids = np.unique(lab[rr][solid[sl]])
        S = np.isin(lab, ids[ids > 0])
    return S, env <= 0


def _block_centers(inner, n):
    """Hasta n centros repartidos por ángulo alrededor del centroide de la zona interior del tramo."""
    idx = np.argwhere(inner)
    if not len(idx):
        return []
    c = idx.mean(axis=0)
    ang = np.arctan2(idx[:, 1] - c[1], idx[:, 2] - c[2])
    order = np.argsort(ang)
    picks = order[((np.arange(n) + 0.5) / n * len(order)).astype(int)]
    return [idx[i] for i in picks]


def fine_thickness(field, solid, slabs, h_fine, poro_mm, n_blocks=3, border_mm=1.0):
    """
    Pared y poro medidos a paso h_fine (mm) con el campo reconstruido. Reconstruir un tramo entero a
    0,025 mm no entra en memoria (hasta ~1,6e9 vóxeles en pat_02), así que en cada tramo se evalúan
    n_blocks cajas de una celda de lado, repartidas por ángulo y centradas en la zona interior, con un
    margen de 0,6 poros para que las esferas inscritas no se corten en la caja.
    slabs: {tramo: (a, b)} índices k gruesos de la zona evaluada; devuelve vóxeles por tramo.
    """
    h = np.array(field["h_axes"], float)
    ratio = max(int(round(float(np.mean(h)) / h_fine)), 1)
    hf = tuple(h / ratio)
    cell = field["params"]["cell_mm"]
    half = np.ceil((cell / 2 + max(0.6 * poro_mm, 0.3)) / h).astype(int)
    core = np.ceil((cell / 2) / h).astype(int)
    env = field["env"] <= 0
    res = {}
    for name, (a, b) in slabs.items():
        sub = np.zeros_like(env); sub[a:b] = env[a:b]
        d_in = ndi.distance_transform_edt(env[a:b], sampling=tuple(h))
        inner = np.zeros_like(env); inner[a:b] = d_in > border_mm + cell / 2
        if not inner.any():
            inner[a:b] = d_in > border_mm                      # envolvente delgada: centro lo más adentro posible
        if not inner.any():
            inner = sub
        w, p, wb, nb = [], [], [], 0
        for c in _block_centers(inner, n_blocks):
            lo = np.maximum(c - half, 0); hi = np.minimum(c + half + 1, env.shape)
            S, E = _block_field(field, solid, lo, hi, ratio)
            cl = (np.minimum(c - core, hi - 1) - lo) * ratio
            cl = np.maximum(cl, 0)
            ch = (np.minimum(c + core + 1, hi) - lo) * ratio
            ev = np.zeros(S.shape, bool)
            ev[cl[0]:ch[0], cl[1]:ch[1], cl[2]:ch[2]] = True
            if not (S & ev).any():
                continue
            # distancia a la superficie medida en la grilla gruesa (la caja no ve toda la envolvente)
            sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
            big = tuple(slice(max(int(x) - 20, 0), int(y) + 20) for x, y in zip(lo, hi))
            dc_in = ndi.distance_transform_edt(env[big], sampling=tuple(h))
            dc_out = ndi.distance_transform_edt(~env[big], sampling=tuple(h))
            rel = tuple(slice(int(x) - bb.start, int(y) - bb.start) for x, y, bb in zip(lo, hi, big))
            zf = np.array(S.shape) / np.array([s.stop - s.start for s in sl])
            inn = ndi.zoom(dc_in[rel], zf, order=1, grid_mode=False, mode="nearest") > border_mm
            near = ndi.zoom(dc_out[rel], zf, order=1, grid_mode=False, mode="nearest") <= 1.0
            tw = local_thickness_fine(S, hf)
            tp = local_thickness_fine(~S & near, hf)
            w.append(tw[S & ev & inn]); wb.append(tw[S & ev & ~inn]); p.append(tp[~S & E & ev])
            nb += 1
            del S, E, tw, tp, inn, near, ev
        res[name] = dict(w=np.concatenate(w) if w else np.zeros(0), p=np.concatenate(p) if p else np.zeros(0),
                         wb=np.concatenate(wb) if wb else np.zeros(0), n_bloques=nb)
    return res, dict(h_fina_mm=[float(x) for x in hf], factor=ratio, bloques_por_tramo=int(n_blocks),
                     lado_nucleo_mm=float(cell), margen_mm=float(max(0.6 * poro_mm, 0.3)),
                     nota="muestra de cajas por tramo, no el tramo completo")


def fabricability(field, solid, material, n_frag_kept, slab_mm=8.0, guard_mm=1.0, border_mm=1.0,
                  fab_h=None, poro_mm=None, n_blocks=3):
    """
    Espesor local de pared y poro sobre el sólido real en tres tramos del eje largo
    (extremos y centro). Se evalúa lejos de los planos de corte del tramo (guard_mm).
    Si fab_h (mm), pared y poro se miden con el campo reconstruido a ese paso (fine_thickness) y lo
    medido en la grilla del andamio queda en out['medicion_grilla']. Sin fab_h no cambia nada.
    """
    mat = MATERIALS[material]
    h = tuple(float(x) for x in field["h_axes"])
    env = field["env"] <= 0
    ks = np.nonzero(env.any(axis=(1, 2)))[0]
    k0, k1 = int(ks.min()), int(ks.max())
    ns, ng = int(round(slab_mm / h[0])), int(round(guard_mm / h[0]))
    starts = {"proximal": k0, "centro": (k0 + k1) // 2 - ns // 2, "distal": k1 - ns + 1}
    walls, pores, edges = [], [], []
    per = {}
    for name, a in starts.items():
        a0, b0 = max(a - ng, 0), min(a + ns + ng, env.shape[0])
        S = solid[a0:b0]; E = env[a0:b0]
        ev = np.zeros(S.shape, bool); ev[(a - a0) + ng:(a - a0) + ns - ng] = True
        # interior = a más de borde_mm de la superficie de la envolvente. En el borde, el
        # recorte de la retícula deja aristas en cuña cuyo espesor local tiende a 0 por
        # definición (esfera inscrita); se reportan aparte y no deciden el chequeo.
        inner = ndi.distance_transform_edt(E, sampling=h) > border_mm
        tw = sc.local_thickness(S, h)
        # poro: espacio no sólido dentro de la envolvente + banda exterior de 1 mm (los
        # poros abiertos al borde no se recortan artificialmente; y se evita que el canal
        # del clavo o el exterior, de radio grande, vuelvan gruesa la discretización)
        dom = ~S & (ndi.distance_transform_edt(~E, sampling=h) <= 1.0)
        tp = sc.local_thickness(dom, h)
        w = tw[S & ev & inner]; p = tp[~S & E & ev]
        wb = tw[S & ev & ~inner]
        walls.append(w); pores.append(p); edges.append(wb)
        per[name] = dict(pared_p5_mm=float(np.percentile(w, 5)) if w.size else float("nan"),
                         poro_p5_mm=float(np.percentile(p, 5)) if p.size else float("nan"))
    w = np.concatenate(walls); p = np.concatenate(pores); wb = np.concatenate(edges)
    medicion_grilla = None
    if fab_h:
        medicion_grilla = dict(h_mm=[float(x) for x in h], pared_real=dict(p1=float(np.percentile(w, 1)), p5=float(np.percentile(w, 5)),
                                                                          mediana=float(np.median(w))),
                               poro_real=dict(p5=float(np.percentile(p, FAB_CRITERIA["poro_percentil"])), mediana=float(np.median(p))),
                               frac_vol_pared_bajo_minimo=float((w < mat["pared_min_mm"]).mean()) if mat["pared_min_mm"] else None,
                               por_tramo=per)
        slabs = {n: (a + ng, a + ns - ng) for n, a in starts.items()}
        fine, info_fina = fine_thickness(field, solid, slabs, fab_h, poro_mm, n_blocks, border_mm)
        per = {n: dict(pared_p5_mm=float(np.percentile(r["w"], 5)) if r["w"].size else float("nan"),
                       pared_mediana_mm=float(np.median(r["w"])) if r["w"].size else float("nan"),
                       poro_p5_mm=float(np.percentile(r["p"], 5)) if r["p"].size else float("nan"),
                       poro_mediana_mm=float(np.median(r["p"])) if r["p"].size else float("nan"),
                       n_bloques=r["n_bloques"]) for n, r in fine.items()}
        w = np.concatenate([r["w"] for r in fine.values()])
        p = np.concatenate([r["p"] for r in fine.values()])
        wb = np.concatenate([r["wb"] for r in fine.values()])
    out = dict(material=material, pared_min_material_mm=mat["pared_min_mm"], poro_min_mm=mat["poro_min_mm"],
               pared_real=dict(p1=float(np.percentile(w, 1)), p5=float(np.percentile(w, 5)),
                               mediana=float(np.median(w))),
               poro_real=dict(p5=float(np.percentile(p, FAB_CRITERIA["poro_percentil"])),
                              mediana=float(np.median(p))),
               por_tramo=per, componentes_solidas=int(n_frag_kept), criterios=dict(FAB_CRITERIA))
    checks = {}
    if mat["pared_min_mm"] is not None:
        frac = float((w < mat["pared_min_mm"]).mean())
        out["frac_vol_pared_bajo_minimo"] = frac
        out["borde_recortado"] = dict(banda_mm=border_mm,
                                      frac_vol_bajo_minimo=float((wb < mat["pared_min_mm"]).mean()) if wb.size else 0.0,
                                      frac_vol_bajo_minimo_sobre_total=float((wb < mat["pared_min_mm"]).sum()) / max(w.size + wb.size, 1))
        checks["pared"] = frac <= FAB_CRITERIA["frac_pared_delgada_max"]
    if mat["poro_min_mm"] is not None:
        checks["poro"] = out["poro_real"]["p5"] >= mat["poro_min_mm"]
    checks["una_componente"] = n_frag_kept <= FAB_CRITERIA["componentes_solidas_max"]
    out["chequeos"] = checks
    out["fabricable"] = bool(all(checks.values())) if mat["pared_min_mm"] is not None else None
    if fab_h:
        out["medicion_fina"] = info_fina
        out["medicion_grilla"] = medicion_grilla
    return out
