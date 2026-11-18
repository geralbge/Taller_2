"""
Forma de referencia CONTRALATERAL: tibia de la otra pierna, reflejada y
registrada rígidamente sobre las zonas sanas de la tibia afectada.

Fundamento: la reconstrucción con la anatomía contralateral reflejada es la
práctica habitual en planificación de implantes específicos de paciente
(Tetsworth et al., 2017, SICOT-J 3:16; revisado en Rezapourian et al., 2026,
Biomimetics 11(2):145).

Procedimiento
-------------
1) Puntos de superficie (LPS, mm) de ambas tibias segmentadas (rellenas), cada
   una en su ROI orientada (ambas ROI vienen de la MISMA TC -> mismo marco físico).
2) Reflexión de la contralateral respecto del plano sagital: x -> -x (LPS).
   (Cualquier plano sirve: la reflexión + el registro rígido equivalen a la
   reflexión respecto del plano óptimo.)
3) Inicialización: alinea el eje largo (k) de la ROI contralateral reflejada con
   el de la afectada y los centroides de las zonas de comparación.
4) ICP punto a punto con recorte decreciente (10 -> 2 mm), usando solo
   la superficie SANA de la tibia afectada a ambos lados del defecto
   (ventana D1–D2 excluida, hasta ICP_RANGE_MM de cada lado).
5) Para cada vóxel de la ROI afectada dentro de la ventana [k1, k2] se toma el
   valor de la tibia contralateral transformada (interpolación trilineal >= 0,5).

QC: RMS del registro sobre la superficie sana (mm) y fracción de pares usados.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ICP_RANGE_MM = 60.0     # superficie sana usada a cada lado del defecto
ICP_GAP_MM = 3.0        # margen adicional excluido junto a D1/D2
ICP_TRIM_MM = 10.0      # umbral inicial de recorte de pares (mm), decrece por iteración
ICP_TRIM_FINAL_MM = 2.0 # umbral final
ICP_ITERS = 250
QC_INLIER_MM = 3.0      # QC: fracción de la superficie sana a <= 3 mm tras el registro
MAX_POINTS = 40000
MIRROR = np.diag([-1.0, 1.0, 1.0])


# ------------------------------------------------------------------ geometría
def img_geometry(img):
    sp_xyz = np.array(img.GetSpacing(), float)
    org = np.array(img.GetOrigin(), float)
    D = np.array(img.GetDirection(), float).reshape(3, 3)
    return sp_xyz, org, D


def idx_kji_to_phys(idx_kji, img):
    sp_xyz, org, D = img_geometry(img)
    return org + (D @ (np.asarray(idx_kji, float)[:, ::-1] * sp_xyz).T).T


def phys_to_idx_kji(p, img):
    sp_xyz, org, D = img_geometry(img)
    ijk = (np.linalg.inv(D) @ (np.asarray(p, float) - org).T).T / sp_xyz
    return ijk[:, ::-1]


def surface_points(mask, img, kmin=None, kmax=None, max_points=MAX_POINTS, seed=0):
    m = mask.copy()
    if kmin is not None:
        m[:max(kmin, 0)] = False
    if kmax is not None:
        m[kmax + 1:] = False
    surf = m & ~ndi.binary_erosion(m)
    idx = np.argwhere(surf)
    if len(idx) > max_points:
        idx = idx[np.random.default_rng(seed).choice(len(idx), max_points, replace=False)]
    return idx_kji_to_phys(idx, img)


def kabsch(a, b):
    """R, t que minimizan ||R a + t - b||."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def rotation_between(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = float(np.dot(u, v)); w = np.cross(u, v); s = np.linalg.norm(w)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(np.array([1, 0, 0]), np.array([1, 0, 0]))
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / s
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


# ------------------------------------------------------------------ registro
def register(contra_mask, contra_img, aff_mask, aff_img, k1, k2, sp_k):
    """
    Devuelve dict con R, t (aplican a puntos contralaterales YA reflejados),
    rms_mm, inlier_frac, n_points.
    """
    gap = int(round(ICP_GAP_MM / sp_k)); rng = int(round(ICP_RANGE_MM / sp_k))
    dst = np.vstack([surface_points(aff_mask, aff_img, k1 - gap - rng, k1 - gap, seed=1),
                     surface_points(aff_mask, aff_img, k2 + gap, k2 + gap + rng, seed=2)])
    if len(dst) < 100:
        raise RuntimeError("Muy poca superficie sana alrededor del defecto para registrar la contralateral")
    src_all = surface_points(contra_mask, contra_img, seed=3) @ MIRROR.T

    # inicialización: eje largo (columna k de la dirección de cada ROI) y centroides
    _, _, Da = img_geometry(aff_img); _, _, Dc = img_geometry(contra_img)
    R0 = rotation_between(MIRROR @ Dc[:, 2], Da[:, 2])
    src0 = src_all @ R0.T
    # centroide: misma extensión axial que la zona de comparación, centrada por proyección en el eje
    ax = Da[:, 2]
    pd = dst @ ax; ps = src0 @ ax
    t_ax = (np.median(pd) - np.median(ps))
    sel = np.abs(ps + t_ax - np.median(pd)) <= (pd.max() - pd.min()) / 2 + 20
    t0 = dst.mean(0) - src0[sel].mean(0)
    t0 = t0 - ax * (t0 @ ax) + ax * t_ax
    R, t = R0, t0

    # ICP con umbral de recorte decreciente (10 mm -> ICP_TRIM_FINAL_MM)
    for it in range(ICP_ITERS):
        cur = src_all @ R.T + t
        dist, j = cKDTree(cur).query(dst)
        thr = max(ICP_TRIM_FINAL_MM, ICP_TRIM_MM * (0.97 ** it))
        ok = dist <= thr
        if ok.sum() < 50:
            break
        dR, dt = kabsch(cur[j[ok]], dst[ok])
        R, t = dR @ R, dR @ t + dt
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if thr <= ICP_TRIM_FINAL_MM and np.linalg.norm(dt) < 1e-4 and ang < 1e-5:
            break
    cur = src_all @ R.T + t
    dist, _ = cKDTree(cur).query(dst)
    ok = dist <= QC_INLIER_MM
    return dict(R=R, t=t, rms_mm=float(np.sqrt(np.mean(dist[ok] ** 2))) if ok.any() else float("nan"),
                median_mm=float(np.median(dist)), inlier_frac=float(ok.mean()), n_points=int(len(dst)),
                iterations=it + 1)


def reference_from_contralateral(contra_mask, contra_img, aff_shape, aff_img, k1, k2, reg, bbox_ji=None):
    """Máscara de referencia (forma contralateral) en la grilla afectada, solo en [k1, k2]."""
    ref = np.zeros(aff_shape, bool)
    j0, j1, i0, i1 = 0, aff_shape[1] - 1, 0, aff_shape[2] - 1
    if bbox_ji is not None:
        j0, j1, i0, i1 = bbox_ji
    kk, jj, ii = np.meshgrid(np.arange(k1, k2 + 1), np.arange(j0, j1 + 1), np.arange(i0, i1 + 1),
                             indexing="ij")
    idx = np.stack([kk.ravel(), jj.ravel(), ii.ravel()], 1)
    q = idx_kji_to_phys(idx, aff_img)
    # q = R (M p) + t  ->  p = M^-1 R^T (q - t)
    p = ((q - reg["t"]) @ reg["R"]) @ MIRROR.T
    c = phys_to_idx_kji(p, contra_img)
    vals = ndi.map_coordinates(contra_mask.astype(np.float32), c.T, order=1, mode="constant", cval=0.0)
    ref[k1:k2 + 1, j0:j1 + 1, i0:i1 + 1] = (vals >= 0.5).reshape(kk.shape)
    return ref


def bbox_ji(mask, sp, margin_mm=15.0):
    """Caja (j0, j1, i0, i1) del hueso afectado en el plano, con margen."""
    jj = np.nonzero(mask.any(axis=(0, 2)))[0]; ii = np.nonzero(mask.any(axis=(0, 1)))[0]
    mj, mi = int(round(margin_mm / sp[1])), int(round(margin_mm / sp[2]))
    return (max(jj.min() - mj, 0), min(jj.max() + mj, mask.shape[1] - 1),
            max(ii.min() - mi, 0), min(ii.max() + mi, mask.shape[2] - 1))
