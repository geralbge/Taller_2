"""
Etapa A del pipeline: post-procesamiento de la segmentación y reconstrucción 3D
de la tibia (superficie externa), con evaluación geométrica.

Para cada caso:
  1) Post-procesamiento (igual para predicción y referencia; sin parámetros
     ajustados con datos de prueba):
       - componente conexa 3D más grande (conectividad 26)
       - relleno de cavidad medular: fill_holes 2D por corte perpendicular al
         eje largo de la ROI + fill_holes 3D  (misma regla que la evaluación
         armonizada "tibia rellena")
  2) Malla: marching cubes (nivel 0,5) sobre la máscara con 2 vóxeles de
     margen (malla cerrada) y antialiasing gaussiano sigma=0,5 vóxel (evita
     aristas no-manifold por contactos diagonales; cambio de volumen ~0,1 %), vértices llevados a coordenadas físicas LPS [mm] con
     origin/spacing/direction de la ROI; suavizado Taubin (lambda=0,5,
     mu=-0,53, 10 iteraciones; preserva volumen).  STL binario, cabecera
     "SPACE=LPS" (3D Slicer lo interpreta en LPS).
  3) Control de calidad de malla: estanqueidad (cada arista en 2 caras),
     volumen de malla vs volumen de la máscara.
  4) Evaluación predicción vs referencia (ambas post-procesadas igual):
     distancias superficie-superficie simétricas (vértice -> vértice más
     cercano de la otra malla; las mallas tienen resolución de vóxel, el error
     de esta aproximación es menor a ~1/2 arista): media, mediana, p95, máx.

Casos: validación cruzada Fischer (44; predicción de cada tibia hecha por el
fold que NO la usó para entrenar) y prueba externa DRYAD (4).

Uso (desde la raíz del repositorio):
    .\\.venv_nnunet\\Scripts\\python.exe src\\reconstruction\\reconstruct_tibia.py
    opciones: --only fischer|dryad   --no-stl (no guarda STL, solo métricas)

Salidas:
    results/meshes/<conjunto>/<caso>_pred.stl, <caso>_ref.stl  (resolución completa)
    results/meshes/<conjunto>/<caso>_pred_cad.stl  (~2e4 caras, para SolidWorks)
    results/meshes/<conjunto>/<caso>_pred_filled.nii.gz (máscara post-procesada)
    results/evaluation/reconstruction/reconstruction_metrics.csv
    results/evaluation/reconstruction/reconstruction_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[2]
MODEL = (ROOT / "data" / "nnunet" / "nnUNet_results" / "Dataset502_TibiaFischer"
         / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
FISCHER_GT = ROOT / "data" / "nnunet" / "nnUNet_raw" / "Dataset502_TibiaFischer" / "labelsTr"
DRYAD = ROOT / "data" / "interim" / "dryad_external" / "nnunet_test"
MESH_OUT = ROOT / "results" / "meshes"
EVAL_OUT = ROOT / "results" / "evaluation" / "reconstruction"

TAUBIN_LAMBDA, TAUBIN_MU, TAUBIN_ITERS = 0.5, -0.53, 10


# ---------------------------------------------------------------- máscara
def largest_component(m):
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == np.argmax(sizes)


def fill(m):
    ax = int(np.argmax(m.shape))
    out = m.copy()
    for i in range(m.shape[ax]):
        idx = [slice(None)] * 3; idx[ax] = i
        out[tuple(idx)] = ndi.binary_fill_holes(m[tuple(idx)])
    return ndi.binary_fill_holes(out)


def postprocess(m):
    return fill(largest_component(m))


# ---------------------------------------------------------------- malla
ANTIALIAS_SIGMA_VOX = 0.5   # elimina aristas no-manifold por contactos diagonales de vóxeles
CAD_STEP = 4                # malla gruesa (~2e4 caras) importable como sólido en CAD


def mask_to_mesh(m, spacing_xyz, origin_xyz, direction_3x3, step=1):
    """m: array (z,y,x). Devuelve vértices en LPS [mm] y caras."""
    padded = ndi.gaussian_filter(np.pad(m.astype(np.float32), 2), ANTIALIAS_SIGMA_VOX)
    verts_zyx, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step)
    idx_xyz = verts_zyx[:, ::-1] - 2.0                      # índice continuo (x,y,z)
    phys = origin_xyz + (direction_3x3 @ (idx_xyz * spacing_xyz).T).T
    faces = faces.astype(np.int64)
    if mesh_volume(phys, faces) < 0:                        # normales hacia afuera
        faces = faces[:, ::-1].copy()
    return phys, faces


def taubin(verts, faces, lam=TAUBIN_LAMBDA, mu=TAUBIN_MU, iters=TAUBIN_ITERS):
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    W = sparse.diags(1.0 / deg) @ A
    v = verts.copy()
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v


def mesh_volume(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def is_watertight(faces):
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def write_stl(path, verts, faces, name="tibia"):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True); norm[norm == 0] = 1
    n = n / norm
    header = f"SPACE=LPS {name}".encode()[:80].ljust(80, b" ")
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n; rec["v"] = tri
    with open(path, "wb") as f:
        f.write(header); f.write(struct.pack("<I", len(faces))); f.write(rec.tobytes())


def surface_distances(va, vb):
    da, _ = cKDTree(vb).query(va)
    db, _ = cKDTree(va).query(vb)
    d = np.concatenate([da, db])
    return dict(mean_mm=float(d.mean()), median_mm=float(np.median(d)),
                p95_mm=float(max(np.percentile(da, 95), np.percentile(db, 95))),
                max_mm=float(max(da.max(), db.max())))


# ---------------------------------------------------------------- casos
def read(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return (arr, np.array(img.GetSpacing()), np.array(img.GetOrigin()),
            np.array(img.GetDirection()).reshape(3, 3), img)


def write_mask(arr, ref_img, path):
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(arr.astype(np.uint8)); out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path), useCompression=True)


def process(case, dataset, pred_path, ref_path, save_stl):
    pa, sp, org, D, pimg = read(pred_path)
    ra, sp2, org2, D2, _ = read(ref_path)
    if pa.shape != ra.shape or not np.allclose(sp, sp2, atol=1e-4) or not np.allclose(org, org2, atol=1e-3):
        raise RuntimeError(f"{case}: geometría pred/ref distinta")
    pm, rm = postprocess(pa == 1), postprocess(ra > 0)
    vox_mm3 = float(np.prod(sp))
    out = dict(case=case, dataset=dataset)
    meshes = {}
    for tag, m in (("pred", pm), ("ref", rm)):
        v, f = mask_to_mesh(m, sp, org, D)
        v = taubin(v, f)
        vol_mesh = abs(mesh_volume(v, f)) / 1000
        vol_mask = m.sum() * vox_mm3 / 1000
        out[f"{tag}_n_vertices"] = len(v); out[f"{tag}_n_faces"] = len(f)
        out[f"{tag}_watertight"] = is_watertight(f)
        out[f"{tag}_vol_mask_cm3"] = round(vol_mask, 3)
        out[f"{tag}_vol_mesh_cm3"] = round(vol_mesh, 3)
        out[f"{tag}_mesh_vs_mask_pct"] = round(100 * (vol_mesh - vol_mask) / vol_mask, 3)
        meshes[tag] = v
        if save_stl:
            d = MESH_OUT / dataset; d.mkdir(parents=True, exist_ok=True)
            write_stl(d / f"{case}_{tag}.stl", v, f, f"{case}_{tag}")
            if tag == "pred":
                vc, fc = mask_to_mesh(m, sp, org, D, step=CAD_STEP)
                vc = taubin(vc, fc)
                write_stl(d / f"{case}_{tag}_cad.stl", vc, fc, f"{case}_{tag}_cad")
                out["pred_cad_n_faces"] = len(fc)
                out["pred_cad_watertight"] = is_watertight(fc)
    if save_stl:
        write_mask(pm, pimg, MESH_OUT / dataset / f"{case}_pred_filled.nii.gz")
    out["components_removed_pred"] = int(ndi.label(pa == 1, structure=np.ones((3, 3, 3)))[1] - 1)
    out.update({f"surf_{k}": round(v, 4) for k, v in surface_distances(meshes["pred"], meshes["ref"]).items()})
    out["vol_diff_pct"] = round(100 * (out["pred_vol_mesh_cm3"] - out["ref_vol_mesh_cm3"]) / out["ref_vol_mesh_cm3"], 3)
    print(f"[{dataset}] {case:18s} dist media={out['surf_mean_mm']:.3f} p95={out['surf_p95_mm']:.2f} "
          f"máx={out['surf_max_mm']:.2f} mm | ΔV={out['vol_diff_pct']:+.2f} % | "
          f"estanca={out['pred_watertight']}/{out['ref_watertight']} | "
          f"malla-vs-máscara {out['pred_mesh_vs_mask_pct']:+.2f} %")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fischer", "dryad"])
    ap.add_argument("--no-stl", action="store_true")
    a = ap.parse_args()
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    if a.only in (None, "fischer"):
        for f in range(5):
            for p in sorted((MODEL / f"fold_{f}" / "validation").glob("*.nii.gz")):
                jobs.append((p.name[:-7], "fischer_cv", p, FISCHER_GT / p.name))
    if a.only in (None, "dryad"):
        for p in sorted((DRYAD / "predictions").glob("dryad_*.nii.gz")):
            jobs.append((p.name[:-7], "dryad_external", p, DRYAD / "labelsTs" / p.name))

    rows = [process(*j, save_stl=not a.no_stl) for j in jobs]
    with open(EVAL_OUT / "reconstruction_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

    summary = dict(postprocessing="componente conexa mayor + relleno 2D (⟂ eje largo) + relleno 3D",
                   mesh=f"marching cubes 0,5; Taubin λ={TAUBIN_LAMBDA}, μ={TAUBIN_MU}, {TAUBIN_ITERS} it.; LPS mm",
                   distance="vértice-a-vértice más cercano, simétrica")
    print("\n============ RESUMEN (media ± DE) ============")
    for ds in ("fischer_cv", "dryad_external"):
        rs = [r for r in rows if r["dataset"] == ds]
        if not rs:
            continue
        s = {}
        for k in ("surf_mean_mm", "surf_median_mm", "surf_p95_mm", "surf_max_mm", "vol_diff_pct",
                  "pred_mesh_vs_mask_pct"):
            x = np.array([r[k] for r in rs], float)
            s[k] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                        min=float(x.min()), max=float(x.max()))
        s["n"] = len(rs)
        s["all_watertight"] = all(r["pred_watertight"] and r["ref_watertight"] for r in rs)
        summary[ds] = s
        print(f"\n{ds} (n={len(rs)})  mallas estancas: {s['all_watertight']}")
        for k in ("surf_mean_mm", "surf_median_mm", "surf_p95_mm", "surf_max_mm", "vol_diff_pct",
                  "pred_mesh_vs_mask_pct"):
            v = s[k]
            print(f"  {k:22s} {v['mean']:8.3f} ± {v['sd']:.3f}   [{v['min']:.3f} – {v['max']:.3f}]")
    with open(EVAL_OUT / "reconstruction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nCSV : {(EVAL_OUT / 'reconstruction_metrics.csv').relative_to(ROOT)}")
    print(f"STL : {MESH_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
