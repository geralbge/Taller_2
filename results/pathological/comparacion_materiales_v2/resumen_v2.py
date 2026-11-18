"""resumen_v2.csv: 12 andamios (chequeo fino) y métricas ref vs sis. 'siscorr' = modo opcional con la regla
congelada; no reemplaza a 'sis' (automático)."""
import csv, glob, json, os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

R = str(Path(__file__).resolve().parents[1])  # results/pathological
C = rf"{R}\comparacion_materiales_v2"
CASOS = {"pat_01": (rf"{R}\pat_01_manual_eval\defecto_manual.nii.gz", rf"{R}\pat_01_ft\defecto.nii.gz"),
         "pat_02": (rf"{R}\pat_02_manual_eval\final\defecto_manual.nii.gz", rf"{R}\pat_02_ft\defecto.nii.gz")}


def surf(m, aff):
    b = m & ~ndi.binary_erosion(m, structure=np.ones((3, 3, 3)))
    return np.argwhere(b) @ aff[:3, :3].T + aff[:3, 3]


def defect_metrics(pr, ps):
    ir, is_ = nib.load(pr), nib.load(ps)
    aff = ir.affine
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    vox = abs(np.linalg.det(aff[:3, :3]))
    a, b = np.asarray(ir.dataobj) > 0, np.asarray(is_.dataobj) > 0
    ax = int(np.argmax(np.array(a.shape) * sp))
    ext = lambda m: float(np.ptp(np.nonzero(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]) + 1) * sp[ax]
    pa, pb = surf(a, aff), surf(b, aff)
    d = np.concatenate([cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0]])
    return dict(dice=round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 4), V_ref_mm3=round(float(a.sum() * vox), 1),
                V_sis_mm3=round(float(b.sum() * vox), 1), dif_vol_pct=round(100 * (float(b.sum()) - float(a.sum())) / float(a.sum()), 1),
                L_ref_mm=round(ext(a), 1), L_sis_mm=round(ext(b), 1), dist_media_mm=round(float(d.mean()), 3),
                dist_p95_mm=round(float(np.percentile(d, 95)), 3), dist_max_mm=round(float(d.max()), 3))


rows = []
for pat in CASOS:
    dm = defect_metrics(*CASOS[pat])
    for var in ("ref", "sis", "siscorr"):
        for mat in ("ti", "pcl"):
            d = rf"{C}\{pat}_{var}_{mat}"
            js = glob.glob(os.path.join(d, "*_scaffold_report.json"))
            if not js:
                rows.append(dict(paciente=pat, defecto=var, material=mat, estado="SIN REPORTE"))
                continue
            j = json.load(open(js[0], encoding="utf-8"))
            m, p, f4 = j["metricas"], j["parametros"], j["f4"]
            fb = f4["fabricabilidad"]; g = fb["medicion_grilla"]; vs = m.get("vs_defecto_verdadero", {})
            vsd = vs.get("distancia_superficial_mm", {})
            rows.append(dict(
                paciente=pat, defecto=var, material=f4["material"]["clave"], correccion_defecto="modo opcional" if var == "siscorr" else "no aplicada",
                celda_mm=p["celda_mm"], pared_objetivo_mm=p["pared_objetivo_mm"], clavo="canal_clavo" in f4,
                porosidad_pct=round(m["porosidad_pct"], 2), interconectividad_pct=m["interconectividad_pct"],
                V_solido_mm3=round(m["V_solido_mm3"], 1), V_envolvente_mm3=round(m["V_envolvente_mm3"], 1),
                componentes=fb["componentes_solidas"], solido_quitado_pct=round(100 * m["fragmentos"].get("frac_quitada_mayor", 0), 2),
                pared_fina_p1_mm=round(fb["pared_real"]["p1"], 3), pared_fina_p5_mm=round(fb["pared_real"]["p5"], 3), pared_fina_mediana_mm=round(fb["pared_real"]["mediana"], 3),
                poro_fino_p5_mm=round(fb["poro_real"]["p5"], 3), poro_fino_mediana_mm=round(fb["poro_real"]["mediana"], 3),
                frac_pared_bajo_min_fina=round(fb["frac_vol_pared_bajo_minimo"], 5),
                pared_grilla_mediana_mm=round(g["pared_real"]["mediana"], 3), frac_pared_bajo_min_grilla=round(g["frac_vol_pared_bajo_minimo"], 4),
                chequeos=";".join(f"{k}={v}" for k, v in fb["chequeos"].items()), fabricable=fb["fabricable"],
                vsref_coincidencia_geom=round(vs["coincidencia_geometrica"], 3) if vs else "",
                vsref_ocupacion_pct=round(vs["ocupacion_pct"], 1) if vs else "",
                vsref_dist_media_mm=round(vsd.get("mean_mm", float("nan")), 3) if vs else "",
                vsref_dist_p95_mm=round(vsd.get("p95_mm", float("nan")), 3) if vs else "",
                **{f"defecto_{k}": v for k, v in (defect_metrics(CASOS[pat][0], rf"{d}\defecto_corregido.nii.gz") if var == "siscorr" else dm).items()},
                tiempo_s=j["tiempo_s"]))
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(rf"{C}\resumen_v2.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, delimiter=";", restval="")
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
