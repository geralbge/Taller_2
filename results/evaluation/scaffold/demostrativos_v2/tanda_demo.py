"""4 casos demostrativos virtuales, uno por tipo: el de Dice del defecto (modelo ajustado) más cercano a la
mediana del tipo en f3conn_rows.csv. Regla fijada antes de generar andamios. Salta lo ya hecho."""
import csv, json, os, statistics as st, subprocess, sys, time
from pathlib import Path
import numpy as np, SimpleITK as sitk
ROOT = Path(__file__).resolve().parents[4]  # raíz del repositorio
sys.path.insert(0, str(ROOT / "src" / "defect"))
import defect_core as dc
OUT = Path(__file__).parent
B1, B2 = ROOT / "src/finetune/benchmark_v2_f0", ROOT / "resultados_ft/benchmark_v2_f0"
RAW = ROOT / "data/nnunet/nnUNet_raw/Dataset502_TibiaFischer"
PY, GEN = str(ROOT / ".venv_nnunet/Scripts/python.exe"), str(ROOT / "src/scaffold/generate_scaffold.py")
MAT = {"ti": ("Ti6Al4V_LPBF", "2.5", "0.3"), "pcl": ("PCL_bTCP_extrusion", "4.2", "0.5")}
rows = [r for r in csv.DictReader(open(B2 / "f3conn_rows.csv", encoding="utf-8")) if r["cond"] == "ft" and r["f3_dice_defect"]]
sel = {}
for s in "ABCD":
    x = [r for r in rows if r["scenario"] == s]
    m = st.median(float(r["f3_dice_defect"]) for r in x)
    c = min(x, key=lambda r: (abs(float(r["f3_dice_defect"]) - m), r["cid"]))
    sel[s] = dict(cid=c["cid"], dice_ft=float(c["f3_dice_defect"]), mediana_tipo=m, n_tipo=len(x), implante=c["implant"], L_mm=float(c["gt_length_mm"]))
json.dump(sel, open(OUT / "seleccion.json", "w", encoding="utf-8"), indent=1)
log = open(OUT / "tanda.log", "a", encoding="utf-8")
rd = lambda p: sitk.ReadImage(str(p))
arr = sitk.GetArrayFromImage
for s, c in sel.items():
    cid = c["cid"]; d = OUT / f"tipo_{s}_{cid}"; d.mkdir(exist_ok=True)
    est, gtp = d / "defecto_estimado.nii.gz", d / "defecto_verdadero.nii.gz"
    if not (est.is_file() and gtp.is_file()):
        meta = json.load(open(B2 / "meta" / f"{cid}.json", encoding="utf-8"))
        sp = np.array(meta["spacing_kji"], float); k1, k2 = meta["clicks_k"]; tk = tuple(meta["tibia_k"])
        img = rd(B2 / "pred_ft" / f"{cid}.nii.gz")
        pred = arr(img) == 1
        gt = arr(rd(B2 / "gt" / f"{cid}.nii.gz")) > 0
        intact = dc.postprocess_bone(arr(rd(RAW / "labelsTr" / f"{meta['case']}.nii.gz")) > 0)
        g, dg, _ = dc.characterize(gt, sp, k1, k2, tibia_extent=tk, reference=intact)
        d_gt = np.zeros_like(dg); d_gt[g["k_start"]:g["k_end"] + 1] = dg[g["k_start"]:g["k_end"] + 1]
        e, de, _ = dc.characterize(pred, sp, k1, k2, tibia_extent=tk)
        d_e = np.zeros_like(de); d_e[e["k_start"]:e["k_end"] + 1] = de[e["k_start"]:e["k_end"] + 1]
        print(cid, "Dice", 2 * (d_e & d_gt).sum() / (d_e.sum() + d_gt.sum()), "esperado", c["dice_ft"], file=log, flush=True)
        dc.write_like(d_e, img, est); dc.write_like(d_gt, img, gtp)
    for m, (mat, cell, w) in MAT.items():
        o = d / m
        if list(o.glob("*_scaffold_report.json")):
            continue
        cmd = [PY, GEN, "--defect", str(est), "--out", str(o), "--cell", cell, "--wall", w, "--material", mat, "--fab-h", "0.025", "--gt", str(gtp)]
        t0 = time.time()
        print(f"##### {o} inicio {time.strftime('%H:%M:%S')}", file=log, flush=True)
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print(f"EXIT {r.returncode} TIEMPO_s {time.time() - t0:.1f}\n", file=log, flush=True)
print("FIN", file=log, flush=True)
