"""Modo opcional: andamios sobre el defecto corregido (regla congelada en virtuales: sector, huésped, masa 3 mm,
metal 3 mm, 600 HU). No reemplaza a los andamios 'sis' automáticos. Salta lo ya hecho."""
import os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; R = ROOT / "results" / "pathological"; OUT = R / "comparacion_materiales_v2"
PY = str(ROOT / ".venv_nnunet" / "Scripts" / "python.exe"); GEN = str(ROOT / "src" / "scaffold" / "generate_scaffold.py")
MAT = {"ti": ("Ti6Al4V_LPBF", "2.5", "0.3"), "pcl": ("PCL_bTCP_extrusion", "4.2", "0.5")}
REF = {"pat_01": R / "pat_01_manual_eval" / "defecto_manual.nii.gz", "pat_02": R / "pat_02_manual_eval" / "final" / "defecto_manual.nii.gz"}
log = open(OUT / "tanda_corregidos.log", "a", encoding="utf-8")
for pat in ("pat_01", "pat_02"):
    ft = R / f"{pat}_ft"; ct = ft / "work" / "roi_0000.nii.gz"
    nail = ["--nail", "auto"] if pat == "pat_01" else []
    for m, (mat, c, w) in MAT.items():
        d = OUT / f"{pat}_siscorr_{m}"
        if list(d.glob("*_scaffold_report.json")):
            continue
        cmd = [PY, GEN, "--defect", str(ft / "defecto.nii.gz"), "--out", str(d), "--cell", c, "--wall", w, "--material", mat,
               "--fab-h", "0.025", "--ct", str(ct), "--gt", str(REF[pat]), "--defect-refine", "sector", "--refine-hu", "600",
               "--refine-reference", str(ft / "referencia.nii.gz"), "--refine-bone", str(ft / "tibia_segmentada.nii.gz"),
               "--refine-mass", "3", "--refine-metal-excl", "3"] + nail
        t0 = time.time()
        print(f"##### {d.name} inicio {time.strftime('%H:%M:%S')}", file=log, flush=True)
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print(f"EXIT {r.returncode} TIEMPO_s {time.time() - t0:.1f}\n", file=log, flush=True)
print("FIN", file=log, flush=True)
