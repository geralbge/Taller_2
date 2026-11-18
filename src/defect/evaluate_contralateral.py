"""
Compara la forma de referencia por INTERPOLACIÓN vs CONTRALATERAL (reflejada y
registrada) con defectos virtuales en Fischer (modo oráculo: hueso remanente =
etiqueta verdadera menos la resección; la contralateral es su etiqueta verdadera).

Sujetos con ambas tibias (21 pares). Cada tibia hace de "afectada" con los 6
defectos virtuales (circ/parc × 30/50/70 mm) y la otra hace de contralateral.

Métricas por caso: Dice de la referencia vs tibia intacta en el tramo resecado,
error de volumen del defecto (%), error de pérdida cortical máxima (pp), error de
longitud (mm), RMS y fracción de inliers del registro.

Uso (desde la raíz del repositorio):
    .\\.venv_nnunet\\Scripts\\python.exe src\\defect\\evaluate_contralateral.py [--subjects 006 z035 ...]

Salidas: results/evaluation/defect/contralateral_oracle.csv y _summary.json
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contralateral as cl        # noqa: E402
import defect_core as dc          # noqa: E402
import virtual_defects as vd      # noqa: E402

ROOT = HERE.parents[1]
LBL = ROOT / "data" / "nnunet" / "nnUNet_raw" / "Dataset502_TibiaFischer" / "labelsTr"
OUT = ROOT / "results" / "evaluation" / "defect"


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def pairs():
    sides = {}
    for p in LBL.glob("fischer_*.nii.gz"):
        sub, side = p.name[len("fischer_"):-7].rsplit("_", 1)
        sides.setdefault(sub, {})[side] = p
    return {s: v for s, v in sorted(sides.items()) if set(v) == {"L", "R"}}


def run_pair(sub, paths):
    rows = []
    data = {}
    for side, p in paths.items():
        lbl, img = dc.read_image(p)
        data[side] = (dc.postprocess_bone(lbl > 0), img)
    for aff, con in (("L", "R"), ("R", "L")):
        A, ai = data[aff]; C, ci = data[con]
        sp = dc.spacing_kji(ai)
        for cfg in vd.configs():
            if cfg["kind"] == "ctrl":
                continue
            d = vd.make_defect(A, sp, cfg)
            k1, k2 = d["clicks"]; a, b = d["k_start"], d["k_end"]
            bone = A & ~d["removal"]
            t0 = time.time()
            reg = cl.register(C, ci, bone, ai, k1, k2, sp[0])
            ref_c = cl.reference_from_contralateral(C, ci, A.shape, ai, k1, k2, reg, cl.bbox_ji(bone, sp))
            t_reg = time.time() - t0
            rt, _, _ = dc.characterize(bone, sp, k1, k2, tibia_extent=d["tibia_k"], reference=A)
            ri, _, ref_i = dc.characterize(bone, sp, k1, k2, tibia_extent=d["tibia_k"])
            rc, _, _ = dc.characterize(bone, sp, k1, k2, tibia_extent=d["tibia_k"], reference=ref_c)
            row = dict(subject=sub, affected=aff, config=cfg["name"], kind=cfg["kind"],
                       reg_rms_mm=round(reg["rms_mm"], 3), reg_inlier_frac=round(reg["inlier_frac"], 3),
                       reg_iterations=reg["iterations"], reg_time_s=round(t_reg, 1),
                       dice_ref_interp=round(dice(ref_i[a:b + 1], A[a:b + 1]), 4),
                       dice_ref_contra=round(dice(ref_c[a:b + 1], A[a:b + 1]), 4),
                       true_volume_mm3=round(rt["volume_mm3"], 1),
                       true_cortical_loss_max_pct=round(rt["cortical_loss_max_pct"], 1))
            for tag, r in (("interp", ri), ("contra", rc)):
                row[f"err_volume_pct_{tag}"] = round(100 * (r["volume_mm3"] - rt["volume_mm3"]) / rt["volume_mm3"], 2)
                row[f"err_length_mm_{tag}"] = round(r.get("length_mm", 0.0) - rt["length_mm"], 2)
                row[f"err_cortical_loss_pp_{tag}"] = round(r["cortical_loss_max_pct"] - rt["cortical_loss_max_pct"], 2)
                row[f"criteria_ok_{tag}"] = bool(r["meets_criteria"] == rt["meets_criteria"])
            rows.append(row)
            print(f"{sub:5s} {aff} {cfg['name']:7s} RMS={reg['rms_mm']:.2f} mm inl={reg['inlier_frac']:.2f}  "
                  f"Dice_ref interp={row['dice_ref_interp']:.3f} contra={row['dice_ref_contra']:.3f}  "
                  f"ΔV interp={row['err_volume_pct_interp']:+.1f} % contra={row['err_volume_pct_contra']:+.1f} %")
    return rows


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(np.nanmean(x)), sd=float(np.nanstd(x, ddof=1)), mean_abs=float(np.nanmean(np.abs(x))),
                min=float(np.nanmin(x)), max=float(np.nanmax(x)), n=int(np.isfinite(x).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()
    P = pairs()
    if a.subjects:
        P = {s: P[s] for s in a.subjects}
    rows = []
    for sub, paths in P.items():
        rows += run_pair(sub, paths)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "contralateral_oracle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for kind in ("circ", "parc", "todos"):
        rs = [r for r in rows if kind == "todos" or r["kind"] == kind]
        summ[kind] = {k: stats([r[k] for r in rs]) for k in
                      ("dice_ref_interp", "dice_ref_contra", "err_volume_pct_interp", "err_volume_pct_contra",
                       "err_cortical_loss_pp_interp", "err_cortical_loss_pp_contra",
                       "err_length_mm_interp", "err_length_mm_contra", "reg_rms_mm", "reg_inlier_frac")}
        for tag in ("interp", "contra"):
            summ[kind][f"criteria_agreement_{tag}"] = float(np.mean([r[f"criteria_ok_{tag}"] for r in rs]))
        d = np.array([r["dice_ref_contra"] - r["dice_ref_interp"] for r in rs])
        summ[kind]["dice_contra_minus_interp"] = stats(d)
        summ[kind]["frac_contra_better"] = float(np.mean(d > 0))
    dc.save_json(summ, OUT / "contralateral_oracle_summary.json")
    print("\n================ RESUMEN interpolación vs contralateral (oráculo) ================")
    for kind, s in summ.items():
        print(f"\n{kind} (n={s['dice_ref_interp']['n']})")
        for k in ("dice_ref_interp", "dice_ref_contra", "err_volume_pct_interp", "err_volume_pct_contra",
                  "err_cortical_loss_pp_interp", "err_cortical_loss_pp_contra", "reg_rms_mm"):
            v = s[k]
            print(f"  {k:30s} {v['mean']:+8.3f} ± {v['sd']:.3f}  |abs| {v['mean_abs']:.3f}  [{v['min']:.3f} – {v['max']:.3f}]")
        print(f"  acuerdo de criterios: interp {100 * s['criteria_agreement_interp']:.1f} %  "
              f"contra {100 * s['criteria_agreement_contra']:.1f} %   contra mejor Dice en "
              f"{100 * s['frac_contra_better']:.0f} % de casos")
    print(f"\nCSV: {(OUT / 'contralateral_oracle.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
