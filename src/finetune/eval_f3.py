"""
Re-evaluación F3 + métricas completas del plan (Pista A), sobre el benchmark nuevo del fold 0
generado por benchmark_v2.py (10 tibias de validación x 13 casos: 6 limpios + 1 control + 6 realistas).

No vuelve a generar ni a predecir: lee lo que ya produjo benchmark_v2.py
  <bench>/meta/*.json, <bench>/gt/*.nii.gz, <bench>/aux/*.nii.gz, <bench>/pred_<modelo>/*.nii.gz
y la etiqueta intacta de Dataset502 (<raw>/Dataset502_TibiaFischer/labelsTr/<caso>.nii.gz).
La TC (<bench>/imagesTs) es opcional: solo se usa para el control de calidad por densidad.

Qué agrega respecto de benchmark_v2.evaluate
--------------------------------------------
Segmentación (por modelo: base, ft)
  - Banda de ±5 mm alrededor de CADA extremo verdadero del defecto (proximal y distal, por
    separado): Dice, ASSD y HD95 (mm) entre segmentación y verdad dentro de la banda.
  - Error de posición de cada extremo (mm): extremo estimado - extremo verdadero en el eje largo
    (signo + = el extremo estimado queda hacia distal).
  - Falsos positivos dentro del defecto: vóxeles marcados como hueso dentro del defecto verdadero
    (mm3 y % del volumen del defecto). Incluye fragmentos, secuestros, metal, espaciador y ruido.
  - Globales: Dice, ASSD y HD95 de toda la tibia.
F3 – forma geométrica del defecto (método vigente: interpolación entre extremos sanos)
  - Defecto estimado (tramo principal) vs defecto verdadero (referencia = tibia intacta):
    Dice, ASSD, HD95 y error de volumen (%).
  - Tres condiciones: segmentación del modelo base, del ajustado y "gtseg" (interpolación aplicada
    a la verdad). gtseg aísla el error propio del método F3 del error de segmentación.

Convención de máscaras: la segmentación y la verdad se comparan tal cual (convención Fischer, canal
medular sin etiquetar), igual que en benchmark_v2. El defecto se calcula con dc.characterize, que
rellena el canal (superficie externa).

Uso (mismo entorno que benchmark_v2.py; colocar este archivo junto a benchmark_v2.py/defect_core.py):
  python eval_f3.py --bench <ruta>/benchmark_v2_f0 --raw %nnUNet_raw%
Salidas en --bench: f3_rows.csv, f3_summary.json, f3_resumen.md
Dependencias: numpy, scipy, SimpleITK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defect_core as dc  # noqa: E402

SRC_NAME = "Dataset502_TibiaFischer"
END_BAND_MM = 5.0       # banda a cada lado de cada extremo (plan F2: ±5 mm)
SD_MARGIN_MM = 10.0     # margen extra para que la distancia superficial no se corte en el borde de la banda
AUX_NONHOST = {"fragment": 1, "sequestrum": 2, "metal": 3, "spacer": 4}  # solo informativo; se lee de v2 si está


# ------------------------------------------------------------------ utilidades
def read(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def dice(x, y):
    s = int(x.sum()) + int(y.sum())
    return float(2 * (x & y).sum() / s) if s else float("nan")


def surface(m):
    return m & ~ndi.binary_erosion(m)


def surf_dist(a, b, sp, k_keep=None):
    """
    ASSD y HD95 (mm) entre las superficies de a y b (bool, misma grilla, spacing sp kji).
    Si k_keep=(k0, k1) solo se cuentan los puntos de superficie con k0 <= k <= k1, pero la
    distancia se mide a la superficie completa del otro objeto. Trabaja en un recorte.
    """
    if not a.any() or not b.any():
        return float("nan"), float("nan")
    u = a | b
    idx = np.nonzero(u)
    mk = [int(np.ceil(SD_MARGIN_MM / s)) + 1 for s in sp]
    lo = [max(int(i.min()) - m, 0) for i, m in zip(idx, mk)]
    hi = [min(int(i.max()) + m + 1, n) for i, m, n in zip(idx, mk, a.shape)]
    if k_keep is not None:
        lo[0] = max(lo[0], k_keep[0] - mk[0]); hi[0] = min(hi[0], k_keep[1] + mk[0] + 1)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    A, B = a[sl], b[sl]
    if not A.any() or not B.any():
        return float("nan"), float("nan")
    sa, sb = surface(A), surface(B)
    if k_keep is not None:
        keep = np.zeros(A.shape[0], bool)
        keep[max(k_keep[0] - lo[0], 0):max(k_keep[1] - lo[0] + 1, 0)] = True
        ka, kb = sa & keep[:, None, None], sb & keep[:, None, None]
    else:
        ka, kb = sa, sb
    if not ka.any() and not kb.any():
        return float("nan"), float("nan")
    da = ndi.distance_transform_edt(~sb, sampling=sp)[ka]   # de a hacia b
    db = ndi.distance_transform_edt(~sa, sampling=sp)[kb]   # de b hacia a
    d = np.concatenate([da, db])
    return float(d.mean()), float(np.percentile(d, 95))


def band(k, sp, n):
    h = int(round(END_BAND_MM / sp[0]))
    return max(k - h, 0), min(k + h, n - 1)


def main_segment(defect, res):
    out = np.zeros_like(defect)
    if res.get("defect_detected"):
        a, b = res["k_start"], res["k_end"]
        out[a:b + 1] = defect[a:b + 1]
    return out


def r(x, nd=3):
    return round(float(x), nd) if x is not None and x == x else ""


# ------------------------------------------------------------------ evaluación
def evaluate(a):
    bench, raw = Path(a.bench), Path(a.raw) / SRC_NAME
    rows, cache = [], {}
    metas = sorted((bench / "meta").glob("*.json"))
    if a.limit:
        metas = metas[:a.limit]
    t0 = time.time()
    for n_case, mp in enumerate(metas, 1):
        meta = json.load(open(mp, encoding="utf-8")); cid = mp.stem; case = meta["case"]
        sp = np.array(meta["spacing_kji"], float); vox = float(np.prod(sp))
        if case not in cache:
            cache = {case: dc.postprocess_bone(read(raw / "labelsTr" / f"{case}.nii.gz") > 0)}
        intact = cache[case]
        gt = read(bench / "gt" / f"{cid}.nii.gz") > 0
        aux = read(bench / "aux" / f"{cid}.nii.gz")
        ctp = bench / "imagesTs" / f"{cid}_0000.nii.gz"
        ct = read(ctp) if (a.use_ct and ctp.is_file()) else None
        k1, k2 = meta["clicks_k"]; tk = tuple(meta["tibia_k"]); nk = gt.shape[0]
        cfg = meta["config"]; is_ctrl = cfg["scenario"] == "ctrl"

        if is_ctrl:
            g = dict(defect_detected=False); d_gt = np.zeros_like(gt)
        else:
            g, d_full, _ = dc.characterize(gt, sp, k1, k2, tibia_extent=tk, reference=intact)
            d_gt = main_segment(d_full, g)
        vol_gt = float(d_gt.sum() * vox)
        bands = {}
        if g.get("defect_detected"):
            bands = dict(prox=band(g["k_start"], sp, nk), dist=band(g["k_end"], sp, nk))

        conds = [("gtseg", gt)] if not is_ctrl else []
        for m in a.models:
            pp = bench / f"pred_{m}" / f"{cid}.nii.gz"
            if pp.is_file():
                conds.append((m, read(pp) == 1))
        for name, seg in conds:
            e, d_full_e, _ = dc.characterize(seg, sp, k1, k2, tibia_extent=tk, ct=ct if name != "gtseg" else None)
            d_e = main_segment(d_full_e, e)
            row = dict(case=case, cid=cid, cond=name, set=cfg["set"], scenario=cfg["scenario"], config=cfg["name"],
                       implant=meta.get("implant", "none"), gt_detected=g.get("defect_detected", False),
                       est_detected=e["defect_detected"],
                       detection_ok=bool(g.get("defect_detected", False) == e["defect_detected"]),
                       criteria_ok=bool(g.get("meets_criteria", False) == e["meets_criteria"]),
                       gt_length_mm=r(g.get("length_mm", 0.0), 1), est_length_mm=r(e.get("length_mm", 0.0), 1))
            # ---- F3: forma del defecto
            if g.get("defect_detected"):
                row["f3_dice_defect"] = r(dice(d_e, d_gt), 4)
                assd, hd = surf_dist(d_e, d_gt, sp)
                row["f3_assd_defect_mm"], row["f3_hd95_defect_mm"] = r(assd), r(hd)
                row["f3_err_volume_pct"] = r(100 * (d_e.sum() * vox - vol_gt) / max(vol_gt, 1), 1)
                row["err_length_mm"] = r(e.get("length_mm", 0.0) - g["length_mm"], 1)
                if e["defect_detected"]:
                    row["err_end_prox_mm"] = r((e["k_start"] - g["k_start"]) * sp[0], 2)
                    row["err_end_dist_mm"] = r((e["k_end"] - g["k_end"]) * sp[0], 2)
            if name == "gtseg":
                rows.append(row); continue
            # ---- segmentación: bandas de ±5 mm en cada extremo
            for side, (b0, b1) in bands.items():
                w = np.zeros(nk, bool); w[b0:b1 + 1] = True
                row[f"seg_dice_{side}"] = r(dice(seg[w], gt[w]), 4)
                assd, hd = surf_dist(seg, gt, sp, k_keep=(b0, b1))
                row[f"seg_assd_{side}_mm"], row[f"seg_hd95_{side}_mm"] = r(assd), r(hd)
            # ---- falsos positivos dentro del defecto verdadero
            if g.get("defect_detected"):
                fp = seg & d_gt & ~gt
                row["fp_in_defect_mm3"] = r(fp.sum() * vox, 0)
                row["fp_in_defect_pct"] = r(100 * fp.sum() / max(d_gt.sum(), 1), 2)
            row["fp_nonhost_mm3"] = r(sum(float((seg & (aux == v)).sum()) for v in AUX_NONHOST.values()) * vox, 0)
            # ---- globales
            row["glob_dice"] = r(dice(seg, gt), 4)
            assd, hd = surf_dist(seg, gt, sp)
            row["glob_assd_mm"], row["glob_hd95_mm"] = r(assd), r(hd)
            rows.append(row)
        el = time.time() - t0
        print(f"[F3] {n_case}/{len(metas)} {cid}  ({el / n_case:.1f} s/caso)", flush=True)
    return rows


# ------------------------------------------------------------------ resumen
METRICS = ["f3_dice_defect", "f3_assd_defect_mm", "f3_hd95_defect_mm", "f3_err_volume_pct", "err_length_mm",
           "err_end_prox_mm", "err_end_dist_mm", "seg_dice_prox", "seg_dice_dist", "seg_assd_prox_mm",
           "seg_assd_dist_mm", "seg_hd95_prox_mm", "seg_hd95_dist_mm", "fp_in_defect_mm3", "fp_in_defect_pct",
           "fp_nonhost_mm3", "glob_dice", "glob_assd_mm", "glob_hd95_mm"]
SIGNED = {"f3_err_volume_pct", "err_length_mm", "err_end_prox_mm", "err_end_dist_mm"}  # se resume |x|
HIGHER_BETTER = {"f3_dice_defect", "seg_dice_prox", "seg_dice_dist", "glob_dice"}
GROUPS = ("limpio", "realista", "A", "B", "C", "D", "ctrl")


def num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else None


def vals(rows, cond, grp, k):
    out = []
    for x in rows:
        if x["cond"] == cond and (x["set"] == grp or x["scenario"] == grp):
            v = num(x.get(k, ""))
            if v is not None:
                out.append(abs(v) if k in SIGNED else v)
    return out


def summarize(rows, models):
    conds = ["gtseg"] + list(models)
    summ = {}
    for grp in GROUPS:
        for c in conds:
            rr = [x for x in rows if x["cond"] == c and (x["set"] == grp or x["scenario"] == grp)]
            if not rr:
                continue
            s = dict(n=len(rr), detection_ok=sum(x["detection_ok"] for x in rr), criteria_ok=sum(x["criteria_ok"] for x in rr))
            for k in METRICS:
                v = vals(rows, c, grp, k)
                if v:
                    q1, md, q3 = np.percentile(v, [25, 50, 75])
                    s[k] = dict(median=round(md, 4), q1=round(q1, 4), q3=round(q3, 4), p90=round(float(np.percentile(v, 90)), 4))
            summ[f"{c}/{grp}"] = s
    # base vs ft pareado. Unidad independiente = tibia (n=10): mediana por tibia y Wilcoxon de rangos con signo.
    tests = {}
    if "base" in models and "ft" in models:
        for grp in ("limpio", "realista"):
            for k in METRICS:
                pb, pf = {}, {}
                for x in rows:
                    if not (x["set"] == grp or x["scenario"] == grp):
                        continue
                    v = num(x.get(k, ""))
                    if v is None:
                        continue
                    v = abs(v) if k in SIGNED else v
                    (pb if x["cond"] == "base" else pf if x["cond"] == "ft" else {}).setdefault(x["case"], []).append(v)
                tib = sorted(set(pb) & set(pf))
                if len(tib) < 5:
                    continue
                xb = np.array([np.median(pb[t]) for t in tib]); xf = np.array([np.median(pf[t]) for t in tib])
                try:
                    p = float(stats.wilcoxon(xb, xf).pvalue) if np.any(xb != xf) else 1.0
                except ValueError:
                    p = float("nan")
                better = (xf > xb) if k in HIGHER_BETTER else (xf < xb)
                tests[f"{grp}/{k}"] = dict(n_tibias=len(tib), base_median=round(float(np.median(xb)), 4),
                                           ft_median=round(float(np.median(xf)), 4),
                                           tibias_ft_mejor=int(better.sum()), wilcoxon_p=round(p, 4))
    return summ, tests


def write_md(summ, tests, models, path):
    keys = [("f3_dice_defect", "F3 Dice del defecto"), ("f3_assd_defect_mm", "F3 ASSD del defecto (mm)"),
            ("f3_err_volume_pct", "F3 error abs. de volumen (%)"), ("err_end_prox_mm", "Error abs. extremo proximal (mm)"),
            ("err_end_dist_mm", "Error abs. extremo distal (mm)"), ("seg_dice_prox", "Dice banda ±5 mm proximal"),
            ("seg_dice_dist", "Dice banda ±5 mm distal"), ("seg_assd_prox_mm", "ASSD banda proximal (mm)"),
            ("seg_assd_dist_mm", "ASSD banda distal (mm)"), ("fp_in_defect_pct", "FP dentro del defecto (% vol.)"),
            ("glob_dice", "Dice global"), ("glob_hd95_mm", "HD95 global (mm)")]
    conds = ["gtseg"] + list(models)
    L = ["# Re-evaluación F3 y benchmark fold 0 — resumen automático", "",
         "Mediana [Q1–Q3]. gtseg = interpolación F3 aplicada a la segmentación verdadera (error propio del método).", ""]
    for grp in ("limpio", "realista", "A", "B", "C"):
        L += [f"## {grp}", "", "| Métrica | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for k, lab in keys:
            cells = []
            for c in conds:
                s = summ.get(f"{c}/{grp}", {}).get(k)
                cells.append(f"{s['median']:.3g} [{s['q1']:.3g}–{s['q3']:.3g}]" if s else "—")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")
    if tests:
        L += ["## Base vs ajustado (pareado por tibia, n = 10; Wilcoxon)", "",
              "| Conjunto/métrica | Base | Ajustado | Tibias en que mejora el ajustado | p |", "|---|---|---|---|---|"]
        for k, t in tests.items():
            L.append(f"| {k} | {t['base_median']:.4g} | {t['ft_median']:.4g} | {t['tibias_ft_mejor']}/{t['n_tibias']} | {t['wilcoxon_p']} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="carpeta benchmark_v2_f0 (meta/, gt/, aux/, pred_*/)")
    ap.add_argument("--raw", default=os.environ.get("nnUNet_raw"), help="nnUNet_raw (con Dataset502_TibiaFischer)")
    ap.add_argument("--models", nargs="*", default=["base", "ft"])
    ap.add_argument("--use-ct", action="store_true", help="usar imagesTs para el control por densidad (opcional)")
    ap.add_argument("--limit", type=int, default=0, help="solo los primeros N casos (prueba)")
    ap.add_argument("--f3v2", action="store_true",
                    help="F3 v2: banda sana con límite superior de área + defecto = componente 3D principal (verdad y estimado)")
    ap.add_argument("--conn", action="store_true",
                    help="solo la regla de conectividad 3D (verdad y estimado), banda sana original")
    a = ap.parse_args()
    bench = Path(a.bench)
    pre = "f3v2" if a.f3v2 else ("f3conn" if a.conn else "f3")
    if a.f3v2:
        dc.set_f3_v2(True)
    elif a.conn:
        dc.set_connected_only(True)
    rows = evaluate(a)
    keys = []
    for x in rows:
        for k in x:
            if k not in keys:
                keys.append(k)
    with open(bench / f"{pre}_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(rows)
    summ, tests = summarize(rows, a.models)
    json.dump(dict(end_band_mm=END_BAND_MM, summary=summ, base_vs_ft=tests),
              open(bench / f"{pre}_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_md(summ, tests, a.models, bench / f"{pre}_resumen.md")
    print(f"Listo: {bench / (pre + '_rows.csv')}, {pre}_summary.json, {pre}_resumen.md")


if __name__ == "__main__":
    main()
