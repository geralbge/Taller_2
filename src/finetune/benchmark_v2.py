"""
Benchmark nuevo (F1/F3): tibias de VALIDACIÓN del fold 0, que ningún modelo vio al entrenar
(ni el original de Dataset502 ni el ajustado de Dataset503, porque el split se conserva).

Dos conjuntos, con semillas fijas, idénticos para ambos modelos:
  limpio    : 6 configuraciones fijas = {circunferencial 360°, parcial 240°} x {30, 50, 70 mm},
              centro en el 45 % / 55 % de la tibia, generadas con el escenario D del generador v2
              (resección limpia, sin implantes) + 1 control sin defecto
  realista  : escenarios A, B y C, 2 casos cada uno, longitud 25-70 mm (rango de evaluación),
              alineados (sin desalineación: la referencia intacta sigue valiendo)
La verdad es la etiqueta "host" del generador; la referencia para caracterizar es la tibia intacta.

Pasos (se pueden correr por separado):
  python benchmark_v2.py generate --fold 0
  python benchmark_v2.py predict  --fold 0 --models base ft
  python benchmark_v2.py evaluate --fold 0 --models base ft
Salidas en --out (por defecto ./benchmark_v2_f0):
  imagesTs/, gt/, aux/, meta/, pred_<modelo>/, results_rows.csv, results_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import virtual_defects_v2 as v2   # noqa: E402
import defect_core as dc           # noqa: E402

SRC_NAME = "Dataset502_TibiaFischer"
PLANS = "nnUNetResEncUNetMPlans"
SEED = 20261105
BAND_MM = 10.0          # banda a cada lado del defecto para las métricas locales
AUX = v2.AUX
NONHOST = ("fragment", "sequestrum", "metal", "spacer")
NEWBONE = ("callus", "involucrum", "cap")


def clean_configs():
    out, i = [], 0
    for L in (30, 50, 70):
        for kind, sec in (("circ", 360.0), ("parc", 240.0)):
            out.append(dict(name=f"{kind}{L}", set="limpio", scenario="D", length_mm=float(L), sector_deg=sec,
                            center_frac=0.45 if i % 2 == 0 else 0.55))
            i += 1
    out.append(dict(name="ctrl", set="limpio", scenario="ctrl"))
    return out


def real_configs():
    return [dict(name=f"real{s}{j}", set="realista", scenario=s, rep=j) for s in "ABC" for j in range(2)]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    return o.tolist() if isinstance(o, np.ndarray) else o


def val_cases(pre, fold):
    return json.load(open(Path(pre) / SRC_NAME / "splits_final.json", encoding="utf-8"))[fold]["val"]


def generate(a):
    raw, out = Path(a.raw) / SRC_NAME, Path(a.out)
    for d in ("imagesTs", "gt", "aux", "meta"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for ci, case in enumerate(val_cases(a.preprocessed, a.fold)):
        ct_img = sitk.ReadImage(str(raw / "imagesTr" / f"{case}_0000.nii.gz"))
        lb_img = sitk.ReadImage(str(raw / "labelsTr" / f"{case}.nii.gz"))
        ct = sitk.GetArrayFromImage(ct_img); lbl = sitk.GetArrayFromImage(lb_img)
        sp = np.array(lb_img.GetSpacing()[::-1], float)
        intact = v2.fill_tibia(lbl)
        ks = np.nonzero(intact.any((1, 2)))[0]; t0, t1 = int(ks.min()), int(ks.max())
        for cj, cfg in enumerate(clean_configs() + real_configs()):
            cid = f"{case}__{cfg['name']}"
            if (out / "meta" / f"{cid}.json").is_file():
                continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, a.fold, ci, cj]))
            if cfg["scenario"] == "ctrl":
                kc = int(t0 + 0.5 * (t1 - t0)); h = int(round(25.0 / sp[0]))
                ct_o, lab_o, aux_o = ct, (lbl > 0).astype(np.uint8), np.zeros(lbl.shape, np.uint8)
                meta = dict(scenario="ctrl", clicks_k=(kc - h, kc + h), tibia_k=(t0, t1))
            else:
                p = v2.sample_params(cfg["scenario"], rng, 0)
                if cfg["set"] == "limpio":
                    p.update(length_mm=cfg["length_mm"], center_frac=cfg["center_frac"], sector_deg=cfg["sector_deg"],
                             sector_dir_deg=0.0, sector_edge_jitter_deg=0.0)
                else:
                    p.update(length_mm=float(rng.uniform(25, 70)), malalign=False)
                r = v2.make_defect_v2(ct, lbl, sp, p, rng, label_policy="host")
                ct_o, lab_o, aux_o, meta = r["ct"], r["label"], r["aux"], _jsonable(r["meta"])
                meta["clicks_k"] = list(r["clicks"])
            meta.update(case=case, config=cfg, spacing_kji=sp.tolist())
            for arr, path, dt in ((ct_o, out / "imagesTs" / f"{cid}_0000.nii.gz", None),
                                  (lab_o, out / "gt" / f"{cid}.nii.gz", np.uint8),
                                  (aux_o, out / "aux" / f"{cid}.nii.gz", np.uint8)):
                o = sitk.GetImageFromArray(arr if dt is None else arr.astype(dt)); o.CopyInformation(ct_img)
                sitk.WriteImage(o, str(path), useCompression=True)
            json.dump(_jsonable(meta), open(out / "meta" / f"{cid}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            print(f"[GEN] {cid}  {cfg['scenario']}  {meta.get('length_mm', 0):.0f} mm")


def model_dir(results, name, fold):
    if name == "base":
        return Path(results) / SRC_NAME / f"nnUNetTrainer__{PLANS}__3d_fullres"
    return Path(results) / f"Dataset503_TibiaDefectsF{fold}" / f"nnUNetTrainer_FT__{PLANS}__3d_fullres"


def predict(a):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    out = Path(a.out)
    cids = sorted(p.name[:-12] for p in (out / "imagesTs").glob("*_0000.nii.gz"))
    cuda = torch.cuda.is_available()
    for m in a.models:
        po = out / f"pred_{m}"; po.mkdir(exist_ok=True)
        todo = [c for c in cids if not (po / f"{c}.nii.gz").is_file()]
        if not todo:
            continue
        pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=cuda,
                             perform_everything_on_device=cuda, device=torch.device("cuda" if cuda else "cpu"),
                             verbose=False, allow_tqdm=True)
        pr.initialize_from_trained_model_folder(str(model_dir(a.results, m, a.fold)), use_folds=(a.fold,),
                                                checkpoint_name="checkpoint_final.pth")
        pr.predict_from_files([[str(out / "imagesTs" / f"{c}_0000.nii.gz")] for c in todo],
                              [str(po / f"{c}.nii.gz") for c in todo], save_probabilities=False, overwrite=True,
                              num_processes_preprocessing=2, num_processes_segmentation_export=2)
        print(f"[PRED] {m}: {len(todo)} casos")


def _dice(x, y):
    s = x.sum() + y.sum()
    return float(2 * (x & y).sum() / s) if s else float("nan")


def evaluate(a):
    out, raw = Path(a.out), Path(a.raw) / SRC_NAME
    rows, intact_cache = [], {}
    for mp in sorted((out / "meta").glob("*.json")):
        meta = json.load(open(mp, encoding="utf-8")); cid = mp.stem; case = meta["case"]
        sp = np.array(meta["spacing_kji"]); vox = float(np.prod(sp))
        if case not in intact_cache:
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(raw / "labelsTr" / f"{case}.nii.gz")))
            intact_cache = {case: (lbl > 0, dc.postprocess_bone(lbl > 0))}
        lbl0, intact = intact_cache[case]
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "gt" / f"{cid}.nii.gz"))) > 0
        aux = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "aux" / f"{cid}.nii.gz")))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "imagesTs" / f"{cid}_0000.nii.gz")))
        k1, k2 = meta["clicks_k"]; tk = tuple(meta["tibia_k"])
        is_ctrl = meta["config"]["scenario"] == "ctrl"
        if is_ctrl:
            g = dict(defect_detected=False, length_mm=0.0, volume_mm3=0.0, cortical_loss_max_pct=0.0, meets_criteria=False)
        else:
            g, _, _ = dc.characterize(gt, sp, k1, k2, tibia_extent=tk, reference=intact)
        b = int(round(BAND_MM / sp[0]))
        win = np.zeros(gt.shape[0], bool); win[max(k1 - b, 0):k2 + b + 1] = True
        for m in a.models:
            pp = out / f"pred_{m}" / f"{cid}.nii.gz"
            if not pp.is_file():
                continue
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))) == 1
            e, _, _ = dc.characterize(pred, sp, k1, k2, tibia_extent=tk, ct=ct)
            row = dict(case=case, cid=cid, model=m, set=meta["config"]["set"], scenario=meta["config"]["scenario"],
                       config=meta["config"]["name"], implant=meta.get("implant", "none"),
                       ulbdc=(meta.get("solomin") or {}).get("code", ""),
                       gt_detected=g["defect_detected"], est_detected=e["defect_detected"],
                       detection_ok=g["defect_detected"] == e["defect_detected"],
                       gt_length_mm=round(g.get("length_mm", 0.0), 1), est_length_mm=round(e.get("length_mm", 0.0), 1),
                       gt_volume_mm3=round(g.get("volume_mm3", 0.0), 0), est_volume_mm3=round(e.get("volume_mm3", 0.0), 0),
                       gt_cl_max_pct=round(g.get("cortical_loss_max_pct", 0.0), 1),
                       est_cl_max_pct=round(e.get("cortical_loss_max_pct", 0.0), 1),
                       criteria_ok=g["meets_criteria"] == e["meets_criteria"],
                       qc_density_warning=e.get("qc_density_warning", False),
                       dice_band=round(_dice(pred[win], gt[win]), 4),
                       dice_outside=round(_dice(pred[~win], gt[~win]), 4))
            if g["defect_detected"]:
                row["err_length_mm"] = round(row["est_length_mm"] - row["gt_length_mm"], 1)
                row["err_volume_pct"] = round(100 * (row["est_volume_mm3"] - row["gt_volume_mm3"]) / max(row["gt_volume_mm3"], 1), 1)
                row["err_cl_pp"] = round(row["est_cl_max_pct"] - row["gt_cl_max_pct"], 1)
            for k in NONHOST:      # material que NO es hueso huésped y el modelo marcó como hueso
                row[f"fp_{k}_mm3"] = round(float((pred & (aux == AUX[k])).sum()) * vox, 0)
            nb = np.isin(aux, [AUX[k] for k in NEWBONE]) & gt
            row["newbone_recall"] = round(float((pred & nb).sum() / nb.sum()), 3) if nb.any() else ""
            rows.append(row)
            print(f"[EVAL] {m:4s} {cid}  dice_band={row['dice_band']:.3f}  det={row['detection_ok']}  "
                  f"errL={row.get('err_length_mm', '')}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
    with open(out / "results_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    summ = {}
    for m in a.models:
        for grp in ("limpio", "realista", "A", "B", "C", "D"):
            rr = [r for r in rows if r["model"] == m and (r["set"] == grp or r["scenario"] == grp)]
            if not rr:
                continue
            def med(k):
                x = [abs(r[k]) for r in rr if isinstance(r.get(k), (int, float)) and r.get(k) == r.get(k)]
                return round(float(np.median(x)), 2) if x else None
            summ[f"{m}/{grp}"] = dict(n=len(rr), detection_ok=sum(r["detection_ok"] for r in rr),
                                     criteria_ok=sum(r["criteria_ok"] for r in rr),
                                     dice_band_median=round(float(np.median([r["dice_band"] for r in rr])), 4),
                                     dice_outside_median=round(float(np.median([r["dice_outside"] for r in rr])), 4),
                                     abs_err_length_mm_median=med("err_length_mm"),
                                     abs_err_volume_pct_median=med("err_volume_pct"),
                                     abs_err_cl_pp_median=med("err_cl_pp"),
                                     fp_nonhost_mm3_total=sum(r[f"fp_{k}_mm3"] for r in rr for k in NONHOST))
    json.dump(dict(band_mm=BAND_MM, summary=summ), open(out / "results_summary.json", "w", encoding="utf-8"), indent=1)
    for k, s in summ.items():
        print(k, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "predict", "evaluate", "all"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["base", "ft"])
    ap.add_argument("--raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--results", default=os.environ.get("nnUNet_results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"benchmark_v2_f{a.fold}"
    if a.step in ("generate", "all"):
        generate(a)
    if a.step in ("predict", "all"):
        predict(a)
    if a.step in ("evaluate", "all"):
        evaluate(a)


if __name__ == "__main__":
    main()
