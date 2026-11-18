"""
Prueba externa del modelo de segmentación tibial (nnU-Net Dataset502) en DRYAD.

Casos: S192803 L/R y S193761 L/R (4 tibias, 2 sujetos, nunca usados en
entrenamiento, validación ni ajuste).

Etapas (se ejecutan en orden; las ya hechas se saltan salvo --overwrite):
  1) ROI   : ROI orientada 120x120x525 mm con el MISMO generador usado en
             Fischer (src/FISCHER/create_oriented_tibia_roi.py) a partir de
             los clics P1/P2 de 3D Slicer.
  2) REF   : referencia de tibia (etiqueta 1 del split; usa *_corrected.* si
             existe) remuestreada a la ROI con vecino más cercano, como en
             prepare_fischer_gt_rois.py. Informa la fracción de tibia retenida.
  3) PRED  : inferencia con el ensemble de los 5 folds (checkpoint_final.pth),
             nnUNetResEncUNetMPlans, 3d_fullres. Mirroring (TTA) activado por
             defecto, igual que la validación de nnU-Net.
  4) METR  : Dice, IoU, HD95, ASSD (mm), volúmenes y diferencia de volumen.

Uso (desde la raíz del repositorio):
    .\\.venv_nnunet\\Scripts\\python.exe src\\DRYAD\\evaluate_dryad_external.py
Opciones:
    --no-tta      inferencia sin mirroring (más rápida en CPU; se registra)
    --overwrite   rehace todas las etapas
    --stage roi|ref|pred|metrics|all   (por defecto all)

Salidas:
    data/interim/dryad_external/nnunet_test/imagesTs/<case>_0000.nii.gz
    data/interim/dryad_external/nnunet_test/labelsTs/<case>.nii.gz
    data/interim/dryad_external/nnunet_test/predictions/<case>.nii.gz
    results/evaluation/dryad_external/external_test_metrics.csv
    results/evaluation/dryad_external/external_test_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
DRYAD = ROOT / "data" / "interim" / "dryad_external"
TEST = DRYAD / "nnunet_test"
IMG_DIR = TEST / "imagesTs"
LBL_DIR = TEST / "labelsTs"
PRED_DIR = TEST / "predictions"
META_DIR = TEST / "roi_metadata"
REPORT = ROOT / "results" / "evaluation" / "dryad_external"
ROI_SCRIPT = ROOT / "src" / "FISCHER" / "create_oriented_tibia_roi.py"

NNUNET_RESULTS = ROOT / "data" / "nnunet" / "nnUNet_results"
MODEL_DIR = (NNUNET_RESULTS / "Dataset502_TibiaFischer"
             / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
FOLDS = (0, 1, 2, 3, 4)
CHECKPOINT = "checkpoint_final.pth"

CASES = [("S192803", "L", "Left"), ("S192803", "R", "Right"),
         ("S193761", "L", "Left"), ("S193761", "R", "Right")]


def case_id(subj, s):
    return f"dryad_{subj}_{s}"


def geometry_equal(a, b, tol=1e-4):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def find_split(subj, side_long):
    d = DRYAD / subj
    for name in (f"{subj}_{side_long}_tibfib_split_corrected.nrrd",
                 f"{subj}_{side_long}_tibfib_split_corrected.nii.gz",
                 f"{subj}_{side_long}_tibfib_split.nii.gz"):
        if (d / name).is_file():
            return d / name, ("corrected" in name)
    raise FileNotFoundError(f"No hay split para {subj} {side_long}")


# ------------------------------------------------------------------ etapa 1
def stage_roi(overwrite):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for subj, s, _ in CASES:
        cid = case_id(subj, s)
        out = IMG_DIR / f"{cid}_0000.nii.gz"
        if out.is_file() and not overwrite:
            print(f"[ROI] {cid}: ya existe")
            continue
        ct = DRYAD / subj / f"{subj}_CT_HU.nii.gz"
        mk = DRYAD / subj / f"dryad_{subj}_clicks" / f"Tibia_{s}_run1.mrk.json"
        for p in (ct, mk, ROI_SCRIPT):
            if not p.is_file():
                raise FileNotFoundError(p)
        print(f"[ROI] {cid}: generando ...")
        r = subprocess.run([sys.executable, str(ROI_SCRIPT), "--ct", str(ct),
                            "--markup", str(mk), "--out", str(out),
                            "--metadata", str(META_DIR / f"{cid}_roi_metadata.json")],
                           text=True, capture_output=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise RuntimeError(f"Falló la ROI de {cid}")
        roi = sitk.ReadImage(str(out))
        print(f"      size={roi.GetSize()} spacing={tuple(round(x, 3) for x in roi.GetSpacing())}")


# ------------------------------------------------------------------ etapa 2
def stage_ref(overwrite):
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for subj, s, side_long in CASES:
        cid = case_id(subj, s)
        out = LBL_DIR / f"{cid}.nii.gz"
        roi = sitk.ReadImage(str(IMG_DIR / f"{cid}_0000.nii.gz"))
        split_path, corrected = find_split(subj, side_long)
        split = sitk.ReadImage(str(split_path))
        ct_ref = sitk.ReadImage(str(DRYAD / subj / f"{subj}_Left_tibfib_split.nii.gz"))
        if not geometry_equal(split, ct_ref):
            raise RuntimeError(f"Geometría del split distinta a la original: {split_path}")
        arr = sitk.GetArrayFromImage(split)
        vox = float(np.prod(split.GetSpacing()))
        tib_cm3 = (arr == 1).sum() * vox / 1000
        fib_cm3 = (arr == 2).sum() * vox / 1000
        if not tib_cm3 > 2 * fib_cm3:
            raise RuntimeError(f"{cid}: etiqueta 1 no parece tibia ({tib_cm3:.1f} vs {fib_cm3:.1f} cm3)")
        tib = sitk.GetImageFromArray((arr == 1).astype(np.uint8))
        tib.CopyInformation(split)
        del arr

        res = sitk.ResampleImageFilter()
        res.SetReferenceImage(roi)
        res.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        res.SetInterpolator(sitk.sitkNearestNeighbor)
        res.SetDefaultPixelValue(0)
        res.SetOutputPixelType(sitk.sitkUInt8)
        tib_roi = res.Execute(tib)
        if not geometry_equal(tib_roi, roi):
            raise RuntimeError("Geometría ROI/ref distinta")
        roi_cm3 = sitk.GetArrayViewFromImage(tib_roi).sum() * float(np.prod(roi.GetSpacing())) / 1000
        retention = roi_cm3 / tib_cm3
        flag = "" if retention >= 0.98 else "  <-- AVISO: la ROI corta parte de la tibia"
        print(f"[REF] {cid}: split={'corregido' if corrected else 'original'}  "
              f"tibia={tib_cm3:.1f} cm3  en ROI={roi_cm3:.1f} cm3  retención={retention:.3f}{flag}")
        if overwrite or not out.is_file():
            sitk.WriteImage(tib_roi, str(out), useCompression=True)
        rows.append(dict(case=cid, split_file=split_path.name, corrected=corrected,
                         tibia_full_cm3=round(tib_cm3, 2), tibia_in_roi_cm3=round(roi_cm3, 2),
                         roi_retention=round(retention, 4)))
    return rows


# ------------------------------------------------------------------ etapa 3
def stage_pred(overwrite, tta):
    import torch
    os.environ.setdefault("nnUNet_results", str(NNUNET_RESULTS))
    os.environ.setdefault("nnUNet_raw", str(ROOT / "data" / "nnunet" / "nnUNet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(ROOT / "data" / "nnunet" / "nnUNet_preprocessed"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pending = [case_id(a, b) for a, b, _ in CASES
               if overwrite or not (PRED_DIR / f"{case_id(a, b)}.nii.gz").is_file()]
    if not pending:
        print("[PRED] todas las predicciones ya existen")
        return
    for f in FOLDS:
        p = MODEL_DIR / f"fold_{f}" / CHECKPOINT
        if not p.is_file():
            raise FileNotFoundError(p)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"[PRED] dispositivo={device}  TTA={'sí' if tta else 'no'}  folds={FOLDS}  casos={pending}")
    if not cuda:
        print("       AVISO: sin GPU; puede tardar bastante (5 folds x TTA).")
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                                perform_everything_on_device=cuda, device=device,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(MODEL_DIR), use_folds=FOLDS,
                                                   checkpoint_name=CHECKPOINT)
    t0 = time.time()
    predictor.predict_from_files([[str(IMG_DIR / f"{c}_0000.nii.gz")] for c in pending],
                                 [str(PRED_DIR / f"{c}.nii.gz") for c in pending],
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=1,
                                 num_processes_segmentation_export=1)
    print(f"[PRED] listo en {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------ etapa 4
def surface(mask):
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1))


def metrics(pred, ref, spacing_zyx):
    p, r = pred > 0, ref > 0
    inter = np.logical_and(p, r).sum()
    ps, rs = p.sum(), r.sum()
    dice = 2 * inter / (ps + rs) if (ps + rs) else np.nan
    iou = inter / np.logical_or(p, r).sum() if (ps + rs) else np.nan
    if ps == 0 or rs == 0:
        return dice, iou, np.nan, np.nan
    sp, sr = surface(p), surface(r)
    d_to_r = ndi.distance_transform_edt(~sr, sampling=spacing_zyx)[sp]
    d_to_p = ndi.distance_transform_edt(~sp, sampling=spacing_zyx)[sr]
    hd95 = max(np.percentile(d_to_r, 95), np.percentile(d_to_p, 95))
    assd = (d_to_r.sum() + d_to_p.sum()) / (len(d_to_r) + len(d_to_p))
    return dice, iou, hd95, assd


def stage_metrics(ref_rows, tta):
    REPORT.mkdir(parents=True, exist_ok=True)
    ref_info = {r["case"]: r for r in (ref_rows or [])}
    rows = []
    for subj, s, _ in CASES:
        cid = case_id(subj, s)
        ref_img = sitk.ReadImage(str(LBL_DIR / f"{cid}.nii.gz"))
        pred_img = sitk.ReadImage(str(PRED_DIR / f"{cid}.nii.gz"))
        if not geometry_equal(ref_img, pred_img):
            raise RuntimeError(f"{cid}: geometría pred/ref distinta")
        ref = sitk.GetArrayFromImage(ref_img) > 0
        pred = sitk.GetArrayFromImage(pred_img) == 1
        sp = np.array(ref_img.GetSpacing()[::-1])
        vox = float(np.prod(sp))
        d, j, h, a = metrics(pred, ref, sp)
        vr, vp = ref.sum() * vox / 1000, pred.sum() * vox / 1000
        row = dict(case=cid, subject=subj, side=s, dice=round(d, 4), iou=round(j, 4),
                   hd95_mm=round(h, 3), assd_mm=round(a, 3),
                   vol_ref_cm3=round(vr, 2), vol_pred_cm3=round(vp, 2),
                   vol_diff_pct=round(100 * (vp - vr) / vr, 2), tta=tta)
        row.update({k: v for k, v in ref_info.get(cid, {}).items() if k != "case"})
        rows.append(row)
        print(f"[METR] {cid}: Dice={d:.4f} IoU={j:.4f} HD95={h:.2f} mm ASSD={a:.3f} mm "
              f"Vref={vr:.1f} Vpred={vp:.1f} cm3 ({row['vol_diff_pct']:+.1f} %)")

    with open(REPORT / "external_test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    agg = {m: dict(mean=float(np.mean([r[m] for r in rows])), sd=float(np.std([r[m] for r in rows], ddof=1)),
                   min=float(np.min([r[m] for r in rows])), max=float(np.max([r[m] for r in rows])))
           for m in ("dice", "iou", "hd95_mm", "assd_mm", "vol_diff_pct")}
    summary = dict(dataset="DRYAD (Andreassen et al.) S192803, S193761",
                   n_subjects=2, n_tibias=len(rows), model="Dataset502 nnUNetResEncUNetM 3d_fullres",
                   folds=list(FOLDS), checkpoint=CHECKPOINT, tta=tta,
                   reference="TibFib del dataset separada semiautomáticamente (watershed) "
                             "con corrección manual donde se indica",
                   hd95_definition="max(percentil 95 dirigido pred->ref, ref->pred)",
                   aggregate=agg)
    with open(REPORT / "external_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nResumen (media ± DE, n=4):")
    for m, v in agg.items():
        print(f"  {m:13s} {v['mean']:.4f} ± {v['sd']:.4f}   [{v['min']:.4f} – {v['max']:.4f}]")
    print(f"\nCSV : {(REPORT / 'external_test_metrics.csv').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["roi", "ref", "pred", "metrics", "all"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    tta = not a.no_tta
    ref_rows = None
    if a.stage in ("roi", "all"):
        stage_roi(a.overwrite)
    if a.stage in ("ref", "all", "metrics"):
        ref_rows = stage_ref(a.overwrite)
    if a.stage in ("pred", "all"):
        stage_pred(a.overwrite, tta)
    if a.stage in ("metrics", "all"):
        stage_metrics(ref_rows, tta)


if __name__ == "__main__":
    main()
