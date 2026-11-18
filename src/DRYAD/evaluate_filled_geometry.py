"""
Evaluación armonizada "tibia rellena" (geometría externa) para validación
cruzada (Fischer, 44 tibias) y prueba externa (DRYAD, 4 tibias).

Motivo: las etiquetas de Fischer excluyen (en grado variable) el canal medular;
la referencia de DRYAD es hueso relleno. Para comparar con una misma regla se
define como objetivo el volumen encerrado por la superficie externa de la tibia
y se aplica EL MISMO relleno, SIN parámetros ajustables, a predicciones y
referencias de ambos conjuntos:

    relleno(m) = fill_holes_3D( fill_holes_2D por corte perpendicular al eje
                                 longitudinal de la ROI )

El eje longitudinal es el eje del array de mayor tamaño (eje e3 de la ROI de
525 mm). No se usa cierre morfológico ni umbral alguno (nada se ajusta con
los casos de prueba).

Se informan métricas CRUDAS (etiquetas tal cual) y RELLENAS, y para la tibia
rellena además por franjas anatómicas fijas del eje longitudinal (proximal 0-10 %,
10-20 %, diáfisis 20-80 %, distal 80-90 %, 90-100 %), idénticas para CV y externo.

Uso (desde la raíz del repositorio):
    .\\.venv_nnunet\\Scripts\\python.exe src\\DRYAD\\evaluate_filled_geometry.py

Salidas en results/evaluation/segmentation/filled_geometry/:
    cv_fischer_raw_vs_filled.csv
    external_dryad_raw_vs_filled.csv
    summary_raw_vs_filled.json
"""

from pathlib import Path
import csv
import json

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
MODEL = (ROOT / "data" / "nnunet" / "nnUNet_results" / "Dataset502_TibiaFischer"
         / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
FISCHER_GT = ROOT / "data" / "nnunet" / "nnUNet_raw" / "Dataset502_TibiaFischer" / "labelsTr"
DRYAD = ROOT / "data" / "interim" / "dryad_external" / "nnunet_test"
OUT = ROOT / "results" / "evaluation" / "segmentation" / "filled_geometry"


def fill(mask):
    ax = int(np.argmax(mask.shape))
    out = mask.copy()
    for i in range(mask.shape[ax]):
        idx = [slice(None)] * 3
        idx[ax] = i
        out[tuple(idx)] = ndi.binary_fill_holes(mask[tuple(idx)])
    return ndi.binary_fill_holes(out)


def surface(m):
    return m & ~ndi.binary_erosion(m, structure=ndi.generate_binary_structure(3, 1))


def metrics(p, r, sp):
    inter = (p & r).sum(); ps, rs = p.sum(), r.sum()
    dice = 2 * inter / (ps + rs); iou = inter / (p | r).sum()
    sp_, sr_ = surface(p), surface(r)
    d1 = ndi.distance_transform_edt(~sr_, sampling=sp)[sp_]
    d2 = ndi.distance_transform_edt(~sp_, sampling=sp)[sr_]
    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    assd = (d1.sum() + d2.sum()) / (len(d1) + len(d2))
    vox = float(np.prod(sp)) / 1000
    return dict(dice=dice, iou=iou, hd95_mm=hd95, assd_mm=assd,
                vol_ref_cm3=rs * vox, vol_pred_cm3=ps * vox,
                vol_diff_pct=100 * (ps - rs) / rs, precision=inter / ps, recall=inter / rs)


REGIONS = [("prox_00_10", 0.0, 0.1), ("prox_10_20", 0.1, 0.2), ("diaf_20_80", 0.2, 0.8),
           ("dist_80_90", 0.8, 0.9), ("dist_90_100", 0.9, 1.0)]


def regional(pf, rf, long_ax):
    """
    Métricas volumétricas por franjas del eje longitudinal, definidas sobre la
    extensión de la referencia rellena. En la ROI el índice crece de P1
    (proximal) a P2 (distal). Solo Dice/precisión/recall/volumen (las métricas
    de superficie por franja quedarían sesgadas por las caras de corte).
    """
    idx = np.nonzero(rf.any(axis=tuple(a for a in range(3) if a != long_ax)))[0]
    z0, z1 = idx.min(), idx.max() + 1
    L = z1 - z0
    out = {}
    for name, a, b in REGIONS:
        lo, hi = z0 + int(round(a * L)), z0 + int(round(b * L))
        sl = [slice(None)] * 3
        sl[long_ax] = slice(lo, hi)
        p, r = pf[tuple(sl)], rf[tuple(sl)]
        inter = (p & r).sum(); ps, rs = p.sum(), r.sum()
        out[f"{name}_dice"] = 2 * inter / (ps + rs) if ps + rs else np.nan
        out[f"{name}_precision"] = inter / ps if ps else np.nan
        out[f"{name}_recall"] = inter / rs if rs else np.nan
        out[f"{name}_vol_diff_pct"] = 100 * (ps - rs) / rs if rs else np.nan
    return out


def evaluate(pairs, label):
    rows = []
    for case, pred_path, ref_path in pairs:
        pi, ri = sitk.ReadImage(str(pred_path)), sitk.ReadImage(str(ref_path))
        if pi.GetSize() != ri.GetSize() or not np.allclose(pi.GetSpacing(), ri.GetSpacing(), atol=1e-4) \
                or not np.allclose(pi.GetOrigin(), ri.GetOrigin(), atol=1e-3):
            raise RuntimeError(f"{case}: geometría pred/ref distinta")
        p = sitk.GetArrayFromImage(pi) == 1
        r = sitk.GetArrayFromImage(ri) > 0
        sp = np.array(pi.GetSpacing()[::-1])
        raw = metrics(p, r, sp)
        pf, rf = fill(p), fill(r)
        fil = metrics(pf, rf, sp)
        row = dict(case=case,
                   ref_hollow_pct=round(100 * (1 - r.sum() / rf.sum()), 2),
                   pred_hollow_pct=round(100 * (1 - p.sum() / pf.sum()), 2))
        row.update({f"raw_{k}": round(float(v), 4) for k, v in raw.items()})
        row.update({f"filled_{k}": round(float(v), 4) for k, v in fil.items()})
        reg = regional(pf, rf, int(np.argmax(p.shape)))
        row.update({f"filled_{k}": round(float(v), 4) for k, v in reg.items()})
        rows.append(row)
        print(f"[{label}] {case:18s} Dice crudo={raw['dice']:.4f}  relleno={fil['dice']:.4f}  "
              f"| rell prox0-10={reg['prox_00_10_dice']:.3f} diáf={reg['diaf_20_80_dice']:.3f} "
              f"dist90-100={reg['dist_90_100_dice']:.3f}  "
              f"hueco ref={row['ref_hollow_pct']:.1f}%  hueco pred={row['pred_hollow_pct']:.1f}%")
    return rows


def write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def agg(rows):
    keys = [k for k in rows[0] if k.startswith(("raw_", "filled_")) or k.endswith("hollow_pct")]
    return {k: dict(mean=float(np.nanmean([r[k] for r in rows])),
                    sd=float(np.nanstd([r[k] for r in rows], ddof=1)),
                    median=float(np.nanmedian([r[k] for r in rows])),
                    min=float(np.nanmin([r[k] for r in rows])),
                    max=float(np.nanmax([r[k] for r in rows]))) for k in keys}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cv_pairs = []
    for f in range(5):
        for p in sorted((MODEL / f"fold_{f}" / "validation").glob("*.nii.gz")):
            cv_pairs.append((p.name[:-7], p, FISCHER_GT / p.name))
    print(f"Fischer CV: {len(cv_pairs)} tibias")
    cv = evaluate(cv_pairs, "CV")
    write(cv, OUT / "cv_fischer_raw_vs_filled.csv")

    ext_pairs = [(p.name[:-7], p, DRYAD / "labelsTs" / p.name)
                 for p in sorted((DRYAD / "predictions").glob("dryad_*.nii.gz"))]
    print(f"\nDRYAD externo: {len(ext_pairs)} tibias")
    ext = evaluate(ext_pairs, "EXT")
    write(ext, OUT / "external_dryad_raw_vs_filled.csv")

    summary = dict(
        fill_rule="fill_holes 2D por corte perpendicular al eje mayor del array + fill_holes 3D; sin parámetros",
        cv_fischer=dict(n=len(cv), metrics=agg(cv)),
        external_dryad=dict(n=len(ext), metrics=agg(ext)))
    with open(OUT / "summary_raw_vs_filled.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ RESUMEN (media ± DE) ================")
    for name, rows in (("CV Fischer (n=%d)" % len(cv), cv), ("Externo DRYAD (n=%d)" % len(ext), ext)):
        a = agg(rows)
        print(f"\n{name}")
        for k in ("dice", "iou", "hd95_mm", "assd_mm", "vol_diff_pct", "precision", "recall"):
            print(f"  {k:13s} crudo {a['raw_' + k]['mean']:8.4f} ± {a['raw_' + k]['sd']:.4f}   "
                  f"relleno {a['filled_' + k]['mean']:8.4f} ± {a['filled_' + k]['sd']:.4f}")
        print(f"  hueco en referencia: {a['ref_hollow_pct']['mean']:.1f} % "
              f"[{a['ref_hollow_pct']['min']:.1f} – {a['ref_hollow_pct']['max']:.1f}]")
        print("  Por región (tibia rellena):   Dice            precisión       recall          ΔV %")
        for name, _, _ in REGIONS:
            vals = [a[f"filled_{name}_{m}"] for m in ("dice", "precision", "recall", "vol_diff_pct")]
            print(f"    {name:12s} " + "  ".join(f"{v['mean']:7.3f}±{v['sd']:.3f}" for v in vals))
    print(f"\nResultados en: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
