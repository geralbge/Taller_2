"""
Evaluación de un caso patológico contra una segmentación manual de referencia.

Compara las segmentaciones de los modelos (p. ej. base Dataset502 y ajustado Dataset503,
fold 0) con la segmentación manual del hueso tibial hecha en 3D Slicer, y caracteriza el
defecto "real" (el que resulta de la máscara manual) con el MISMO algoritmo del pipeline
(defect_core.characterize: referencia interpolada, defecto angular, criterios).

Entradas
  --manual  máscara manual (.nii.gz / .nrrd / .seg.nrrd exportada como labelmap) en la grilla
            de la ROI (work/roi_0000.nii.gz). Si la grilla difiere se remuestrea (vecino más
            cercano) y se avisa. Hueso = cualquier etiqueta > 0, o --label N.
  --runs    carpetas de resultados del pipeline (con work/pred/roi.nii.gz y work/roi_0000.nii.gz)
  --names   nombres cortos para cada corrida
  --clicks  Defect_run1.mrk.json (D1, D2; mismos límites que usó run_case.py)
  --out     carpeta de salida

Salidas: eval_manual.json, eval_manual_perfil.png, defecto_manual.nii.gz, referencia_manual.nii.gz
         (+ variante "huésped": defecto_manual_huesped.nii.gz)

Variantes de la referencia manual
  A "todo el hueso": todo lo marcado como hueso tibial (cortical, callo, fragmentos).
  B "huésped": solo las componentes 3D conectadas a la tibia sana proximal o distal
    (bandas de 5 mm por fuera de D1/D2). Los fragmentos sueltos no cuentan como huésped.
  La TC no permite juzgar viabilidad biológica: A y B acotan el resultado. Si difieren en
  el cumplimiento de criterios, la decisión depende de la definición clínica -> experto.

Tolerancias de concordancia (las mismas usadas para calibrar el control de densidad):
  |dL| < 3 mm, |dV| < 10 %, |d pérdida cortical| < 10 pp.

Uso (desde la raíz del repositorio):
  .\\.venv_nnunet\\Scripts\\python.exe src\\defect\\eval_manual_reference.py ^
     --manual data\\interim\\pathological\\pat_01\\manual\\pat_01_manual.nii.gz ^
     --runs results\\pathological\\pat_01_ft results\\pathological\\pat_01_base_f0 --names ajustado base ^
     --clicks data\\interim\\pathological\\pat_01\\clicks\\Defect_run1.mrk.json ^
     --out results\\pathological\\pat_01_manual_eval
Dependencias: numpy, scipy, SimpleITK, matplotlib; src/defect/defect_core.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "defect"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import defect_core as dc  # noqa: E402

TOL = dict(dL_mm=3.0, dV_pct=10.0, dloss_pp=10.0)
MARGIN_MIN_MM = 25.0   # margen mínimo de la máscara manual por fuera de D1/D2 (banda + búsqueda)


# ------------------------------------------------------------------ E/S
def read_points(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    m = d["markups"][0]
    cs = m.get("coordinateSystem", "LPS")
    pts = {}
    for p in m["controlPoints"]:
        x = np.array(p["position"], float)
        if cs == "RAS":
            x[:2] *= -1
        pts[p["label"]] = x
    return pts


def same_grid(a, b, tol=1e-3):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def load_manual(path, ref_img, label=None):
    img = sitk.ReadImage(str(path))
    if img.GetNumberOfComponentsPerPixel() > 1:          # .seg.nrrd con capas
        img = sitk.VectorIndexSelectionCast(img, 0)
    resampled = False
    if not same_grid(img, ref_img):
        img = sitk.Resample(img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, img.GetPixelID())
        resampled = True
    a = sitk.GetArrayFromImage(img)
    m = (a == label) if label is not None else (a > 0)
    return m, resampled


# ------------------------------------------------------------------ métricas
def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def surface_dist(a, b, sp):
    """HD95 y ASSD (mm) entre superficies de a y b, en el recorte común."""
    if not a.any() or not b.any():
        return float("nan"), float("nan")
    idx = np.nonzero(a | b)
    sl = tuple(slice(max(int(x.min()) - 2, 0), int(x.max()) + 3) for x in idx)
    a, b = a[sl], b[sl]
    sa = a & ~ndi.binary_erosion(a)
    sb = b & ~ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sb, sampling=sp)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=sp)[sb]
    d = np.concatenate([da, db])
    return float(np.percentile(d, 95)), float(d.mean())


def seg_metrics(model, manual, sp, k_lo, k_hi):
    a, b = model[k_lo:k_hi + 1], manual[k_lo:k_hi + 1]
    vox = float(np.prod(sp))
    hd95, assd = surface_dist(a, b, sp)
    va, vb = a.sum() * vox, b.sum() * vox
    return dict(dice=dice(a, b), hd95_mm=hd95, assd_mm=assd,
                vol_modelo_mm3=float(va), vol_manual_mm3=float(vb),
                dif_vol_pct=float(100 * (va - vb) / vb) if vb else float("nan"),
                falso_neg_mm3=float((b & ~a).sum() * vox), falso_pos_mm3=float((a & ~b).sum() * vox))


def loss_profile(bone_raw, ref, sp, k1, k2):
    bone = dc.postprocess_bone(bone_raw)
    loss = np.full(bone.shape[0], np.nan)
    for k in range(k1, k2 + 1):
        _, loss[k] = dc.angular_defect_slice(ref[k], bone[k], sp[1:])
    return loss


def host_only(m, k1, k2, sp):
    """Componentes 3D conectadas a la tibia sana por fuera de D1/D2 (bandas de dc.BAND_MM)."""
    b = int(round(dc.BAND_MM / sp[0]))
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    touch = set(np.unique(lab[max(k1 - b, 0):k1])) | set(np.unique(lab[k2 + 1:k2 + 1 + b]))
    touch.discard(0)
    return np.isin(lab, list(touch)), int(n), int(len(touch))


def short(r):
    keys = ["defect_detected", "k_start", "k_end", "length_mm", "volume_mm3", "cortical_loss_max_pct",
            "cortical_loss_mean_pct", "n_segments", "diaphyseal", "meets_criteria", "qc_warning",
            "ref_band_shift_prox_mm", "ref_band_shift_dist_mm", "dense_frac_pct", "qc_density_warning"]
    return {k: (float(r[k]) if isinstance(r.get(k), (np.floating, float)) else
                int(r[k]) if isinstance(r.get(k), (np.integer,)) else r.get(k)) for k in keys if k in r}


def agreement(rm, rr, dm, dr):
    if not (rm.get("defect_detected") and rr.get("defect_detected")):
        return dict(dice_defecto=dice(dm, dr), concuerda=False)
    dL = rm["length_mm"] - rr["length_mm"]
    dV = 100 * (rm["volume_mm3"] - rr["volume_mm3"]) / rr["volume_mm3"]
    dP = rm["cortical_loss_max_pct"] - rr["cortical_loss_max_pct"]
    return dict(dice_defecto=dice(dm, dr), dL_mm=float(dL), dV_pct=float(dV), dloss_max_pp=float(dP),
                mismo_veredicto_criterios=bool(rm["meets_criteria"] == rr["meets_criteria"]),
                concuerda=bool(abs(dL) < TOL["dL_mm"] and abs(dV) < TOL["dV_pct"] and abs(dP) < TOL["dloss_pp"]
                               and rm["meets_criteria"] == rr["meets_criteria"]))


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", type=int, default=None)
    a = ap.parse_args()
    names = a.names or [Path(r).name for r in a.runs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ct, roi_img = dc.read_image(Path(a.runs[0]) / "work" / "roi_0000.nii.gz")
    preds = {}
    for n, r in zip(names, a.runs):
        p, pimg = dc.read_image(Path(r) / "work" / "pred" / "roi.nii.gz")
        if not same_grid(pimg, roi_img):
            raise SystemExit(f"{r}: la predicción no está en la grilla de la ROI de {a.runs[0]}")
        preds[n] = p == 1
    sp = dc.spacing_kji(roi_img)
    man, resampled = load_manual(a.manual, roi_img, a.label)
    if not man.any():
        raise SystemExit("La máscara manual está vacía (¿etiqueta equivocada o exportada sin geometría de referencia?).")

    pts = read_points(a.clicks)
    dpts = [pts[l] for l in ("D1", "D2")] if {"D1", "D2"} <= set(pts) else list(pts.values())[:2]
    k1, k2 = sorted(int(round(dc.physical_to_k(roi_img, p))) for p in dpts)
    ks = np.nonzero(man.any(axis=(1, 2)))[0]
    ka, kb = int(ks.min()), int(ks.max())
    avisos = []
    if resampled:
        avisos.append("La máscara manual no estaba en la grilla de la ROI: se remuestreó (vecino más cercano). "
                      "Revisar que se haya exportado con roi_0000 como geometría de referencia.")
    m_prox, m_dist = (k1 - ka) * sp[0], (kb - k2) * sp[0]
    if min(m_prox, m_dist) < MARGIN_MIN_MM:
        avisos.append(f"Margen de la máscara manual por fuera de D1/D2 = {m_prox:.0f} / {m_dist:.0f} mm "
                      f"(< {MARGIN_MIN_MM:.0f} mm): la banda sana de la referencia puede quedar fuera.")
    huecos = [int(k) for k in range(ka, kb + 1) if not man[k].any()]
    if huecos:
        avisos.append(f"{len(huecos)} cortes vacíos dentro de la máscara manual (k {huecos[:5]}...).")

    # extensión de la tibia (para el criterio diafisario): de la primera corrida (la máscara manual es un tramo)
    ext = dc.tibia_extent_k(dc.postprocess_bone(preds[names[0]]))

    def crop(m):
        x = np.zeros_like(m); x[ka:kb + 1] = m[ka:kb + 1]; return x

    # ---- defecto con la referencia manual (variantes A y B)
    host, ncomp, nhost = host_only(man, k1, k2, sp)
    res, dmask, rmask, loss = {}, {}, {}, {}
    for n, m in [("manual", man), ("manual_huesped", host)]:
        r, d, rf = dc.characterize(m, sp, k1, k2, tibia_extent=ext, ct=ct)
        res[n], dmask[n], rmask[n] = r, d, rf
        loss[n] = loss_profile(m, rf, sp, k1, k2)
    # ---- defecto de cada modelo: igual que el pipeline (máscara completa) y recortada al tramo manual
    for n in names:
        for tag, m in [(n, preds[n]), (n + "_recorte", crop(preds[n]))]:
            r, d, rf = dc.characterize(m, sp, k1, k2, tibia_extent=ext, ct=ct)
            res[tag], dmask[tag], rmask[tag] = r, d, rf
            if tag == n:
                loss[n] = loss_profile(m, rf, sp, k1, k2)

    # ---- métricas de segmentación (máscaras procesadas como en el pipeline: componentes + relleno)
    man_pp = dc.postprocess_bone(crop(man))
    seg = {}
    for n in names:
        mp = dc.postprocess_bone(crop(preds[n]))
        seg[n] = dict(ventana_D1_D2=seg_metrics(mp, man_pp, sp, k1, k2),
                      tramo_manual=seg_metrics(mp, man_pp, sp, ka, kb),
                      ventana_D1_D2_sin_relleno=seg_metrics(crop(preds[n]), man, sp, k1, k2))

    # ---- concordancia de defectos con la referencia manual (variante A)
    conc = {n: agreement(res[n], res["manual"], dmask[n], dmask["manual"]) for n in names}

    # ---- veredicto
    A, B = res["manual"], res["manual_huesped"]
    lines = []
    if A["meets_criteria"] and B["meets_criteria"]:
        lines.append("Con la referencia manual, pat_01 CUMPLE los criterios (en las dos variantes).")
    elif not A["meets_criteria"] and not B["meets_criteria"]:
        lines.append("Con la referencia manual, pat_01 NO CUMPLE los criterios (en las dos variantes): "
                     "no es un defecto segmentario crítico según la definición del proyecto -> excluir del FEA.")
    else:
        lines.append("El cumplimiento depende de si los fragmentos sueltos cuentan como hueso huésped "
                     f"(todo el hueso: {A['meets_criteria']}; solo huésped: {B['meets_criteria']}) -> "
                     "decisión clínica, llevar a validación por experto.")
    if A["meets_criteria"] or B["meets_criteria"]:
        ok = [n for n in names if conc[n]["concuerda"]]
        lines.append(("Modelos que concuerdan con la referencia (tolerancias): " + ", ".join(ok)) if ok else
                     "Ningún modelo concuerda con la referencia dentro de las tolerancias: para el FEA usar "
                     "defecto_manual.nii.gz (delimitación corregida manualmente) y declararlo.")

    summary = dict(manual=str(a.manual), runs=dict(zip(names, a.runs)), k1=k1, k2=k2,
                   tramo_manual_k=[ka, kb], margen_mm=[float(m_prox), float(m_dist)],
                   spacing_kji_mm=sp.tolist(), extension_tibia_k=list(ext),
                   componentes_manual=ncomp, componentes_huesped=nhost,
                   tolerancias=TOL, avisos=avisos,
                   defecto={k: short(v) for k, v in res.items()},
                   segmentacion=seg, concordancia_defecto=conc, veredicto=lines)
    (out / "eval_manual.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    dc.write_like(dmask["manual"], roi_img, out / "defecto_manual.nii.gz")
    dc.write_like(rmask["manual"], roi_img, out / "referencia_manual.nii.gz")
    dc.write_like(dmask["manual_huesped"], roi_img, out / "defecto_manual_huesped.nii.gz")

    # ---- figura: área por corte y pérdida cortical por corte
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kk = np.arange(max(ka, k1 - 40), min(kb, k2 + 40) + 1)
    z = kk * sp[0]
    area = lambda m: dc.postprocess_bone(crop(m))[kk].sum(axis=(1, 2)) * sp[1] * sp[2]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(z, area(man), "k", lw=2, label="manual")
    for n in names:
        ax[0].plot(z, area(preds[n]), lw=1.2, label=n)
    ax[0].set_ylabel("área ósea (mm²)")
    for n in names:
        mp = dc.postprocess_bone(crop(preds[n]))
        ax[1].plot(z, [dice(mp[k], man_pp[k]) for k in kk], lw=1.2, label=n)
    ax[1].set_ylabel("Dice por corte vs manual"); ax[1].set_ylim(0, 1.02)
    for n, st in [("manual", "k-"), ("manual_huesped", "k--")] + [(n, "-") for n in names]:
        ax[2].plot(z, loss[n][kk], st, lw=2 if n.startswith("manual") else 1.2, label=n)
    ax[2].axhline(50, color="r", ls=":", label="umbral 50 %")
    ax[2].set_ylabel("pérdida cortical (%)"); ax[2].set_xlabel("posición en la ROI (mm, k·Δk)")
    for x in ax:
        x.axvspan(k1 * sp[0], k2 * sp[0], color="0.9", zorder=0)
        x.legend(fontsize=8, loc="best")
    ax[0].set_title("pat_01: modelos vs segmentación manual (zona gris = D1–D2)")
    plt.tight_layout(); plt.savefig(out / "eval_manual_perfil.png", dpi=110); plt.close()

    # ---- resumen en consola
    print("\n==== RESUMEN ====")
    for w in avisos:
        print("AVISO:", w)
    print(f"D1–D2: k {k1}–{k2} ({(k2 - k1 + 1) * sp[0]:.1f} mm); tramo manual k {ka}–{kb}")
    for n in ["manual", "manual_huesped"] + names:
        r = res[n]
        print(f"{n:16s} L={r.get('length_mm', 0):5.1f} mm  V={r.get('volume_mm3', 0):7.0f} mm³  "
              f"pérdida máx={r.get('cortical_loss_max_pct', 0):5.1f} %  tramos={r['n_segments']}  "
              f"criterios={r['meets_criteria']}")
    for n in names:
        s, c = seg[n]["ventana_D1_D2"], conc[n]
        print(f"{n:16s} hueso D1–D2: Dice={s['dice']:.3f} HD95={s['hd95_mm']:.1f} mm  "
              f"FN={s['falso_neg_mm3']:.0f} mm³ FP={s['falso_pos_mm3']:.0f} mm³ | defecto: Dice={c['dice_defecto']:.2f} "
              + (f"dL={c['dL_mm']:+.1f} mm dV={c['dV_pct']:+.0f} % dPérd={c['dloss_max_pp']:+.0f} pp" if 'dL_mm' in c else ""))
    print("\n".join(lines))
    print("Salidas en", out)


if __name__ == "__main__":
    main()
