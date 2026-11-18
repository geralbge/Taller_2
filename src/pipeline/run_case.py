"""
FLUJO COMPLETO DEL PROTOTIPO (demostración): TC + clics -> andamio.

  [1] TC (DICOM o NIfTI) -> HU
  [2] ROI orientada de la tibia afectada (clics P1, P2) [+ contralateral]
  [3] Segmentación nnU-Net (por defecto el modelo AJUSTADO con defectos v2,
      Dataset503 fold 0, en models/; --model base usa el Dataset502 original)
  [4] Post-procesamiento + reconstrucción 3D (STL, LPS mm)
  [5] Caracterización del defecto entre D1 y D2:
        referencia por interpolación (siempre) y contralateral (si hay clics de
        la otra tibia); criterios del proyecto; controles de calidad
  [6] Modelo 3D del defecto (= envolvente del andamio)
  [7] Andamio poroso TPMS (giroide-lámina, celda 2,5 mm, pared 0,3 mm por defecto)
      + métricas de arquitectura porosa y verificación geométrica
  [8] Reporte JSON + resumen en consola con tiempos por etapa

Clics (3D Slicer, Markups -> Save, en <clicks>/):
  Tibia_run1.mrk.json         P1 (proximal), P2 (distal) sobre la tibia afectada
  Defect_run1.mrk.json        D1, D2: hueso sano justo por encima/debajo del defecto
  Tibia_contra_run1.mrk.json  (opcional) P1, P2 sobre la tibia de la otra pierna

Uso (desde la raíz del repositorio):
  .\\.venv_nnunet\\Scripts\\python.exe src\\pipeline\\run_case.py --ct <TC.nii.gz | carpeta DICOM> --clicks <carpeta> --out <carpeta>
      [--folds 0] [--model base|ft] [--reference auto|interp|contra] [--tta]
      [--cell 2.5] [--wall 0.3] [--no-scaffold]
      F4 (opcionales; sin ellos el andamio es el de siempre):
      [--material Ti6Al4V_LPBF|PCL_bTCP_extrusion] [--nail auto [--nail-diameter 10]] [--plug 8]
      [--grad T_EXT T_INT P1 P2]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("defect", "reconstruction", "scaffold"):
    sys.path.insert(0, str(ROOT / "src" / sub))
import contralateral as cl          # noqa: E402
import defect_core as dc            # noqa: E402
import generate_scaffold as gs      # noqa: E402
import reconstruct_tibia as rt      # noqa: E402

ROI_SCRIPT = ROOT / "src" / "FISCHER" / "create_oriented_tibia_roi.py"
NN = ROOT / "data" / "nnunet"
MODELS = {
    "base": NN / "nnUNet_results" / "Dataset502_TibiaFischer" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "ft": ROOT / "models" / "Dataset503_TibiaDefectsF0" / "nnUNetTrainer_FT__nnUNetResEncUNetMPlans__3d_fullres",
}
DEFAULT_MODEL = "ft"   # vigente desde 04/11/2026 (Pista A: F2 ajuste fino, F3 re-evaluación)


def _allow_ft_trainer():
    """nnU-Net busca la clase del trainer por nombre dentro de su paquete. nnUNetTrainer_FT
    solo cambia épocas y tasa de aprendizaje (misma red), así que para INFERIR basta con
    nnUNetTrainer si nnUNetTrainer_FT no está copiado en site-packages."""
    import nnunetv2.inference.predict_from_raw_data as prd
    if hasattr(prd, "recursive_find_trainer_class_by_name"):     # nnunetv2 reciente
        orig = prd.recursive_find_trainer_class_by_name
        if getattr(orig, "_ft_patch", False):
            return

        def find_t(name):
            try:
                return orig(name)
            except RuntimeError:
                if name != "nnUNetTrainer_FT":
                    raise
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
                return nnUNetTrainer
        find_t._ft_patch = True
        prd.recursive_find_trainer_class_by_name = find_t
        return
    orig = prd.recursive_find_python_class                        # nnunetv2 antiguo
    if getattr(orig, "_ft_patch", False):
        return

    def find(folder, name, current_module):
        c = orig(folder, name, current_module)
        if c is None and name == "nnUNetTrainer_FT":
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
            return nnUNetTrainer
        return c
    find._ft_patch = True
    prd.recursive_find_python_class = find
CONTRA_QC = dict(rms_mm=1.5, inlier_frac=0.80)   # registro aceptable para usar la contralateral


class Timer:
    def __init__(self):
        self.t = {}; self._t0 = time.time(); self._last = self._t0

    def lap(self, name):
        now = time.time(); self.t[name] = round(now - self._last, 1); self._last = now
        print(f"   ({self.t[name]:.0f} s)")

    def total(self):
        return round(time.time() - self._t0, 1)


def read_points(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["markups"][0]
    cs = str(m.get("coordinateSystem", d.get("coordinateSystem", "LPS"))).upper()
    pts = [np.array(c["position"], float) for c in m["controlPoints"] if c.get("position") is not None]
    if len(pts) != 2:
        raise ValueError(f"{path.name}: se esperaban 2 puntos, hay {len(pts)}")
    return [p * np.array([-1, -1, 1]) for p in pts] if cs == "RAS" else pts


def load_ct(ct_path, work):
    """DICOM (carpeta) o NIfTI -> NIfTI HU int16 en 'work'."""
    import SimpleITK as sitk
    ct_path = Path(ct_path)
    if ct_path.is_file():
        return ct_path
    out = work / "ct_HU.nii.gz"
    if out.is_file():
        return out
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(ct_path))
    if not ids:
        raise RuntimeError(f"Sin series DICOM en {ct_path}")
    best = max(ids, key=lambda i: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_path), i)))
    r = sitk.ImageSeriesReader(); r.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_path), best))
    r.MetaDataDictionaryArrayUpdateOff(); r.LoadPrivateTagsOff()
    img = r.Execute()
    arr = np.clip(np.round(sitk.GetArrayFromImage(img).astype(np.float32)), -1024, 3071).astype(np.int16)
    o = sitk.GetImageFromArray(arr); o.CopyInformation(img)
    sitk.WriteImage(o, str(out), useCompression=True)
    return out


def make_roi(ct, markup, roi_path):
    if roi_path.is_file():
        return
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROI_SCRIPT), "--ct", str(ct), "--markup", str(markup),
                        "--out", str(roi_path), "--metadata", str(roi_path.with_suffix("").with_suffix(".json"))],
                       text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError("Falló la generación de la ROI")


def segment(rois, preds, model_dir, folds, tta):
    todo = [(r, p) for r, p in zip(rois, preds) if not p.is_file()]
    if not todo:
        return "ya existían"
    import torch
    os.environ.setdefault("nnUNet_results", str(NN / "nnUNet_results"))
    os.environ.setdefault("nnUNet_raw", str(NN / "nnUNet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(NN / "nnUNet_preprocessed"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    _allow_ft_trainer()
    cuda = torch.cuda.is_available()
    if not cuda:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta, perform_everything_on_device=cuda,
                         device=torch.device("cuda" if cuda else "cpu"), verbose=False, allow_tqdm=True)
    pr.initialize_from_trained_model_folder(str(model_dir), use_folds=tuple(folds), checkpoint_name="checkpoint_final.pth")
    for _, p in todo:
        p.parent.mkdir(parents=True, exist_ok=True)
    pr.predict_from_files([[str(r)] for r, _ in todo], [str(p) for _, p in todo], save_probabilities=False,
                          overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
    return "cuda" if cuda else "cpu"


def save_mesh(mask, img, path, cad=False):
    v, f = rt.mask_to_mesh(mask, np.array(img.GetSpacing()), np.array(img.GetOrigin()),
                           np.array(img.GetDirection()).reshape(3, 3), step=rt.CAD_STEP if cad else 1)
    v = rt.taubin(v, f)
    rt.write_stl(path, v, f, path.stem)
    return dict(caras=len(f), watertight=bool(rt.is_watertight(f)), volumen_mm3=abs(rt.mesh_volume(v, f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    ap.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    ap.add_argument("--reference", choices=["auto", "interp", "contra"], default="auto")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cell", type=float, default=2.5)
    ap.add_argument("--wall", type=float, default=0.3)
    ap.add_argument("--no-scaffold", action="store_true")
    ap.add_argument("--material", default="Ti6Al4V_LPBF", choices=list(gs.f4.MATERIALS))
    ap.add_argument("--nail", default="none", choices=["none", "auto"], help="canal para clavo intramedular detectado en la TC")
    ap.add_argument("--nail-diameter", type=float, default=None)
    ap.add_argument("--plug", type=float, default=0.0, help="tapón intramedular (mm); 0 = sin tapón")
    ap.add_argument("--grad", type=float, nargs=4, default=None, metavar=("T_EXT", "T_INT", "P1", "P2"))
    a = ap.parse_args()

    T = Timer()
    out = Path(a.out); work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    clicks = Path(a.clicks)
    mk, dmk, cmk = clicks / "Tibia_run1.mrk.json", clicks / "Defect_run1.mrk.json", clicks / "Tibia_contra_run1.mrk.json"
    for p in (mk, dmk):
        if not p.is_file():
            raise FileNotFoundError(p)
    has_contra = cmk.is_file()
    rep = dict(ct=str(a.ct), modelo=a.model, folds=a.folds, tta=a.tta)

    print("[1] TC"); ct = load_ct(a.ct, work); T.lap("tc")
    print("[2] ROI orientada" + (" (+ contralateral)" if has_contra else ""))
    roi = work / "roi_0000.nii.gz"; make_roi(ct, mk, roi)
    rois, preds = [roi], [work / "pred" / "roi.nii.gz"]
    if has_contra:
        roic = work / "roi_contra_0000.nii.gz"; make_roi(ct, cmk, roic)
        rois.append(roic); preds.append(work / "pred" / "roi_contra.nii.gz")
    T.lap("roi")

    model_dir = MODELS[a.model]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Modelo no encontrado: {model_dir}")
    print(f"[3] Segmentación nnU-Net ({a.model}, folds {a.folds})")
    rep["dispositivo"] = segment(rois, preds, model_dir, a.folds, a.tta); T.lap("segmentacion")

    print("[4] Reconstrucción 3D")
    roi_arr, roi_img = dc.read_image(roi)
    pred, pimg = dc.read_image(preds[0])
    sp = dc.spacing_kji(pimg)
    bone = dc.postprocess_bone(pred == 1)
    dc.write_like(bone, pimg, out / "tibia_segmentada.nii.gz")
    rep["tibia"] = dict(volumen_cm3=float(bone.sum() * np.prod(sp) / 1000),
                        fragmentos=int(dc.ndi.label(bone, structure=np.ones((3, 3, 3)))[1]),
                        malla=save_mesh(bone, pimg, out / "tibia.stl"),
                        malla_cad=save_mesh(bone, pimg, out / "tibia_cad.stl", cad=True))
    T.lap("reconstruccion")

    print("[5] Caracterización del defecto")
    d1, d2 = read_points(dmk)
    k1, k2 = sorted(int(round(dc.physical_to_k(pimg, p))) for p in (d1, d2))
    ext = dc.tibia_extent_k(bone)
    res_i, def_i, ref_i = dc.characterize(pred == 1, sp, k1, k2, tibia_extent=ext, ct=roi_arr)
    rep["defecto_interpolacion"] = res_i
    chosen, res, defect, ref = "interp", res_i, def_i, ref_i
    if has_contra:
        cpred, cimg = dc.read_image(preds[1])
        cbone = dc.postprocess_bone(cpred == 1)
        reg = cl.register(cbone, cimg, bone, pimg, k1, k2, sp[0])
        ref_c = cl.reference_from_contralateral(cbone, cimg, bone.shape, pimg, k1, k2, reg, cl.bbox_ji(bone, sp))
        res_c, def_c, ref_c = dc.characterize(pred == 1, sp, k1, k2, tibia_extent=ext, reference=ref_c, ct=roi_arr)
        reg_ok = reg["rms_mm"] <= CONTRA_QC["rms_mm"] and reg["inlier_frac"] >= CONTRA_QC["inlier_frac"]
        rep["registro_contralateral"] = dict(rms_mm=reg["rms_mm"], inlier_frac=reg["inlier_frac"],
                                             iteraciones=reg["iterations"], aceptable=bool(reg_ok))
        rep["defecto_contralateral"] = res_c
        a_ = ref_i & ref_c
        rep["dice_referencias_interp_vs_contra"] = float(2 * a_.sum() / max(ref_i.sum() + ref_c.sum(), 1))
        use_contra = (a.reference == "contra") or (a.reference == "auto" and reg_ok
                                                     and res_i.get("ref_band_shift_prox_mm", 0) + res_i.get("ref_band_shift_dist_mm", 0) > 2.0)
        if use_contra:
            chosen, res, defect, ref = "contra", res_c, def_c, ref_c
    elif a.reference == "contra":
        raise FileNotFoundError(f"--reference contra requiere {cmk.name}")
    rep["referencia_usada"] = chosen
    rep["defecto"] = res
    dc.write_like(defect, pimg, out / "defecto.nii.gz")
    dc.write_like(ref, pimg, out / "referencia.nii.gz")
    T.lap("caracterizacion")

    if not res.get("defect_detected"):
        print("   No se detectó defecto entre D1 y D2: no se genera andamio.")
    else:
        print("[6] Modelo 3D del defecto")
        rep["malla_defecto"] = save_mesh(defect, pimg, out / "defecto.stl")
        rep["malla_referencia"] = save_mesh(ref, pimg, out / "referencia.stl")
        T.lap("modelo_defecto")
        if not a.no_scaffold:
            print("[7] Andamio TPMS")
            use_ct = a.nail == "auto" or a.nail_diameter
            sr = gs.generate(out / "defecto.nii.gz", out, cell=a.cell, wall=a.wall, name="andamio",
                             material=a.material, ct_path=roi if use_ct else None, nail=a.nail,
                             nail_diameter=a.nail_diameter, bone_raw_path=preds[0] if a.plug > 0 else None,
                             plug_mm=a.plug,
                             grad=dict(zip(("t_ext", "t_int", "p1", "p2"), a.grad)) if a.grad else None)
            rep["andamio"] = sr
            T.lap("andamio")

    rep["tiempos_s"] = T.t; rep["tiempo_total_s"] = T.total()
    dc.save_json(rep, out / "reporte.json")

    print("\n================ RESUMEN ================")
    print(f"Tibia: {rep['tibia']['volumen_cm3']:.1f} cm3, fragmentos {rep['tibia']['fragmentos']}")
    print(f"Referencia usada: {chosen}" + (f"  (registro contralateral RMS {rep['registro_contralateral']['rms_mm']:.2f} mm, "
                                          f"Dice interp-vs-contra {rep['dice_referencias_interp_vs_contra']:.3f})" if has_contra else ""))
    if res.get("defect_detected"):
        print(f"Defecto: longitud {res['length_mm']:.1f} mm, volumen {res['volume_mm3'] / 1000:.2f} cm3, "
              f"pérdida cortical máx {res['cortical_loss_max_pct']:.0f} % (media {res['cortical_loss_mean_pct']:.0f} %), "
              f"segmentos {res['n_segments']}, diafisario {res.get('diaphyseal')}")
        print(f"Cumple criterios del proyecto: {res['meets_criteria']}")
    if res.get("qc_warning"):
        print("AVISO QC: " + ("densidad ósea dentro del defecto (posible sub-segmentación); " if res.get("qc_density_warning") else "")
              + ("banda sana desplazada; " if max(res.get('ref_band_shift_prox_mm', 0), res.get('ref_band_shift_dist_mm', 0)) > 2 else "")
              + "revisar antes de usar el andamio.")
    if "andamio" in rep:
        print("Andamio: " + "\n         ".join(gs.summary_lines(rep["andamio"])))
    print("Tiempos (s): " + ", ".join(f"{k} {v:.0f}" for k, v in T.t.items()) + f"  | total {rep['tiempo_total_s'] / 60:.1f} min")
    print(f"Salida: {out}")


if __name__ == "__main__":
    main()
