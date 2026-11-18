"""
Crea el dataset nnU-Net para el AJUSTE FINO (fold 0) con el generador de defectos realistas v2.

Para cada tibia del fold (entrenamiento Y validación, sin mezclar entre ellos):
  - copia el caso original intacto (control: el modelo no debe olvidar la tibia sana)
  - genera N versiones con un defecto realista, escenario sorteado con la mezcla
      A (post-trauma) 0,3 · B (infección desbridada, con o sin espaciador) 0,3 ·
      C (no unión establecida) 0,3 · D (resección limpia, control) 0,1
    longitud 20-80 mm (rango de entrenamiento; la evaluación usa 25-70 mm)
    etiqueta: política "host" (solo hueso huésped viable = 1; callo, esclerosis, tapón e
    involucro = 1; fragmentos, secuestro, metal y cemento = 0)

El split se conserva: las tibias de validación del fold (y sus defectos) quedan en validación,
así que la validación de nnU-Net sigue siendo sobre tibias no vistas.

Cada caso es reproducible: su semilla depende solo de (SEED, fold, tibia, i).

Uso (en el pod, con nnUNet_raw / nnUNet_preprocessed definidas):
  python make_defect_dataset_v2.py --fold 0 --n 6 --workers 8
Salida:
  $nnUNet_raw/Dataset503_TibiaDefectsF0/{imagesTr,labelsTr,dataset.json,virtual_defects_meta.json}
  $nnUNet_raw/../defect_aux/Dataset503_TibiaDefectsF0/<caso>_aux.nii.gz   (mapa de clases auxiliares, análisis)
  splits_final_Dataset503.json   (copiar a $nnUNet_preprocessed/Dataset503_TibiaDefectsF0/ tras preprocesar)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import virtual_defects_v2 as v2   # noqa: E402

SRC_NAME = "Dataset502_TibiaFischer"
SEED = 20261104
MIX = dict(A=0.3, B=0.3, C=0.3, D=0.1)
MAX_TRIES = 5


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _one_tibia(job):
    case, part, idx, n, fold, src, dst, aux_dir = job
    src, dst, aux_dir = Path(src), Path(dst), Path(aux_dir)
    shutil.copy(src / "imagesTr" / f"{case}_0000.nii.gz", dst / "imagesTr" / f"{case}_0000.nii.gz")
    shutil.copy(src / "labelsTr" / f"{case}.nii.gz", dst / "labelsTr" / f"{case}.nii.gz")
    ct_img = sitk.ReadImage(str(src / "imagesTr" / f"{case}_0000.nii.gz"))
    lb_img = sitk.ReadImage(str(src / "labelsTr" / f"{case}.nii.gz"))
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    lbl = sitk.GetArrayFromImage(lb_img)
    vals = set(np.unique(lbl).tolist())
    if not vals <= {0, 1}:
        raise ValueError(f"{case}: la etiqueta tiene clases {sorted(vals)}; se esperaba solo 0/1")
    sp = np.array(lb_img.GetSpacing()[::-1], float)
    out_ids, meta = [case], {case: dict(scenario="intacta")}
    scen, prob = list(MIX), np.array(list(MIX.values()))
    for i in range(n):
        cid = f"{case}__v2d{i}"
        for t in range(MAX_TRIES):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, fold, idx, i, t]))
            sc = str(rng.choice(scen, p=prob / prob.sum()))
            p = v2.sample_params(sc, rng, 0)
            try:
                r = v2.make_defect_v2(ct, lbl, sp, p, rng, label_policy="host")
                break
            except ValueError as e:          # defecto vacío u otra geometría imposible: se vuelve a sortear
                print(f"[{case}] intento {t} falló: {e}")
        else:
            raise RuntimeError(f"{cid}: {MAX_TRIES} intentos fallidos")
        o = sitk.GetImageFromArray(np.round(r["ct"]).astype(sitk.GetArrayViewFromImage(ct_img).dtype))
        o.CopyInformation(ct_img)
        sitk.WriteImage(o, str(dst / "imagesTr" / f"{cid}_0000.nii.gz"), useCompression=True)
        lo = sitk.GetImageFromArray(r["label"].astype(lbl.dtype)); lo.CopyInformation(lb_img)
        sitk.WriteImage(lo, str(dst / "labelsTr" / f"{cid}.nii.gz"), useCompression=True)
        ao = sitk.GetImageFromArray(r["aux"].astype(np.uint8)); ao.CopyInformation(lb_img)
        sitk.WriteImage(ao, str(aux_dir / f"{cid}_aux.nii.gz"), useCompression=True)
        m = _jsonable(r["meta"]); m["part"] = part; m["attempt"] = t
        meta[cid] = m
        out_ids.append(cid)
        sol = m.get("solomin", {})
        print(f"[{part}] {cid}  {sc}  {m['length_mm']:.0f} mm  {sol.get('code', '')}  implante={m.get('implant')}")
    return part, out_ids, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=6, help="defectos realistas por tibia (además de la intacta)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    a = ap.parse_args()
    raw, pre = Path(a.raw), Path(a.preprocessed)
    src = raw / SRC_NAME
    dst_name = f"Dataset503_TibiaDefectsF{a.fold}"
    dst = raw / dst_name
    aux_dir = raw.parent / "defect_aux" / dst_name
    for d in (dst / "imagesTr", dst / "labelsTr", aux_dir):
        d.mkdir(parents=True, exist_ok=True)
    split = json.load(open(pre / SRC_NAME / "splits_final.json", encoding="utf-8"))[a.fold]
    jobs = [(c, part, idx, a.n, a.fold, str(src), str(dst), str(aux_dir))
            for part in ("train", "val") for idx, c in enumerate(split[part], start=0 if part == "train" else 1000)]
    new_split, meta = {"train": [], "val": []}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for part, ids, m in ex.map(_one_tibia, jobs):
            new_split[part].extend(ids); meta.update(m)
    ds = json.load(open(src / "dataset.json", encoding="utf-8"))
    ds["numTraining"] = len(new_split["train"]) + len(new_split["val"])
    ds["name"] = dst_name
    ds["description"] = "Fischer + defectos realistas v2 (escenarios A/B/C/D), etiqueta = hueso huésped viable"
    json.dump(ds, open(dst / "dataset.json", "w", encoding="utf-8"), indent=2)
    json.dump(meta, open(dst / "virtual_defects_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump([new_split], open("splits_final_Dataset503.json", "w", encoding="utf-8"), indent=2)
    sc = [m["scenario"] for m in meta.values()]
    print(f"\n{dst_name}: train={len(new_split['train'])} val={len(new_split['val'])} ·",
          {s: sc.count(s) for s in sorted(set(sc))})


if __name__ == "__main__":
    main()
