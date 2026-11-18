"""
Relleno de la metáfisis distal de pat_02 a partir de los anillos corticales pintados manualmente en 3D Slicer.
Entrada: data/interim/pathological/pat_02/manual/pat_02_perlas.nii.gz-Tibia-label.nrrd (export de Slicer, segmento Tibia).
Método: (1) en cada corte clave (anillo cerrado pintado) se rellenan los huecos 2D; (2) entre cortes clave consecutivos
se interpola la forma rellena por distancia con signo (lineal) y se une a la máscara existente. Sin extrapolación.
Variante principal: cortes clave 652-684 (interpolación solo en ese tramo). Variante de sensibilidad: 652-689.
Salidas: pat_02_manual_final.nii.gz (+ .json), pat_02_manual_final_hasta689.nii.gz, pat02_relleno.png
"""
import json
from pathlib import Path
import numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
R = Path(__file__).resolve().parents[4]; OUT = R / "results/pathological/pat_02_manual_eval/final"
SRC = R / "data/interim/pathological/pat_02/manual/pat_02_perlas.nii.gz-Tibia-label.nrrd"
ref = sitk.ReadImage(str(R / "results/pathological/pat_02_ft/work/roi_0000.nii.gz"))
img = sitk.ReadImage(str(SRC))
assert img.GetSize() == ref.GetSize() and np.allclose(img.GetOrigin(), ref.GetOrigin(), atol=1e-2) \
    and np.allclose(img.GetSpacing(), ref.GetSpacing(), atol=1e-4) and np.allclose(img.GetDirection(), ref.GetDirection(), atol=1e-4)
g = sitk.GetArrayFromImage(img) > 0
ct = sitk.GetArrayFromImage(ref)
sp = np.array(ref.GetSpacing()[::-1]); a2 = sp[1] * sp[2]


def sdf(m):
    return ndi.distance_transform_edt(~m, sampling=sp[1:]) - ndi.distance_transform_edt(m, sampling=sp[1:])


def build(keys):
    out = g.copy(); info = []
    F = {k: ndi.binary_fill_holes(g[k]) for k in keys}
    for k in keys:
        out[k] |= F[k]
    for ka, kb in zip(keys[:-1], keys[1:]):
        sa, sb = sdf(F[ka]), sdf(F[kb])
        for k in range(ka + 1, kb):
            t = (k - ka) / (kb - ka)
            new = (((1 - t) * sa + t * sb) <= 0) & ~out[k]
            out[k] |= new
            info.append(dict(k=k, agregado_mm2=round(float(new.sum() * a2), 1)))
    return out, info


res = {}
for tag, keys in (("final", [652, 660, 668, 676, 684]), ("final_hasta689", [652, 660, 668, 676, 684, 689])):
    m, info = build(keys)
    o = sitk.GetImageFromArray(m.astype(np.uint8)); o.CopyInformation(ref)
    name = "pat_02_manual_final.nii.gz" if tag == "final" else "pat_02_manual_final_hasta689.nii.gz"
    sitk.WriteImage(o, str(OUT / name), useCompression=True)
    res[tag] = dict(archivo=name, cortes_clave=keys, vol_mm3=float(m.sum() * sp.prod()), vol_agregado_mm3=float((m & ~g).sum() * sp.prod()),
                    agregado_por_corte=info)
fill_info = {k: dict(area_anillo_mm2=round(float(g[k].sum() * a2)), area_rellena_mm2=round(float(ndi.binary_fill_holes(g[k]).sum() * a2)),
                     cerrado=bool((ndi.binary_fill_holes(g[k]) & ~g[k]).sum() * a2 > 100)) for k in (645, 652, 660, 668, 676, 684, 689)}
mfin = sitk.GetArrayFromImage(sitk.ReadImage(str(OUT / "pat_02_manual_final.nii.gz"))) > 0
tramos = {k: dict(area_mm2=round(float(mfin[k].sum() * a2)), area_rellena2D_mm2=round(float(ndi.binary_fill_holes(mfin[k]).sum() * a2)))
          for k in list(range(645, 652)) + list(range(685, 690))}
J = dict(entrada=str(SRC), metodo=__doc__.strip(), vol_entrada_mm3=float(g.sum() * sp.prod()), anillos=fill_info, variantes=res,
         areas_fuera_de_interpolacion=tramos)
(OUT / "pat_02_manual_final.json").write_text(json.dumps(J, indent=1, ensure_ascii=False), encoding="utf-8")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ks = [645, 652, 656, 660, 664, 668, 672, 676, 680, 684, 687, 689]
fig, ax = plt.subplots(2, 6, figsize=(24, 9)); ax = ax.ravel()
for a, k in zip(ax, ks):
    a.imshow(ct[k], cmap="gray", vmin=-200, vmax=1400)
    if g[k].any(): a.contour(g[k], [0.5], colors="yellow", linewidths=0.7)
    if (mfin[k] & ~g[k]).any(): a.contourf((mfin[k] & ~g[k]).astype(float), [0.5, 1.5], colors="lime", alpha=0.35)
    a.set_title(f"k={k}" + (" (clave)" if k in (652, 660, 668, 676, 684) else "")); a.axis("off")
fig.suptitle("pat_02: máscara manual (amarillo) + relleno 2D de anillos e interpolación por forma (verde), k 652–684")
plt.tight_layout(); plt.savefig(OUT / "pat02_relleno.png", dpi=90)
print(json.dumps({k: v for k, v in J.items() if k != "metodo"}, indent=1, ensure_ascii=False)[:4000])
