"""
Pasos 4-7 de la verificación independiente de pat_02 (no modifica el pipeline; solo lee).
  4 truncamiento distal (pérdida cortical por corte cerca de D2 y más allá)
  5 articulación (plafón tibial distal, distancia defecto-plafón, figura coronal/sagital)
  6 ft vs base (Dice, origen de la diferencia de volumen de defecto, figura)
  7 aviso de densidad (HU>700 / >2500 dentro del defecto y fracción en perlas)
Uso: .venv_nnunet\\Scripts\\python.exe results\\pathological\\pat_02_manual_eval\\p4_7_analisis.py
Salida: p4_7_analisis.json, pat02_articulacion.png, pat02_ft_vs_base.png, pat02_truncamiento.png
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "src" / "defect"))
import defect_core as dc  # noqa: E402

OUT = R / "results/pathological/pat_02_manual_eval"
RUNS = {"ft": R / "results/pathological/pat_02_ft", "base": R / "results/pathological/pat_02_base_f0"}
CLK = R / "data/interim/pathological/pat_02/clicks"
S3 = np.ones((3, 3, 3))

ct, img = dc.read_image(RUNS["ft"] / "work/roi_0000.nii.gz")
sp = dc.spacing_kji(img); vox = float(np.prod(sp)); a2 = sp[1] * sp[2]
seed = dc.read_image(OUT / "pat_02_semilla_thr400.nii.gz")[0] > 0
beads = dc.read_image(OUT / "pat_02_perlas.nii.gz")[0] > 0
pred = {n: dc.read_image(r / "work/pred/roi.nii.gz")[0] == 1 for n, r in RUNS.items()}
dpipe = {n: dc.read_image(r / "defecto.nii.gz")[0] > 0 for n, r in RUNS.items()}
dseed = dc.read_image(OUT / "defecto_manual.nii.gz")[0] > 0


def clicks(f):
    d = json.loads((CLK / f).read_text(encoding="utf-8"))["markups"][0]
    out = {}
    for p in d["controlPoints"]:
        x = np.array(p["position"], float)
        if d.get("coordinateSystem", "LPS") == "RAS":
            x[:2] *= -1
        out[p["label"]] = np.array(img.TransformPhysicalPointToContinuousIndex(x.tolist()))  # (i,j,k)
    return out


cD, cT = clicks("Defect_run1.mrk.json"), clicks("Tibia_run1.mrk.json")
k1, k2 = sorted(int(round(v[2])) for v in list(cD.values())[:2])
nk = ct.shape[0]
J = {}

# ---------------------------------------------------------------- caracterización (igual que eval_manual_reference)
ext = dc.tibia_extent_k(dc.postprocess_bone(pred["ft"]))
masks = {"semilla": seed, "ft": pred["ft"], "base": pred["base"]}
res, dmask, ref, bonepp, loss = {}, {}, {}, {}, {}
for n, m in masks.items():
    info = {}
    bonepp[n] = dc.postprocess_bone(m)
    r_, d_, rf = dc.characterize(m, sp, k1, k2, tibia_extent=ext, ct=ct)
    res[n], dmask[n], ref[n] = r_, d_, rf
    loss[n] = np.full(nk, np.nan)
    for k in range(k1, k2 + 1):
        loss[n][k] = dc.angular_defect_slice(rf[k], bonepp[n][k], sp[1:])[1]
    p_lo, p_hi, p_sh, p_ok = dc.find_healthy_band(bonepp[n], k1, -1, sp)
    d_lo, d_hi, d_sh, d_ok = dc.find_healthy_band(bonepp[n], k2, +1, sp)
    res[n]["banda_prox_k"] = [int(p_lo), int(p_hi - 1)]; res[n]["banda_dist_k"] = [int(d_lo), int(d_hi - 1)]
    res[n]["area_banda_prox_mm2"] = float(bonepp[n][p_lo:p_hi].sum((1, 2)).mean() * a2)
    res[n]["area_banda_dist_mm2"] = float(bonepp[n][d_lo:d_hi].sum((1, 2)).mean() * a2)
J["coherencia_defecto_pipeline"] = {n: dict(dice_recalculado_vs_defecto_nii=float(2 * (dmask[n] & dpipe[n]).sum() / (dmask[n].sum() + dpipe[n].sum())))
                                    for n in RUNS}
J["bandas_referencia"] = {n: {k: res[n][k] for k in ("banda_prox_k", "banda_dist_k", "area_banda_prox_mm2", "area_banda_dist_mm2",
                                                       "ref_volume_mm3", "k_start", "k_end", "length_mm", "volume_mm3")} for n in masks}

# ---------------------------------------------------------------- 4. truncamiento
def hull_loss(m_raw, m_fill, k):
    """Pérdida de contorno (%) respecto de la envolvente convexa del propio corte (sin referencia interpolada)."""
    if m_fill[k].sum() < 20:
        return np.nan
    h = convex_hull_image(m_fill[k])
    return dc.angular_defect_slice(h, m_raw[k], sp[1:])[1]


raw = {"semilla": seed, "ft": pred["ft"], "base": pred["base"]}
calib = {n: [hull_loss(raw[n], bonepp[n], k) for k in range(450, 551, 5)] for n in raw}
k_end_tib = {n: int(np.nonzero(bonepp[n].any((1, 2)))[0].max()) for n in raw}
ks4 = list(range(k2 - int(round(15 / sp[0])), k2 + 1, 4))
if ks4[-1] != k2:
    ks4.append(k2)
ks4b = list(range(k2 + 4, max(k_end_tib.values()) + 1, 4))
rows = []
for k in ks4 + ks4b:
    row = dict(k=k, mm_desde_D2=round((k - k2) * sp[0], 1))
    for n in raw:
        if k <= k2:
            row[f"perdida_{n}_pct"] = None if np.isnan(loss[n][k]) else round(float(loss[n][k]), 1)
            ra = ref[n][k].sum()
            row[f"frac_area_defecto_{n}"] = round(float(dmask[n][k].sum() / ra), 3) if ra else None
        row[f"perdida_envolvente_{n}_pct"] = None if np.isnan(h := hull_loss(raw[n], bonepp[n], k)) else round(float(h), 1)
        row[f"area_hueso_{n}_mm2"] = round(float(bonepp[n][k].sum() * a2))
    rows.append(row)
J["truncamiento"] = dict(
    k1=k1, k2=k2, nk_roi=nk,
    mm_roi_mas_alla={lab: round(float((nk - 1 - v[2]) * sp[0]), 1) for lab, v in
                     [("Defect_P1(D1)", cD["P1"]), ("Defect_P2(D2)", cD["P2"]), ("Tibia_P1", cT["P1"]), ("Tibia_P2", cT["P2"])]},
    mm_roi_antes={lab: round(float(v[2] * sp[0]), 1) for lab, v in
                  [("Defect_P1(D1)", cD["P1"]), ("Defect_P2(D2)", cD["P2"]), ("Tibia_P1", cT["P1"]), ("Tibia_P2", cT["P2"])]},
    k_click={lab: round(float(v[2]), 2) for lab, v in
             [("Defect_P1", cD["P1"]), ("Defect_P2", cD["P2"]), ("Tibia_P1", cT["P1"]), ("Tibia_P2", cT["P2"])]},
    ultimo_k_con_tibia=k_end_tib,
    mm_tibia_mas_alla_de_D2={n: round((v - k2) * sp[0], 1) for n, v in k_end_tib.items()},
    calibracion_perdida_envolvente_diafisis_sana_k450_550={n: dict(mediana=float(np.nanmedian(v)), p95=float(np.nanpercentile(v, 95)), max=float(np.nanmax(v))) for n, v in calib.items()},
    perfil=rows)

# ---------------------------------------------------------------- 5. plafón
foot = (bonepp["ft"][680] | bonepp["base"][680])
foot = ndi.binary_erosion(foot, iterations=int(round(3 / sp[1])))
jj, ii = np.nonzero(foot)
last = {n: [] for n in ("ft", "base", "semilla")}
jsp = []
for j, i in zip(jj, ii):
    for n in last:
        col = bonepp[n][640:760, j, i]
        last[n].append(640 + int(np.nonzero(col)[0].max()) if col.any() else np.nan)
    prof = ndi.uniform_filter1d(ct[660:730, j, i].astype(float), 3)
    lf = last["ft"][-1]
    if not np.isnan(lf):
        lo, hi = int(lf) - 8 - 660, int(lf) + 12 - 660
        jsp.append(660 + lo + int(np.argmin(prof[lo:hi])))
plaf_k = {n: dict(mediana=float(np.nanmedian(v)), p10=float(np.nanpercentile(v, 10)), p90=float(np.nanpercentile(v, 90))) for n, v in last.items()}
plaf_k["espacio_articular_minHU"] = dict(mediana=float(np.median(jsp)), p10=float(np.percentile(jsp, 10)), p90=float(np.percentile(jsp, 90)))
KP = plaf_k["ft"]["mediana"]   # plafón (superficie articular de la tibia, eje central) = fin distal mediano de la tibia ft en la huella central


def def_end_dist(d):
    """Distancia mínima (mm, a lo largo de k) entre el defecto y el fin distal de la tibia, columna a columna en la huella central."""
    ds = []
    for j, i, lf in zip(jj, ii, last["ft"]):
        col = d[:, j, i]
        if col.any() and not np.isnan(lf):
            ds.append((lf - np.nonzero(col)[0].max()) * sp[0])
    return float(min(ds)) if ds else None


width = max(float((np.ptp(np.nonzero(bonepp["ft"][k])[0]) + 1) * sp[1]) for k in range(670, 691))
dmasks_all = {"semilla": dseed, "ft": dpipe["ft"], "base": dpipe["base"]}
J["articulacion"] = dict(
    huella_central_columnas=int(len(jj)), plafon_k=plaf_k, plafon_k_usado=KP,
    ancho_max_tibia_distal_RL_mm_k670_690=width,
    defecto={n: dict(k_start=int(res[n]["k_start"]), k_end=int(res[n]["k_end"]),
                     dist_kend_a_plafon_mm=round((KP - res[n]["k_end"]) * sp[0], 1),
                     dist_kstart_a_plafon_mm=round((KP - res[n]["k_start"]) * sp[0], 1),
                     dist_min_columna_a_fin_tibia_mm=def_end_dist(dmasks_all[n]),
                     dentro_cuadrado_AO_distal=bool((KP - res[n]["k_start"]) * sp[0] <= width))
             for n in masks},
    k_extremo_articular_tibia_por_modelo=k_end_tib)

# ---------------------------------------------------------------- 6. ft vs base
def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else float("nan")


w = slice(k1, k2 + 1)
dF, dB, rF, rB = dpipe["ft"], dpipe["base"], ref["ft"], ref["base"]
J["ft_vs_base"] = dict(
    dice_hueso_postproc_D1D2=dice(bonepp["ft"][w], bonepp["base"][w]),
    dice_hueso_crudo_D1D2=dice(pred["ft"][w], pred["base"][w]),
    vol_hueso_D1D2_mm3={n: float(bonepp[n][w].sum() * vox) for n in ("ft", "base")},
    hueso_solo_ft_mm3=float((bonepp["ft"][w] & ~bonepp["base"][w]).sum() * vox),
    hueso_solo_base_mm3=float((bonepp["base"][w] & ~bonepp["ft"][w]).sum() * vox),
    dice_referencia=dice(rF, rB), vol_ref_mm3=dict(ft=float(rF.sum() * vox), base=float(rB.sum() * vox)),
    dice_defecto=dice(dF, dB), vol_def_mm3=dict(ft=float(dF.sum() * vox), base=float(dB.sum() * vox)),
    def_ft_y_base_mm3=float((dF & dB).sum() * vox),
    def_ft_fuera_ref_base_mm3=float((dF & ~rB).sum() * vox),
    def_ft_dentro_ref_base_no_def_base_mm3=float((dF & rB & ~dB).sum() * vox),
    def_ft_dentro_ref_base_no_def_base_hueso_base_mm3=float((dF & rB & ~dB & bonepp["base"]).sum() * vox),
    def_base_no_ft_mm3=float((dB & ~dF).sum() * vox),
    def_base_no_ft_hueso_ft_mm3=float((dB & ~dF & bonepp["ft"]).sum() * vox),
    mitad_prox_vs_dist={n: dict(prox=float(d_[k1:(k1 + k2) // 2].sum() * vox), dist=float(d_[(k1 + k2) // 2:k2 + 1].sum() * vox))
                        for n, d_ in (("ft", dF), ("base", dB))},
)

# ---- 6b. conectividad del defecto ft (origen de las 2 componentes sólidas del andamio)
conn = {}
for n in ("ft", "base"):
    conn[n] = {}
    for nm, st in (("26", S3), ("18", ndi.generate_binary_structure(3, 2)), ("6", None)):
        lab, nl = ndi.label(dpipe[n], structure=st)
        sz = np.bincount(lab.ravel())[1:]
        order = np.argsort(sz)[::-1][:3]
        conn[n][nm] = dict(n=int(nl), mayores=[dict(vol_mm3=round(float(sz[c] * vox), 1),
                                                    centro_kji=[round(float(x), 1) for x in ndi.center_of_mass(lab == c + 1)],
                                                    k=[int(x) for x in np.nonzero((lab == c + 1).any((1, 2)))[0][[0, -1]]],
                                                    HU_mediana=float(np.median(ct[lab == c + 1])))
                                               for c in order])
J["conectividad_defecto"] = conn
p6b = OUT / "p6b_andamio_componentes.json"
if p6b.exists():
    J["andamio_ft_componentes_malla"] = json.loads(p6b.read_text(encoding="utf-8"))["componentes"][:3]

# ---------------------------------------------------------------- 7. densidad
bd1 = ndi.binary_dilation(beads, S3, 1)
dist_b = ndi.distance_transform_edt(~beads, sampling=sp)
b2mm = dist_b <= 2.0
dens = {}
for n in ("ft", "base"):
    d_ = dpipe[n]; nt = d_.sum(); c = ct[d_]
    near = bonepp[n] | (ct >= dc.METAL_HU)
    sl = (slice(k1 - 10, k2 + 11), slice(None), slice(None))
    dist = np.full(ct.shape, np.inf); dist[sl] = ndi.distance_transform_edt(~near[sl], sampling=sp)
    qc = d_ & (ct >= dc.DENSE_HU) & (ct < dc.METAL_HU) & (dist > dc.DENSE_EXCL_MM)
    lab, nl = ndi.label(qc & ~b2mm, structure=S3)
    sz = np.sort(np.bincount(lab.ravel())[1:])[::-1] * vox
    dens[n] = dict(
        vox_defecto=int(nt), vol_defecto_mm3=float(nt * vox),
        frac_HU_gt700_pct=float(100 * (c > 700).mean()), frac_HU_ge2500_pct=float(100 * (c >= 2500).mean()),
        vol_HU_gt700_mm3=float((c > 700).sum() * vox), vol_HU_ge2500_mm3=float((c >= 2500).sum() * vox),
        HUgt700_en_perlas_pct=float(100 * (d_ & (ct > 700) & beads).sum() / max((d_ & (ct > 700)).sum(), 1)),
        HUgt700_en_perlas_dil1vox_pct=float(100 * (d_ & (ct > 700) & bd1).sum() / max((d_ & (ct > 700)).sum(), 1)),
        HUgt700_a_2mm_de_perlas_pct=float(100 * (d_ & (ct > 700) & b2mm).sum() / max((d_ & (ct > 700)).sum(), 1)),
        HUge2500_en_perlas_pct=float(100 * (d_ & (ct >= 2500) & beads).sum() / max((d_ & (ct >= 2500)).sum(), 1)),
        perlas_dentro_del_defecto_pct_vol_perlas=float(100 * (beads & d_).sum() / beads.sum()),
        qc_denso_recalc_pct=float(100 * qc.sum() / nt), qc_denso_recalc_mm3=float(qc.sum() * vox),
        qc_denso_a_2mm_de_perlas_pct=float(100 * (qc & b2mm).sum() / max(qc.sum(), 1)),
        qc_denso_sin_perlas_pct_del_defecto=float(100 * (qc & ~b2mm).sum() / nt),
        qc_denso_sin_perlas_componentes=int(nl), qc_denso_sin_perlas_mayores_mm3=[round(float(x), 1) for x in sz[:5]],
        qc_denso_sin_perlas_en_semilla_pct=float(100 * (qc & ~b2mm & seed).sum() / max((qc & ~b2mm).sum(), 1)),
        reporte_dense_frac_pct=float(json.loads((RUNS[n] / "reporte.json").read_text(encoding="utf-8"))["defecto"]["dense_frac_pct"]))
J["densidad"] = dens

(OUT / "p4_7_analisis.json").write_text(json.dumps(J, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

# ================================================================ figuras
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COL = {"semilla": "yellow", "ft": "lime", "base": "deepskyblue"}

# ---- articulación: coronal (i fijo: eje k x j, R-L) y sagital (j fijo: eje k x i, A-P), reformateados al eje de la ROI
kc0, kc1 = 520, min(nk, 800)
cj, ci = ndi.center_of_mass(bonepp["ft"][640:690].any(0))
cj, ci = int(round(cj)), int(round(ci))
fig, ax = plt.subplots(1, 2, figsize=(14, 11))
for a, (tit, axis, idx, dx) in zip(ax, [(f"coronal (reformateado al eje de la ROI), i={ci}", 2, ci, sp[1]),
                                        (f"sagital (reformateado al eje de la ROI), j={cj}", 1, idx_ := cj, sp[2])]):
    take = (lambda v, s=idx, ax_=axis: np.take(v, s, axis=ax_)[kc0:kc1])
    slab = (lambda v, s=idx, ax_=axis: np.take(v, range(s - 4, s + 5), axis=ax_).any(axis=ax_)[kc0:kc1])
    im = take(ct)
    extent = [0, im.shape[1] * dx, kc1 * sp[0], kc0 * sp[0]]
    a.imshow(im, cmap="gray", vmin=-200, vmax=1400, extent=extent)
    X = np.arange(im.shape[1]) * dx + dx / 2; Y = np.arange(kc0, kc1) * sp[0]
    for n in ("semilla", "ft", "base"):
        a.contour(X, Y, take(bonepp[n]), [0.5], colors=COL[n], linewidths=0.8)
        a.contour(X, Y, slab(dmasks_all[n]), [0.5], colors=COL[n], linewidths=2.0, linestyles="--")
    bz = slab(beads)
    yy, xx = np.nonzero(bz); a.plot(xx * dx, (yy + kc0) * sp[0], "r.", ms=2)
    for lab, v, mk in [("D1", cD["P1"], "o"), ("D2", cD["P2"], "o"), ("T1", cT["P1"], "s"), ("T2", cT["P2"], "s")]:
        lat = v[1] if axis == 2 else v[0]
        if kc0 <= v[2] < kc1:
            a.plot(lat * dx, v[2] * sp[0], mk, mfc="none", mec="magenta", ms=10, mew=2)
            a.text(lat * dx + 2, v[2] * sp[0], lab, color="magenta", fontsize=11, weight="bold")
    for kk_, c_, l_ in [(k1, "orange", "D1"), (k2, "orange", "D2"), (KP, "red", "plafón")]:
        a.axhline(kk_ * sp[0], color=c_, ls=":", lw=1)
        a.text(1, kk_ * sp[0] - 1, l_, color=c_, fontsize=10)
    a.set_title(tit); a.set_xlabel("mm"); a.set_ylabel("posición a lo largo del eje de la ROI (mm)")
leg = [Line2D([], [], color=COL[n], lw=1, label=f"tibia {n}") for n in COL] + \
      [Line2D([], [], color=COL[n], lw=2, ls="--", label=f"defecto {n} (proyección ±4 vóx)") for n in COL] + \
      [Line2D([], [], color="r", marker=".", ls="", label="perlas (≥1700 HU con núcleo ≥2500)"),
       Line2D([], [], color="magenta", marker="o", mfc="none", ls="", label="clics Defect (D1,D2) / Tibia (T1,T2, cuadrado)")]
ax[1].legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.8)
fig.suptitle(f"pat_02 – defecto vs plafón tibial (plafón k≈{KP:.0f}; ROI ya alineada al eje Tibia P1–P2; ventana k {kc0}–{kc1 - 1})")
plt.tight_layout(); plt.savefig(OUT / "pat02_articulacion.png", dpi=120); plt.close()

# ---- truncamiento: perfiles
kk = np.arange(k1 - 10, min(nk, max(k_end_tib.values()) + 5))
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for n in raw:
    ax[0].plot((kk - k2) * sp[0], loss[n][kk], color={"semilla": "k", "ft": "C2", "base": "C0"}[n], label=f"{n} (ref. interpolada, ventana D1–D2)")
    ax[0].plot((kk - k2) * sp[0], [hull_loss(raw[n], bonepp[n], k) for k in kk], ls=":", color={"semilla": "k", "ft": "C2", "base": "C0"}[n], label=f"{n} (vs envolvente convexa del corte)")
    ax[1].plot((kk - k2) * sp[0], bonepp[n][kk].sum((1, 2)) * a2, color={"semilla": "k", "ft": "C2", "base": "C0"}[n], label=n)
for a in ax:
    a.axvline(0, color="orange", ls="--"); a.axvline((KP - k2) * sp[0], color="r", ls="--")
ax[0].axhline(50, color="r", ls=":"); ax[0].axhline(10, color="0.5", ls=":")
ax[0].set_ylabel("pérdida cortical (%)"); ax[1].set_ylabel("área ósea rellena (mm²)"); ax[1].set_xlabel("mm desde D2 (naranja); rojo = plafón")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "pat02_truncamiento.png", dpi=110); plt.close()

# ---- ft vs base
kk = np.arange(k1, k2 + 1)
diff = np.abs(dF[kk].sum((1, 2)).astype(int) - dB[kk].sum((1, 2)).astype(int))
sel = sorted(kk[np.argsort(diff)[::-1]][:1].tolist() + [k1 + 20, (k1 + k2) // 2, k2 - 15, k2])
sel = sorted(set(sel))[:5]
fig = plt.figure(figsize=(20, 10))
a0 = fig.add_subplot(2, 1, 1)
for n, d_, r_, c in (("ft", dF, rF, "C2"), ("base", dB, rB, "C0")):
    a0.plot(kk * sp[0], r_[kk].sum((1, 2)) * a2, c, ls="--", label=f"referencia {n}")
    a0.plot(kk * sp[0], d_[kk].sum((1, 2)) * a2, c, lw=2, label=f"defecto {n}")
    a0.plot(kk * sp[0], bonepp[n][kk].sum((1, 2)) * a2, c, ls=":", label=f"hueso {n}")
for k in sel:
    a0.axvline(k * sp[0], color="0.6", lw=0.6)
a0.set_ylabel("área (mm²)"); a0.set_xlabel("posición en la ROI (mm)"); a0.legend(ncol=3, fontsize=8)
a0.set_title("pat_02 ft vs base: área por corte de referencia interpolada, defecto y hueso (ventana D1–D2)")
for t, k in enumerate(sel):
    a = fig.add_subplot(2, len(sel), len(sel) + t + 1)
    a.imshow(ct[k], cmap="gray", vmin=-200, vmax=1400)
    for n, d_, r_, c in (("ft", dF, rF, "lime"), ("base", dB, rB, "deepskyblue")):
        if bonepp[n][k].any(): a.contour(bonepp[n][k], [0.5], colors=c, linewidths=0.8)
        if r_[k].any(): a.contour(r_[k], [0.5], colors=c, linewidths=1.2, linestyles="--")
        if d_[k].any(): a.contourf(d_[k], [0.5, 1.5], colors=c, alpha=0.25)
    if beads[k].any(): a.contour(beads[k], [0.5], colors="r", linewidths=0.8)
    a.set_title(f"k={k} ({(k - k2) * sp[0]:+.0f} mm de D2)\ndef ft {dF[k].sum() * a2:.0f} / base {dB[k].sum() * a2:.0f} mm²", fontsize=9)
    a.axis("off")
fig.text(0.01, 0.01, "verde = ft, azul = base; línea fina = hueso, discontinua = referencia, relleno = defecto; rojo = perlas", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "pat02_ft_vs_base.png", dpi=100); plt.close()
print(json.dumps(J, indent=1, ensure_ascii=False, default=float))
