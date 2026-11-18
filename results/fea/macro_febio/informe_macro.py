"""Une la escala 2 (corridas_macro.json) con la celda (celda_*_tensiones.json), escribe CSV/JSON y figuras.
Todo escala linealmente desde la corrida de referencia. Uso: informe_macro.py [datos|figuras]"""
import csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "celda_febio"))
import macro_febio as mf
import scaffold_f4 as f4

H_FINA, H_MEDIA = 0.5, 0.7
POROSIDAD_PCT = {"C_Ti": 62.6345844991183, "C_PCL": 62.83949845339928, "P2_Ti": 62.62, "P2_PCL": 63.88}   # reportes vigentes (P2: comparacion_materiales_v2)
TAG_CELDA = {"C_Ti": "Ti", "C_PCL": "PCL", "P2_Ti": "Ti", "P2_PCL": "PCL"}
MAT = {"Ti": "Ti6Al4V_LPBF", "PCL": "PCL_bTCP_extrusion"}
EPS_LINEAL = 0.01      # hasta aquí se verificó la celda


def cargar():
    corr = json.load(open(HERE / "corridas_macro.json", encoding="utf-8"))
    return {(r["caso"], r["h_mm"], r["banda"]): r for r in corr if r["ok"]}


def seccion_min_central(c):
    """Menor sección en el 50 % central del largo: A_mid no representa una envolvente con secciones muy variables."""
    z = np.load(HERE / "trabajo" / f"{c}_h{H_FINA:g}_b{mf.BANDA_BASE:g}" / "campo.npz")
    S, hs = z["S"], z["hs"]
    a = S.sum(axis=(0, 1)) * hs[0] * hs[1]
    ks = np.nonzero(a)[0]
    n = len(ks)
    return float(a[ks[n // 4:3 * n // 4]].min())


def etapas(tag):
    m = f4.MATERIALS[MAT[tag]]
    if "etapas_degradacion" in m and tag != "Ti":
        return [(e["etapa"], e["factor_E"], e["factor_fluencia"]) for e in m["etapas_degradacion"]]
    return [("t0", 1.0, 1.0)]


def fila(r, F, nombre_carga, etapa, fE, fY, celda, e_rel=1.0):
    """fE: factor sobre E_eff por etapa; e_rel: factor de sensibilidad (E_inf/E)."""
    c = r["caso"]
    tag = TAG_CELDA[c]
    fe = fE * e_rel
    sF = F / abs(r["R_ref_N"][2])
    Y0 = f4.MATERIALS[MAT[tag]]["fluencia_MPa"]
    Y = None if fY is None else Y0 * fY
    eps_med = r["Ez_med_ref"] * sF / fe
    eps_p95 = r["Ez_p95_ref"] * sF / fe
    out = dict(caso=c, material=tag, carga=nombre_carga, F_N=F, etapa=etapa, h_mm=r["h_mm"], E_eff_MPa=round(r["E_MPa"] * fe, 3), nu=r["nu"], A_mid_mm2=round(r["A_mid_mm2"], 1),
               sigma_nom_MPa=round(F / r["A_mid_mm2"], 3), u_max_mm=round(r["u_max_ref_mm"] * sF / fe, 4), eps_ap_media=round(eps_med, 6), eps_ap_p95=round(eps_p95, 6),
               vm_macro_media_MPa=round(r["vm_med_ref"] * sF, 3), vm_macro_p95_MPa=round(r["vm_p95_ref"] * sF, 3), vm_macro_max_MPa=round(r["vm_max_ref"] * sF, 3))
    # En la celda, sigma_local/eps es proporcional al E del sólido: con la etapa, eps sube y E_s baja por el mismo factor.
    # La sensibilidad e_rel mantiene E_s y reduce eps.
    k = {x: v * fE for x, v in celda["vm_sobre_eps_bloque"].items()}
    kn = {x: v * fE for x, v in celda["vm_sobre_eps_nucleo"].items()}
    out.update(vm_solido_media_MPa=round(eps_med * k["media"], 2), vm_solido_p95_MPa=round(eps_med * k["p95"], 2), vm_solido_p99_MPa=round(eps_med * k["p99"], 2), vm_solido_p99_nucleo_MPa=round(eps_med * kn["p99"], 2),
               vm_solido_p99_eps_p95_MPa=round(eps_p95 * k["p99"], 2), fluencia_MPa=None if Y is None else round(Y, 2))
    P = POROSIDAD_PCT[c] / 100
    Amin = seccion_min_central(c)
    out["sigma_s_cap3_MPa"] = round(F / r["A_mid_mm2"] / (1 - P), 2)
    out.update(A_min_central_mm2=round(Amin, 1), sigma_nom_Amin_MPa=round(F / Amin, 3), sigma_s_cap3_Amin_MPa=round(F / Amin / (1 - P), 2))
    if Y is None:
        out.update(FS_p99=None, FS_p99_nucleo=None, FS_p99_eps_p95=None, FS_cap3=None)
    else:
        out.update(FS_p99=round(Y / (eps_med * k["p99"]), 3), FS_p99_nucleo=round(Y / (eps_med * kn["p99"]), 3), FS_p99_eps_p95=round(Y / (eps_p95 * k["p99"]), 3), FS_cap3=round(Y / out["sigma_s_cap3_MPa"], 3))
    out["fuera_rango_lineal"] = bool(eps_p95 > EPS_LINEAL)
    return out


def datos():
    C = cargar()
    celdas = {t: json.load(open(HERE / f"celda_{t}_tensiones.json", encoding="utf-8")) for t in set(TAG_CELDA.values())}
    filas, sens = [], []
    for c in mf.CASOS:
        r = C[(c, H_FINA, mf.BANDA_BASE)]
        tag = TAG_CELDA[c]
        for nom, F in mf.CARGAS.items():
            for et, fE, fY in etapas(tag):
                filas.append(fila(r, F, nom, et, fE, fY, celdas[tag]))
            pr = mf.celda_props(mf.CASOS[c]["celda"])
            sens.append(fila(r, F, nom, "t0_E_inf", 1.0, 1.0, celdas[tag], pr["E_inf"] / pr["E"]))
    conv = []
    for c in mf.CASOS:
        rr = [C.get((c, h, mf.BANDA_BASE)) for h in mf.TAMANOS]
        d = dict(caso=c)
        for h, r in zip(mf.TAMANOS, rr):
            d[f"n_elem_h{h}"] = r["n_elementos"]; d[f"u_max_ref_h{h}"] = round(r["u_max_ref_mm"], 5); d[f"vm_p95_ref_h{h}"] = round(r["vm_p95_ref"], 4)
            d[f"eps_med_ref_h{h}"] = round(r["Ez_med_ref"], 7); d[f"t_s_h{h}"] = r["t_s"]; d[f"ram_gb_h{h}"] = r["ram_commit_pico_gb"]
        a, b = rr[1], rr[2]
        d.update(cambio_u_pct=round(100 * (b["u_max_ref_mm"] / a["u_max_ref_mm"] - 1), 2), cambio_vm_p95_pct=round(100 * (b["vm_p95_ref"] / a["vm_p95_ref"] - 1), 2),
                 cambio_eps_pct=round(100 * (b["Ez_med_ref"] / a["Ez_med_ref"] - 1), 2))
        s = C.get((c, H_MEDIA, mf.BANDA_SENS))
        if s:
            d.update(sens_banda05_cambio_u_pct=round(100 * (s["u_max_ref_mm"] / s["F_ref_N"] / (a["u_max_ref_mm"] / a["F_ref_N"]) - 1), 2),
                     sens_banda05_cambio_vm_p95_pct=round(100 * (s["vm_p95_ref"] / s["F_ref_N"] / (a["vm_p95_ref"] / a["F_ref_N"]) - 1), 2),
                     A_prox_banda05=round(s["A_prox_mm2"], 1), A_prox_base=round(a["A_prox_mm2"], 1), A_dist_banda05=round(s["A_dist_mm2"], 1), A_dist_base=round(a["A_dist_mm2"], 1))
        conv.append(d)
    for nom, rows in (("resultados_macro", filas), ("sensibilidad_E_inf", sens), ("convergencia_macro", conv)):
        cols = list(dict.fromkeys(k for d in rows for k in d))
        with open(HERE / f"{nom}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    json.dump(dict(resultados=filas, sensibilidad_E_inf=sens, convergencia=conv, celdas=celdas), open(HERE / "resultados_macro.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return filas, sens, conv


def figuras():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    base = matplotlib.colormaps["Blues"]
    cmap = LinearSegmentedColormap.from_list("azul", base(np.linspace(0.25, 1, 256)))
    (HERE / "figuras").mkdir(exist_ok=True)
    F = mf.CARGAS["conservadora_3366N"]
    nombres = {"C_Ti": "Caso C, Ti6Al4V", "C_PCL": "Caso C, PCL/β-TCP", "P2_Ti": "pat_02, Ti6Al4V", "P2_PCL": "pat_02, PCL/β-TCP"}
    for c in mf.CASOS:
        z = np.load(HERE / "trabajo" / f"{c}_h{H_FINA:g}_b{mf.BANDA_BASE:g}" / "campo.npz")
        S, vm, hs = z["S"], z["vm"], z["hs"]
        prox, lr = int(z["prox"]), int(z["lr_col"])
        V = np.full(S.shape, np.nan, np.float32)
        V[S] = vm * (F / float(z["F_ref"]))
        vmax = float(np.percentile(V[S], 99))
        idx = np.argwhere(S).mean(axis=0).astype(int)
        # Columnas de la dirección: 0 = i, 1 = j. Ejes de S: x = j, y = i.
        eje_lr = 0 if lr == 1 else 1
        planos = [("Sagital", eje_lr), ("Coronal", 1 - eje_lr)]
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 5.2), dpi=200, constrained_layout=True)
        for a, (nom, eje) in zip(ax, planos):
            sl = [slice(None)] * 3; sl[eje] = int(idx[eje])
            P = V[tuple(sl)]                                  # (otro eje, z)
            otro = 1 - eje
            ext = [0, P.shape[0] * hs[otro], 0, P.shape[1] * hs[2]]
            im = a.imshow(P.T if prox > 0 else P.T[::-1], origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            a.set_title(nom, fontsize=10, loc="left")
            a.set_xlabel("mm", fontsize=8); a.set_ylabel("mm (proximal arriba)", fontsize=8)
            a.tick_params(labelsize=7, length=2)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, extend="max")
        cb.set_label("Von Mises macro (MPa), escala hasta el p99", fontsize=8); cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"{nombres[c]}: andamio homogenizado con 3366 N", fontsize=11, x=0.02, ha="left")
        fig.savefig(HERE / "figuras" / f"von_mises_{c}.png", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else "datos"
    if op == "datos":
        fl, se, co = datos()
        for d in co:
            print({k: d[k] for k in d if k.startswith(("caso", "cambio", "banda"))})
    else:
        figuras()
