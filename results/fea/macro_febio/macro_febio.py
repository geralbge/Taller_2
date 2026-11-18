"""Escala 2 del FEA: andamio homogenizado (A) y esfuerzo local en el sólido por la celda (B), con FEBio 4.

Verificación computacional, sin valor clínico. Sin fijación, plantillas, hueso ni biología.

A) Envolvente del andamio, HEX8 por vóxeles, material isótropo con E_eff y nu_eff de la escala 1.
   La envolvente se reconstruye igual que en scaffold_f4.build_field (SDF suavizada y remuestreada), con el canal
   del clavo cuando el andamio lo tiene. Eje z de la malla = eje largo k de la ROI orientada.
   Cara distal: las caras expuestas del 25 % distal del largo cuya normal mira a distal; empotradas (ux=uy=uz=0).
   Cara proximal: ídem en el 25 % proximal; presión axial uniforme con resultante F (dirección fija, z).
   Se resuelve con una carga de referencia (eps ~ 0,001 %) y se escala linealmente: el análisis definido es lineal
   y la cinemática finita de FEBio no debe entrar (con PCL a 3366 N la deformación nominal sería del orden de 20 %).
   Con E uniforme, tensiones ~ F y desplazamientos ~ F/E, así que una corrida por geometría y nu cubre cargas y etapas.

B) Celda 4x4x4 (~3 vóxeles por pared) a eps = 0,1 %, con tensiones elementales; Von Mises/eps se escala a la
   deformación aparente de cada caso.

Uso (python de .venv):
  macro_febio.py caso <C_Ti|C_PCL|P2_Ti|P2_PCL> <h_mm> [banda]   una corrida
  macro_febio.py celdas                                          corridas de la celda con tensiones (B)
  macro_febio.py tabla                                           A completo, reanudable
  macro_febio.py informe                                         une A y B, escribe CSV/JSON
  macro_febio.py figuras                                         PNG de Von Mises
"""
import csv, json, shutil, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[3]  # raíz del repositorio
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "scaffold"))
sys.path.insert(0, str(HERE.parent / "celda_febio"))
import scaffold_core as sc
import scaffold_f4 as f4
import celda_febio as cf

WORK = HERE / "trabajo"
FIG = HERE / "figuras"
RES = ROOT / "results"
DEMO = RES / "evaluation/scaffold/demostrativos_v2/tipo_C_fischer_z049_L__realC1"
P2 = RES / "pathological/pat_02_manual_eval/final/defecto_manual.nii.gz"

CARGAS = {"moderada_1432N": 1432.0, "conservadora_3366N": 3366.0}
EPS_REF = 1e-5     # carga de referencia: con cuellos delgados (pat_02) 5e-4 ya invertía elementos
TAMANOS = (1.0, 0.7, 0.5)
BANDA_BASE = 0.25  # fracción del largo, desde cada extremo, donde se buscan las caras de carga y de empotramiento
BANDA_SENS = 0.5
R_APERTURA_MM = 1.25   # apertura morfológica: rasgos < 2,5 mm (una celda de Ti) no admiten homogenización y dejan el sistema casi singular
SOLVER = "cg_ichol"   # si falla, se reintenta con AMG; en pat_02 a 1 mm PARDISO, ichol y AMG coinciden a 1e-10

# Escala 1 (results/fea/celda_febio/resumen_celda.md). Andamios vigentes: comparacion_materiales_v2/pat_02_ref_{ti,pcl}
# (2,5/0,30 sin canal de clavo y 4,2/0,50) y demostrativos_v2. pat_02_material/ es antiguo y no se usa.
CELDA = {"Ti": dict(E=16154.0, nu=0.298, E_inf=16949.0), "PCL": dict(E=51.42, nu=0.329, E_inf=53.86)}
CASOS = {
    "C_Ti": dict(mask=DEMO / "defecto_estimado.nii.gz", celda="Ti", canal=None),
    "C_PCL": dict(mask=DEMO / "defecto_estimado.nii.gz", celda="PCL", canal=None),
    "P2_Ti": dict(mask=P2, celda="Ti", canal=None),
    "P2_PCL": dict(mask=P2, celda="PCL", canal=None),
}


def celda_props(celda):
    return CELDA[celda]


# ------------------------------------------------------------------ geometría
def envolvente(caso, h):
    """Máscara (k, j, i), paso por eje (mm) y signo del eje k hacia proximal, leído de la orientación del NIfTI."""
    img = sitk.ReadImage(str(CASOS[caso]["mask"]))
    defect = sitk.GetArrayFromImage(img) > 0
    sp = np.array(img.GetSpacing()[::-1], float)
    # SimpleITK usa LPS: componente +z = superior. Tercera columna de la dirección = eje del índice k.
    d = np.array(img.GetDirection()).reshape(3, 3)
    kz = d[2, 2]
    if abs(kz) < 0.8:
        raise RuntimeError(f"el eje k no es cráneo-caudal (dir z = {kz:.3f}): revisar la ROI")
    prox = 1 if kz > 0 else -1
    lo, hi = sc._crop_box(defect, sp, 1.0)
    crop = defect[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sdf = ndi.gaussian_filter(sc.signed_distance(crop, sp), 0.5)
    env, h_axes = sc._resample_to_fine(sdf, sp, h)
    del sdf
    info = dict(dir_k=[round(float(x), 4) for x in d[:, 2]], lr_col=int(np.argmax(np.abs(d[0, :]))), ap_col=int(np.argmax(np.abs(d[1, :]))))
    if CASOS[caso]["canal"]:
        rep = json.load(open(CASOS[caso]["canal"], encoding="utf-8"))
        cl = rep["f4"]["clavo"]
        F = dict(f=env.copy(), env=env, h_axes=h_axes, origin_idx=np.array(lo, float))
        info["canal"] = f4.apply_channel(F, sp, cl, rep["f4"]["canal_clavo"]["holgura_mm"], None)
    S0 = env <= 0
    # Apertura con esfera de radio R, en mm y sobre EDT: no depende del tamaño de vóxel.
    d_in = ndi.distance_transform_edt(S0, sampling=h_axes)
    nucleo = d_in > R_APERTURA_MM
    S = (ndi.distance_transform_edt(~nucleo, sampling=h_axes) <= R_APERTURA_MM) & S0
    info["V_antes_apertura_mm3"] = float(S0.sum() * np.prod(h_axes))
    lab, n = ndi.label(S)
    tam = np.bincount(lab.ravel()); tam[0] = 0
    keep = lab == tam.argmax()
    info.update(componentes_eliminadas=int(n - 1), vox_eliminados=int(S.sum() - keep.sum()), h_axes=[float(x) for x in h_axes])
    return keep, np.asarray(h_axes, float), prox, info


def malla_macro(S, hk, banda, prox):
    """S ya en (x, y, z) = (j, i, k), permutación cíclica de (k, j, i): conserva la orientación del espacio de índices.
    Devuelve nodos (ijk), conectividad y las caras de carga y de empotramiento."""
    ijk, conn = cf.malla(S, hk)
    nx = np.array(S.shape, np.int64) + 1
    code = lambda a, b, c: a * nx[1] * nx[2] + b * nx[2] + c
    ord_ = code(ijk[:, 0], ijk[:, 1], ijk[:, 2])
    srt = np.argsort(ord_); un = ord_[srt]
    nid = lambda a, b, c: srt[np.searchsorted(un, code(a, b, c))] + 1
    zs = np.nonzero(S.any(axis=(0, 1)))[0]
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    L = z1 - z0
    pad = np.pad(S, 1)
    arriba = pad[1:-1, 1:-1, 1:-1] & ~pad[1:-1, 1:-1, 2:]     # normal +z
    abajo = pad[1:-1, 1:-1, 1:-1] & ~pad[1:-1, 1:-1, :-2]     # normal -z
    def caras(m, sup):
        a, b, c = np.nonzero(m)
        k = c + 1 if sup else c
        q = [nid(a, b, k), nid(a + 1, b, k), nid(a + 1, b + 1, k), nid(a, b + 1, k)]
        return np.stack(q if sup else q[::-1], axis=1)
    idx = np.arange(S.shape[2])
    alto = idx >= z1 - banda * L
    bajo = idx < z0 + banda * L
    sup = np.zeros_like(S); sup[:, :, alto] = arriba[:, :, alto]
    inf = np.zeros_like(S); inf[:, :, bajo] = abajo[:, :, bajo]
    fsup, finf = caras(sup, True), caras(inf, False)
    return ijk, conn, (fsup, finf) if prox > 0 else (finf, fsup)


def escribir(path, ijk, conn, fp, fd, hs, E, nu, p, prox, solver="cg_ichol", max_iter=4000):
    fix = np.unique(fd)
    with open(path, "w", buffering=2**20) as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n<febio_spec version="4.0">\n<Module type="solid"/>\n<Control>\n'
                '<analysis>STATIC</analysis><time_steps>1</time_steps><step_size>1</step_size>\n<plot_level>PLOT_NEVER</plot_level>\n'
                '<solver type="solid"><symmetric_stiffness>1</symmetric_stiffness><dtol>1e-5</dtol><etol>1e-7</etol><rtol>1e-6</rtol><lstol>0.9</lstol>'
                '<max_refs>25</max_refs><qn_method type="BFGS"><max_ups>10</max_ups></qn_method>\n'
                f'{cf.LS[solver].format(it=max_iter)}\n</solver>\n</Control>\n')
        f.write(f'<Material><material id="1" name="m" type="isotropic elastic"><density>1</density><E>{E}</E><v>{nu}</v></material></Material>\n')
        f.write('<Mesh>\n<Nodes name="todos">\n')
        xyz = ijk * hs
        f.write("".join(f'<node id="{i}">{a:.6f},{b:.6f},{c:.6f}</node>\n' for i, (a, b, c) in enumerate(xyz.tolist(), 1)))
        f.write('</Nodes>\n<Elements type="hex8" name="pieza">\n')
        f.write("".join('<elem id="%d">%d,%d,%d,%d,%d,%d,%d,%d</elem>\n' % (i, *r) for i, r in enumerate(conn.tolist(), 1)))
        f.write('</Elements>\n')
        f.write(f'<NodeSet name="dist">{cf._ids(fix)}</NodeSet>\n<Surface name="prox">\n')
        f.write("".join('<quad4 id="%d">%d,%d,%d,%d</quad4>\n' % (i, *r) for i, r in enumerate(fp.tolist(), 1)))
        f.write('</Surface>\n</Mesh>\n<MeshDomains><SolidDomain name="pieza" mat="m"/></MeshDomains>\n<Boundary>\n')
        for d in "xyz":
            f.write(f'<bc name="d{d}" type="prescribed displacement" node_set="dist"><dof>{d}</dof><value lc="1">0</value><relative>0</relative></bc>\n')
        f.write('</Boundary>\n')
        # La carga comprime: en la cara proximal apunta hacia distal.
        f.write(f'<Loads><surface_load type="traction" surface="prox"><scale lc="1">1</scale><traction>0,0,{-prox * p:.9g}</traction></surface_load></Loads>\n')
        f.write('<LoadData><load_controller id="1" name="lc" type="loadcurve"><interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
                '<points><pt>0,0</pt><pt>1,1</pt></points></load_controller></LoadData>\n')
        f.write('<Output><logfile>'
                '<node_data data="ux;uy;uz" delim="," file="u.txt"/>'
                '<node_data data="Rx;Ry;Rz" node_set="dist" delim="," file="r.txt"/>'
                '<element_data data="sx;sy;sz;sxy;syz;sxz" delim="," file="s.txt"/>'
                '<element_data data="Ex;Ey;Ez" delim="," file="e.txt"/>'
                '</logfile></Output>\n</febio_spec>\n')


def ultimo_bloque(path, ncol):
    t = open(path).read()
    i = t.rfind("*Data")
    b = t[t.index("\n", i) + 1:]
    a = np.array(b.replace(",", " ").split(), float)
    return a.reshape(-1, ncol + 1)


def primer_paso(path, ncol, paso=1):
    """Bloque del paso indicado de un logfile."""
    t = open(path).read()
    i = t.index(f"*Step  = {paso}\n")
    i = t.index("*Data", i)
    j = t.find("*Step", i)
    b = t[t.index("\n", i) + 1:(j if j > 0 else None)]
    return np.array(b.replace(",", " ").split(), float).reshape(-1, ncol + 1)


def von_mises(s):
    sx, sy, sz, sxy, syz, sxz = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + sxz ** 2))


# ------------------------------------------------------------------ A
def correr_caso(caso, h, banda=BANDA_BASE, solver=None):
    nombre = f"{caso}_h{h:g}_b{banda:g}"
    wd = WORK / nombre
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    Skji, hk, prox, info = envolvente(caso, h)
    S = np.transpose(Skji, (1, 2, 0))                # (j, i, k) = (x, y, z)
    hs = np.array([hk[1], hk[2], hk[0]])
    ijk, conn, fp, fd = None, None, None, None
    ijk, conn, (fp, fd) = malla_macro(S, 1.0, banda, prox)
    A_vox_z = hs[0] * hs[1]
    A_prox, A_dist = len(fp) * A_vox_z, len(fd) * A_vox_z
    zs = np.nonzero(S.any(axis=(0, 1)))[0]
    z_mid = (zs.min() + zs.max()) // 2
    A_mid = float(S[:, :, z_mid].sum()) * A_vox_z
    area = S.sum(axis=(0, 1)) * A_vox_z
    L = float(len(zs) * hs[2])
    pr = celda_props(CASOS[caso]["celda"])
    F_ref = EPS_REF * pr["E"] * A_mid
    p = F_ref / A_prox
    t0 = time.time()
    escribir(wd / "m.feb", ijk, conn, fp, fd, hs, pr["E"], pr["nu"], p, prox, solver or SOLVER)
    tw = time.time() - t0
    run = cf.correr(wd)
    fin = (wd / "m.log").read_text(errors="replace") if (wd / "m.log").exists() else ""
    ok = "N O R M A L   T E R M I N A T I O N" in fin
    out = dict(caso=caso, h_mm=h, banda=banda, solver=solver or SOLVER, n_elementos=len(conn), n_nodos=len(ijk), L_mm=L, V_mm3=float(S.sum() * hs.prod()), A_mid_mm2=A_mid, A_prox_mm2=A_prox, A_dist_mm2=A_dist,
               A_min_mm2=float(area[zs].min()), A_max_mm2=float(area[zs].max()), E_MPa=pr["E"], nu=pr["nu"], F_ref_N=F_ref, prox_signo=prox, geom=info, t_escribir_s=round(tw, 1), **run, ok=ok)
    if not ok:
        out["cola_log"] = fin[-1500:]
        return out
    u = ultimo_bloque(wd / "u.txt", 3)[:, 1:]
    r = ultimo_bloque(wd / "r.txt", 3)[:, 1:]
    s = ultimo_bloque(wd / "s.txt", 6)[:, 1:]
    e = ultimo_bloque(wd / "e.txt", 3)[:, 1:]
    vm = von_mises(s)
    Rz = r[:, 2].sum()
    um = np.linalg.norm(u, axis=1)
    Ez = e[:, 2]                                     # Ez < 0 = compresión axial
    out.update(R_ref_N=[float(x) for x in r.sum(axis=0)], err_resultante=float(abs(abs(Rz) - F_ref) / F_ref),
               u_max_ref_mm=float(um.max()), uz_medio_prox_ref_mm=float(np.abs(u[np.unique(fp) - 1, 2]).mean()), u_medio_ref_mm=float(um.mean()),
               Ez_med_ref=float(-Ez.mean()), Ez_p95_ref=float(np.percentile(np.abs(Ez), 95)), vm_med_ref=float(vm.mean()), vm_p95_ref=float(np.percentile(vm, 95)), vm_max_ref=float(vm.max()), sz_med_ref=float(-s[:, 2].mean()))
    np.savez_compressed(wd / "campo.npz", S=S, vm=vm.astype(np.float32), hs=hs, prox=prox, F_ref=F_ref, lr_col=info["lr_col"], ap_col=info["ap_col"])
    for q in ("m.feb", "u.txt", "r.txt", "s.txt", "e.txt"):
        if (wd / q).exists():
            (wd / q).unlink()
    return out


def guardar_json(res):
    json.dump(res, open(HERE / "corridas_macro.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ B
def celda_tensiones(tag, ncel=4, nv=25):
    """Corrida de celda con tensiones elementales. Paso 1 = eps 0,1 %: VM/eps es lineal y se escala."""
    key = cf.DISENO[tag][0]
    m = f4.MATERIALS[key]
    S, h, info = cf.giroide(tag, ncel, nv)
    wd = WORK / f"celda_{tag}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    ne, nn, L, _ = cf.escribir_feb(wd / "m.feb", S, h, m["E_MPa"], m["nu"], "cg_ichol", tensiones=True)
    run = cf.correr(wd)
    ok = "N O R M A L   T E R M I N A T I O N" in (wd / "m.log").read_text(errors="replace")
    out = dict(tag=tag, celdas=ncel, vox_por_celda=nv, h_mm=h, n_elementos=ne, dens_rel=float(S.mean()), E_s=m["E_MPa"], **run, ok=ok)
    if ok:
        ev = cf.evaluar(wd, L, m["E_MPa"], S.mean())
        s = primer_paso(wd / "tensiones.txt", 6, 1)[:, 1:]
        vm = von_mises(s)
        idx = np.argwhere(S)
        n = np.array(S.shape)
        nucleo = np.all((idx >= n / 4) & (idx < 3 * n / 4), axis=1)          # 2x2x2 celdas centrales
        eps = ev["lin"]["eps"]
        est = lambda v: dict(media=float(v.mean() / eps), p95=float(np.percentile(v, 95) / eps), p99=float(np.percentile(v, 99) / eps))
        out.update(lin=ev["lin"], sec=ev["sec"], vm_sobre_eps_bloque=est(vm), vm_sobre_eps_nucleo=est(vm[nucleo]), n_nucleo=int(nucleo.sum()))
    for q in wd.iterdir():
        if q.name not in ("m.log", "consola.txt"):
            q.unlink()
    json.dump(out, open(HERE / f"celda_{tag}_tensiones.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "caso":
        r = correr_caso(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else BANDA_BASE, sys.argv[5] if len(sys.argv) > 5 else None)
        print(json.dumps(r, ensure_ascii=False))
    elif op == "celdas":
        for t in sys.argv[2:] or ("Ti", "PCL"):
            celda_tensiones(t)
    elif op == "tabla":
        pv = HERE / "corridas_macro.json"
        res = json.load(open(pv, encoding="utf-8")) if pv.exists() else []
        res = [r for r in res if r["ok"]]
        hechos = {(r["caso"], r["h_mm"], r["banda"]) for r in res}
        plan = [(c, h, BANDA_BASE) for h in TAMANOS for c in CASOS] + [(c, 0.7, BANDA_SENS) for c in CASOS]
        plan.sort(key=lambda x: -x[1])
        for c, h, b in plan:
            if (c, h, b) in hechos:
                continue
            r = correr_caso(c, h, b)
            if not r["ok"] and not r["aborto"]:
                r = correr_caso(c, h, b, "cg_amg")
            print(json.dumps({k: v for k, v in r.items() if k not in ("geom", "cola_log")}, ensure_ascii=False), flush=True)
            res.append(r); guardar_json(res)
