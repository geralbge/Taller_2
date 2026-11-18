"""Escala 1 del FEA en dos escalas: módulo efectivo de la celda del andamio con FEBio 4.

Compresión uniaxial cuasiestática (1 %) de un bloque de giroide lámina, malla HEX8 conforme a vóxeles.
Geometría y materiales salen de src/scaffold (misma función implícita y mismo umbral w_for_wall);
no se repite ningún valor de material en este archivo salvo celda y pared de diseño.

Condiciones de contorno: z=0 uz=0; x=0 ux=0; y=0 uy=0; z=L uz prescrito; resto libre.
Curva de carga en dos pasos: 0,1 × eps y eps. Motivo: "isotropic elastic" de FEBio es Saint Venant-Kirchhoff
(cinemática finita), y al 1 % en compresión F/(A·eps) = 0,985 E en un bloque macizo sin que haya ningún error de
contorno. El módulo lineal es el del paso pequeño (0,1 %); la secante al 1 % se reporta aparte.

Uso (python de .venv):
  celda_febio.py humo                       flujo mínimo: 1 celda, vóxel grueso
  celda_febio.py control                    bloque macizo, ambos materiales
  celda_febio.py caso Ti|PCL <celdas> <vox_por_celda> [solver]
  celda_febio.py tabla [solver]             control + convergencia, reanudable; guarda csv y json
Solvers: cg_ichol (por defecto en tabla) | pardiso | cg_amg | fgmres_ilu0
"""
import csv, ctypes, json, os, shutil, subprocess, sys, time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]  # raíz del repositorio
sys.path.insert(0, str(ROOT / "src" / "scaffold"))
import scaffold_core as sc
import scaffold_f4 as f4

FEBIO = Path(os.environ.get("FEBIO_EXE", r"C:\Program Files\FEBioStudio\bin\febio4.exe"))  # ruta de febio4; se puede cambiar con la variable FEBIO_EXE
HERE = Path(__file__).resolve().parent
WORK = HERE / "trabajo"
EPS = 0.01          # deformación nominal (compresión)
FRAC_LIN = 0.1      # fracción de EPS del primer paso (módulo lineal)
HILOS = 4           # núcleos físicos del equipo
RAM_MAX_GB = 10.0   # límite de la tarea: por encima se aborta el caso
RESERVA_GB = 0.15   # RAM física disponible mínima que se deja al sistema (el archivo de paginación es automático)
T_MAX_S = 1800.0
DISENO = {"Ti": ("Ti6Al4V_LPBF", 2.5, 0.30), "PCL": ("PCL_bTCP_extrusion", 4.2, 0.50)}
PARED_VOX = (3, 4, 6)

LS = {
    "pardiso": '<linear_solver type="pardiso"/>',
    "cg_ichol": '<linear_solver type="cg"><max_iter>{it}</max_iter><tol>1e-7</tol><print_level>0</print_level>'
                '<pc_left type="ichol"/></linear_solver>',
    "cg_amg": '<linear_solver type="cg"><max_iter>{it}</max_iter><tol>1e-7</tol><print_level>0</print_level>'
              '<pc_left type="boomeramg"><max_iter>1</max_iter><tol>0</tol><print_level>0</print_level>'
              '<use_num_funcs>1</use_num_funcs><strong_threshold>0.5</strong_threshold></pc_left></linear_solver>',
    "fgmres_ilu0": '<linear_solver type="fgmres"><max_iter>{it}</max_iter><tol>1e-7</tol><print_level>0</print_level>'
                   '<fail_max_iters>1</fail_max_iters><pc_left type="ilu0"/></linear_solver>',
}


# ------------------------------------------------------------------ memoria (Win32, sin dependencias)
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                       "PagefileUsage", "PeakPagefileUsage")]


class _MSX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
        (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                          "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


_psapi = ctypes.WinDLL("psapi")
_k32 = ctypes.WinDLL("kernel32")
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]


def mem_proc(handle):
    """(commit actual, pico de commit, pico de working set) en GB."""
    m = _PMC(); m.cb = ctypes.sizeof(m)
    _psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(handle)), ctypes.byref(m), m.cb)
    return m.PagefileUsage / 2**30, m.PeakPagefileUsage / 2**30, m.PeakWorkingSetSize / 2**30


def ram_libre_gb():
    s = _MSX(); s.dwLength = ctypes.sizeof(s)
    _k32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2**30


# ------------------------------------------------------------------ geometría
def giroide(tag, n_celdas, nv):
    """Bloque de giroide lámina; ejes (x, y, z) = índices (0, 1, 2). G es invariante a permutación cíclica de
    sus argumentos, así que comprimir en z equivale al eje largo k del andamio (primer argumento)."""
    key, a, t = DISENO[tag]
    h = a / nv
    n = n_celdas * nv
    q = ((np.arange(n) + 0.5) * h * 2 * np.pi / a).astype(np.float32)
    G = sc.TPMS["gyroid"](q[:, None, None], q[None, :, None], q[None, None, :])
    S = np.abs(G) <= sc.w_for_wall("gyroid", "sheet", a, t)
    del G
    lab, k = ndi.label(S)                          # conectividad 6: dos vóxeles que comparten solo arista no transmiten carga en la malla
    tam = np.bincount(lab.ravel()); tam[0] = 0
    keep = lab == tam.argmax()
    return keep, h, dict(islas_eliminadas=int(k - 1), vox_eliminados=int(S.sum() - keep.sum()))


def malla(S, h):
    """Nodos y conectividad HEX8 (orden FEBio, jacobiano positivo en x, y, z)."""
    idx = np.argwhere(S).astype(np.int64)
    nx = np.array(S.shape, np.int64) + 1
    off = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.int64)
    c = np.stack([(idx[:, 0] + o[0]) * nx[1] * nx[2] + (idx[:, 1] + o[1]) * nx[2] + idx[:, 2] + o[2] for o in off], axis=1)
    un, inv = np.unique(c, return_inverse=True)
    conn = inv.reshape(c.shape) + 1
    ijk = np.stack([un // (nx[1] * nx[2]), (un // nx[2]) % nx[1], un % nx[2]], axis=1)
    return ijk, conn


def _ids(v):
    return ",".join(map(str, v.tolist()))


def escribir_feb(path, S, h, E, nu, solver, max_iter=2000, tensiones=False):
    ijk, conn = malla(S, h)
    L = np.array(S.shape) * h
    nid = np.arange(1, len(ijk) + 1)
    sets = dict(bot=nid[ijk[:, 2] == 0], top=nid[ijk[:, 2] == S.shape[2]], x0=nid[ijk[:, 0] == 0], y0=nid[ijk[:, 1] == 0],
                xmax=nid[ijk[:, 0] == S.shape[0]], ymax=nid[ijk[:, 1] == S.shape[1]])
    with open(path, "w", buffering=2**20) as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n<febio_spec version="4.0">\n<Module type="solid"/>\n<Control>\n'
                '<analysis>STATIC</analysis><time_steps>2</time_steps><step_size>1</step_size>\n'
                '<plot_level>PLOT_NEVER</plot_level>\n<solver type="solid">\n'
                '<symmetric_stiffness>1</symmetric_stiffness><dtol>1e-5</dtol><etol>1e-7</etol><rtol>1e-6</rtol><lstol>0.9</lstol>'
                '<max_refs>25</max_refs><qn_method type="BFGS"><max_ups>10</max_ups></qn_method>\n'
                f'{LS[solver].format(it=max_iter)}\n</solver>\n</Control>\n')
        f.write(f'<Material><material id="1" name="m" type="isotropic elastic"><density>1</density><E>{E}</E><v>{nu}</v></material></Material>\n')
        f.write('<Mesh>\n<Nodes name="todos">\n')
        xyz = ijk * h
        f.write("".join(f'<node id="{i}">{a:.6f},{b:.6f},{c:.6f}</node>\n' for i, (a, b, c) in zip(nid.tolist(), xyz.tolist())))
        f.write('</Nodes>\n<Elements type="hex8" name="pieza">\n')
        f.write("".join('<elem id="%d">%d,%d,%d,%d,%d,%d,%d,%d</elem>\n' % (i, *r) for i, r in enumerate(conn.tolist(), 1)))
        f.write('</Elements>\n')
        for k, v in sets.items():
            f.write(f'<NodeSet name="{k}">{_ids(v)}</NodeSet>\n')
        f.write('</Mesh>\n<MeshDomains><SolidDomain name="pieza" mat="m"/></MeshDomains>\n<Boundary>\n')
        for nm, st, d in (("rx", "x0", "x"), ("ry", "y0", "y")):
            fx = "".join(f"<{q}_dof>{int(q == d)}</{q}_dof>" for q in "xyz")
            f.write(f'<bc name="{nm}" type="zero displacement" node_set="{st}">{fx}</bc>\n')
        f.write('<bc name="b" type="prescribed displacement" node_set="bot"><dof>z</dof><value lc="1">0</value><relative>0</relative></bc>\n')
        f.write(f'<bc name="t" type="prescribed displacement" node_set="top"><dof>z</dof><value lc="1">{-EPS * L[2]:.9g}</value><relative>0</relative></bc>\n')
        f.write('</Boundary>\n<LoadData><load_controller id="1" name="lc" type="loadcurve"><interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
                f'<points><pt>0,0</pt><pt>1,{FRAC_LIN}</pt><pt>2,1</pt></points></load_controller></LoadData>\n')
        f.write('<Output><logfile>'
                '<node_data data="Rz" node_set="top" delim="," file="rz_top.txt"/>'
                '<node_data data="Rz" node_set="bot" delim="," file="rz_bot.txt"/>'
                '<node_data data="ux" node_set="xmax" delim="," file="ux_xmax.txt"/>'
                '<node_data data="uy" node_set="ymax" delim="," file="uy_ymax.txt"/>'
                '<node_data data="uz" node_set="top" delim="," file="uz_top.txt"/>'
                + ('<element_data data="sx;sy;sz;sxy;syz;sxz" delim="," file="tensiones.txt"/>' if tensiones else '') +
                '</logfile></Output>\n</febio_spec>\n')
    return len(conn), len(ijk), L, {k: len(v) for k, v in sets.items()}


# ------------------------------------------------------------------ ejecución y lectura
def correr(wd, feb="m.feb"):
    """Corre febio4 vigilando RAM y tiempo; nunca deja al sistema sin RAM física."""
    env = {**os.environ, "OMP_NUM_THREADS": str(HILOS)}
    t0 = time.time()
    motivo = None
    pico = (0.0, 0.0, 0.0)
    with open(wd / "consola.txt", "w") as out:
        p = subprocess.Popen([str(FEBIO), "-i", feb, "-silent"], cwd=wd, stdout=out, stderr=subprocess.STDOUT, env=env)
        while p.poll() is None:
            time.sleep(1.0)
            pico = mem_proc(p._handle) if p.poll() is None else pico
            if pico[0] > RAM_MAX_GB:
                motivo = f"RAM > {RAM_MAX_GB} GB"
            elif ram_libre_gb() < RESERVA_GB:
                motivo = f"RAM física disponible < {RESERVA_GB} GB"
            elif time.time() - t0 > T_MAX_S:
                motivo = f"tiempo > {T_MAX_S:.0f} s"
            if motivo:
                p.kill(); p.wait()
                break
        else:
            pico = mem_proc(p._handle)
    return dict(rc=p.returncode, aborto=motivo, t_s=round(time.time() - t0, 1), ram_commit_pico_gb=round(pico[1], 2), ram_ws_pico_gb=round(pico[2], 2))


def leer_log(path):
    """{paso: array de valores (columna 2)} de un logfile de FEBio."""
    res, paso, buf = {}, None, []
    with open(path) as f:
        for ln in f:
            if ln.startswith("*Step"):
                if paso is not None:
                    res[paso] = np.array(buf)
                paso = int(ln.split("=")[1]); buf = []
            elif ln.startswith("*"):
                continue
            elif ln.strip():
                buf.append(float(ln.replace(",", " ").split()[1]))
    if paso is not None:
        res[paso] = np.array(buf)
    return res


def evaluar(wd, L, E, S_frac):
    rt, rb, ux, uy, uz = (leer_log(wd / f"{n}.txt") for n in ("rz_top", "rz_bot", "ux_xmax", "uy_ymax", "uz_top"))
    A = L[0] * L[1]
    pasos = sorted(rt)
    r = {}
    for p, nom in zip(pasos[-2:], ("lin", "sec")):
        eps = -uz[p].mean() / L[2]
        F = abs(rt[p].sum())
        Eef = F / A / eps
        nux, nuy = (ux[p].mean() / L[0]) / eps, (uy[p].mean() / L[1]) / eps
        r[nom] = dict(eps=float(eps), F_N=float(F), E_eff_MPa=float(Eef), E_rel=float(Eef / E), nu_x=float(nux), nu_y=float(nuy), nu_eff=float(0.5 * (nux + nuy)),
                      desbalance=float(abs(rt[p].sum() + rb[p].sum()) / F))
    r["pasos_log"] = len(pasos)
    return r


def resolver(nombre, S, h, E, nu, solver, max_iter=2000):
    wd = WORK / nombre
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    t0 = time.time()
    ne, nn, L, sets = escribir_feb(wd / "m.feb", S, h, E, nu, solver, max_iter)
    tw = time.time() - t0
    run = correr(wd)
    fin = (wd / "m.log").read_text(errors="replace") if (wd / "m.log").exists() else ""
    ok = "N O R M A L   T E R M I N A T I O N" in fin
    out = dict(caso=nombre, solver=solver, h_mm=h, n_elementos=ne, n_nodos=nn, ngl_aprox=3 * nn, dens_rel=float(S.mean()), nodos_cara=sets, t_escribir_s=round(tw, 1), **run, terminacion_normal=ok)
    if ok:
        out.update(evaluar(wd, L, E, S.mean()))
    else:
        out["cola_log"] = fin[-1500:]
    # los .feb y logs nodales grandes no se conservan; se conservan m.log y consola.txt
    for q in list(wd.glob("*.feb")) + list(wd.glob("*.txt")):
        if q.name != "consola.txt" and q.stat().st_size > 2e7:
            q.unlink()
    return out


def fila(tag, ncel, nv, r):
    key = tag if tag.startswith("control") else DISENO[tag][0]
    d = dict(material=key, celdas=ncel, vox_por_celda=nv, h_mm=round(r["h_mm"], 5), solver=r["solver"], n_elementos=r["n_elementos"], ngl=r["ngl_aprox"],
             dens_rel=round(r["dens_rel"], 4), t_s=r["t_s"], ram_commit_pico_gb=r["ram_commit_pico_gb"], ok=r["terminacion_normal"], aborto=r["aborto"])
    if r["terminacion_normal"]:
        for k, nom in (("lin", "eps0p1"), ("sec", "eps1")):
            d[f"E_eff_MPa_{nom}"] = round(r[k]["E_eff_MPa"], 3)
            d[f"E_rel_{nom}"] = round(r[k]["E_rel"], 5)
            d[f"nu_eff_{nom}"] = round(r[k]["nu_eff"], 4)
    return d


def control(solver):
    """Bloque macizo con las dimensiones del bloque de 3 celdas. La solución exacta es homogénea, así que 16^3 vóxeles bastan."""
    out = []
    for tag, (key, a, _) in DISENO.items():
        m = f4.MATERIALS[key]
        S = np.ones((16, 16, 16), bool)
        r = resolver(f"control_{tag}", S, 3 * a / 16, m["E_MPa"], m["nu"], solver)
        print(json.dumps({k: r[k] for k in ("caso", "t_s", "terminacion_normal")}), r.get("lin"), r.get("sec"), flush=True)
        out.append(r)
    return out


def caso(tag, ncel, nv, solver, max_iter=2000):
    key, a, t = DISENO[tag]
    m = f4.MATERIALS[key]
    S, h, info = giroide(tag, ncel, nv)
    r = resolver(f"{tag}_c{ncel}_v{nv}_{solver}", S, h, m["E_MPa"], m["nu"], solver, max_iter)
    r.update(info, material=key, celdas=ncel, vox_por_celda=nv, pared_vox=t / h)
    print(json.dumps({k: v for k, v in r.items() if k != "cola_log"}), flush=True)
    return r


def guardar(res):
    filas = [fila(*x) for x in res]
    with open(HERE / "resultados_celda.csv", "w", newline="", encoding="utf-8") as f:
        cols = list(dict.fromkeys(k for d in filas for k in d))
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(filas)
    json.dump([dict(tag=t, celdas=c, vox_por_celda=v, **r) for t, c, v, r in res], open(HERE / "resultados_celda.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "humo":
        caso("Ti", 1, 12, "pardiso")
    elif op == "control":
        control(sys.argv[2] if len(sys.argv) > 2 else "pardiso")
    elif op == "caso":
        caso(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "pardiso")
    elif op == "tabla":
        # Reanudable: lo ya resuelto en resultados_celda.json no se repite. Orden de menor a mayor memoria.
        # Malla 2x2x2 a pared de 3, 4 y 6 vóxeles para la convergencia (3x3x3 a 6 vóxeles ~22 GB: excede el límite);
        # 3x3x3 a 3 y 4 vóxeles para el valor reportado.
        solver = sys.argv[2] if len(sys.argv) > 2 else "cg_ichol"
        plan = [(2, 3), (2, 4), (3, 3), (2, 6), (3, 4), (4, 3)]
        previos = json.load(open(HERE / "resultados_celda.json", encoding="utf-8")) if (HERE / "resultados_celda.json").exists() else []
        res = [(r.pop("tag"), r.pop("celdas"), r.pop("vox_por_celda"), r) for r in previos]
        hechos = {(t, c, v) for t, c, v, r in res if r["terminacion_normal"]}
        res = [x for x in res if x[3]["terminacion_normal"]]
        if not any(t.startswith("control") for t, *_ in res):
            res += [(f"control_{t}", 3, 16, r) for t, r in zip(DISENO, control("pardiso"))]
            guardar(res)
        for ncel, pv in plan:
            for tag, (key, a, t) in DISENO.items():
                nv = int(round(a / (t / pv)))
                if (tag, ncel, nv) in hechos:
                    continue
                r = caso(tag, ncel, nv, solver)
                r.pop("material", None); r.pop("celdas", None); r.pop("vox_por_celda", None)
                res.append((tag, ncel, nv, r))
                guardar(res)
