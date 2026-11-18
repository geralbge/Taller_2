"""Benchmark del módulo efectivo de la celda (giroide lámina Ti6Al4V) contra Luo et al. (2023), Thin-Walled Struct. 192:111098.

Reutiliza celda_febio.py sin modificarlo (solo cambia EPS, FRAC_LIN, WORK y RAM_MAX_GB en tiempo de ejecución).
eps = 0,1 % (paso "sec"); el paso intermedio de 0,05 % ("lin") sirve de control de linealidad.

Uso (python de .venv):
  benchmark_luo.py auditoria        espesor real, densidad, área neta
  benchmark_luo.py luo              casos rho 0,309 y 0,461 (3x3x3)
  benchmark_luo.py cc               contorno: laterales libres vs rodillos en las 6 caras
"""
import json, shutil, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import celda_febio as cf
sc = cf.sc

cf.EPS = 0.001
cf.FRAC_LIN = 0.5
cf.WORK = HERE / "trabajo"
cf.RAM_MAX_GB = 8.0
A_MM, E_S, NU = 2.5, 110000.0, 0.32
SALIDA = HERE / "casos.json"


def G_ph(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def grad_ph(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([np.cos(x) * np.cos(y) - np.sin(z) * np.sin(x),
                     np.cos(y) * np.cos(z) - np.sin(x) * np.sin(y),
                     np.cos(z) * np.cos(x) - np.sin(y) * np.sin(z)], axis=-1)


def espesor_real(w, n=200000, seed=1):
    """Espesor normal real (mm) del conjunto |G|<=w, medido sobre puntos de G=0 con trazado a lo largo de la normal."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 2 * np.pi, (n, 3))
    for _ in range(40):                      # Newton a G=0
        g = grad_ph(p); p -= (G_ph(p) / np.maximum((g ** 2).sum(-1), 1e-9))[..., None] * g
    p = p[np.abs(G_ph(p)) < 1e-9]
    g = grad_ph(p); nrm = g / np.linalg.norm(g, axis=-1, keepdims=True)
    t = np.zeros(len(p))
    for signo, objetivo in ((1, w), (-1, -w)):
        lo, hi = np.zeros(len(p)), np.full(len(p), 2.0)
        for _ in range(45):
            m = 0.5 * (lo + hi)
            dentro = signo * G_ph(p + signo * m[..., None] * nrm) < w
            lo[dentro] = m[dentro]; hi[~dentro] = m[~dentro]
        t += lo
    return t * A_MM / (2 * np.pi)


def bloque(ncel, nv, w):
    h = A_MM / nv
    q = ((np.arange(ncel * nv) + 0.5) * h * 2 * np.pi / A_MM).astype(np.float32)
    G = sc.TPMS["gyroid"](q[:, None, None], q[None, :, None], q[None, None, :])
    S = np.abs(G) <= w
    lab, k = ndi.label(S)
    tam = np.bincount(lab.ravel()); tam[0] = 0
    keep = lab == tam.argmax()
    return keep, h, int(k - 1), int(S.sum() - keep.sum())


def w_para_rho(rho, nv):
    """Umbral que da la densidad objetivo sobre la grilla de la celda periódica usada en el caso."""
    q = (np.arange(nv) + 0.5) * 2 * np.pi / nv
    G = np.abs(sc.TPMS["gyroid"](q[:, None, None], q[None, :, None], q[None, None, :])).ravel()
    return float(np.sort(G)[int(round(rho * G.size)) - 1] + 1e-7)


def w_a_t(w):
    return 2 * w / (sc.grad_scale("gyroid") * 2 * np.pi / A_MM)


def resolver_cc(nombre, S, h, cc):
    """Como cf.resolver, pero admite rodillos en x=L e y=L."""
    wd = cf.WORK / nombre
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    ne, nn, L, sets = cf.escribir_feb(wd / "m.feb", S, h, E_S, NU, "cg_ichol", 3000)
    if cc == "rodillos6":
        txt = (wd / "m.feb").read_text()
        extra = ('<bc name="rxm" type="zero displacement" node_set="xmax"><x_dof>1</x_dof><y_dof>0</y_dof><z_dof>0</z_dof></bc>\n'
                 '<bc name="rym" type="zero displacement" node_set="ymax"><x_dof>0</x_dof><y_dof>1</y_dof><z_dof>0</z_dof></bc>\n')
        assert txt.count("</Boundary>") == 1
        (wd / "m.feb").write_text(txt.replace("</Boundary>", extra + "</Boundary>"))
    run = cf.correr(wd)
    fin = (wd / "m.log").read_text(errors="replace") if (wd / "m.log").exists() else ""
    ok = "N O R M A L   T E R M I N A T I O N" in fin
    out = dict(caso=nombre, cc=cc, h_mm=h, n_elementos=ne, n_nodos=nn, dens_rel=float(S.mean()), **run, terminacion_normal=ok)
    if ok:
        out.update(cf.evaluar(wd, L, E_S, S.mean()))
    else:
        out["cola_log"] = fin[-1500:]
    for q in list(wd.glob("*.feb")) + list(wd.glob("*.txt")):
        if q.name != "consola.txt" and q.stat().st_size > 2e7:
            q.unlink()
    return out


def agregar(r):
    previos = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else []
    previos = [x for x in previos if x["caso"] != r["caso"]] + [r]
    SALIDA.write_text(json.dumps(previos, indent=1, ensure_ascii=False), encoding="utf-8")


def correr_caso(nombre, ncel, nv, w, cc, t_obj):
    S, h, islas, quitados = bloque(ncel, nv, w)
    r = resolver_cc(nombre, S, h, cc)
    r.update(celdas=ncel, vox_por_celda=nv, w=w, t_mm_lineal=w_a_t(w), pared_vox=w_a_t(w) / h, islas=islas, vox_quitados=quitados, rho_objetivo=t_obj)
    agregar(r)
    print(json.dumps({k: v for k, v in r.items() if k != "cola_log"}), flush=True)
    return r


def auditoria():
    w = sc.w_for_wall("gyroid", "sheet", 2.5, 0.30)
    t = espesor_real(w)
    # densidad "continua" en grilla 400^3 de la celda periódica (sin escalonado del bloque)
    n = 400
    q = (np.arange(n) + 0.5) * 2 * np.pi / n
    G = np.abs(sc.TPMS["gyroid"](q[:, None, None].astype(np.float32), q[None, :, None].astype(np.float32), q[None, None, :].astype(np.float32)))
    rho_fina = float((G <= w).mean())
    S25, h25, isl, q25 = bloque(4, 25, w)
    fra = [float(S25[:, :, i].mean()) for i in (0, S25.shape[2] // 2, -1)]
    area = {}
    for nv in (25, 33, 50):
        S, *_ = bloque(1, nv, w)
        area[nv] = float(S.mean())
    salida = dict(grad_scale=sc.grad_scale("gyroid"), w=w, t_medio_mm=float(t.mean()), t_min_mm=float(t.min()), t_max_mm=float(t.max()),
                  t_p05_mm=float(np.percentile(t, 5)), t_p95_mm=float(np.percentile(t, 95)), n_pts=len(t),
                  rho_continua_400=rho_fina, rho_grilla=area, rho_bloque_4x25=float(S25.mean()), islas=isl, vox_quitados=q25,
                  fraccion_solida_z0_mitad_zL=fra, area_aparente_mm2=(4 * 2.5) ** 2,
                  estimacion_area_superficie_rho=rho_fina / (0.30 / 2.5))
    (HERE / "auditoria.json").write_text(json.dumps(salida, indent=1), encoding="utf-8")
    print(json.dumps(salida, indent=1), flush=True)


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "auditoria":
        auditoria()
    elif op == "luo":
        auditoria()
        for rho, nv in ((0.309, 33), (0.461, 25)):
            w = w_para_rho(rho, nv)
            correr_caso(f"luo_rho{int(rho*1000)}_c3_v{nv}", 3, nv, w, "libre", rho)
    elif op == "cc":
        w = sc.w_for_wall("gyroid", "sheet", 2.5, 0.30)
        correr_caso("cc_rho374_c3_v25_libre", 3, 25, w, "libre", 0.374)
        correr_caso("cc_rho374_c3_v25_rodillos6", 3, 25, w, "rodillos6", 0.374)
        for rho, nv in ((0.309, 33), (0.461, 25)):
            w = w_para_rho(rho, nv)
            correr_caso(f"luo_rho{int(rho*1000)}_c3_v{nv}_rodillos6", 3, nv, w, "rodillos6", rho)
