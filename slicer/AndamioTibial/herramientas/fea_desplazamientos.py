"""Campo de desplazamiento del FEA macro, para el visor del módulo de Slicer (deformada exagerada).

macro_febio.py guarda en campo.npz la envolvente y el Von Mises, pero no el desplazamiento por nodo
(borra u.txt al terminar). Este script repite la MISMA corrida con las funciones de macro_febio.py,
sin cambiarlas, en una carpeta aparte, y guarda u:

  results/fea/macro_febio/trabajo_u/<corrida>/campo_u.npz   S, u (mm, por nodo, a F_ref), F_ref, vm

No toca trabajo/, corridas_macro.json ni resultados_macro.csv. Comprueba que la envolvente y el Von Mises
coinciden con el campo.npz vigente; si no, no guarda nada.

Uso (Python de .venv, el mismo de macro_febio.py; necesita FEBio):
  python fea_desplazamientos.py                  las 4 corridas finales (h 0.5, banda 0.25)
  python fea_desplazamientos.py C_Ti P2_PCL      solo esas
"""
import json, shutil, sys, time
from pathlib import Path

import numpy as np

MACRO = Path(__file__).resolve().parents[3] / "results" / "fea" / "macro_febio"
sys.path.insert(0, str(MACRO))
import macro_febio as mf          # noqa: E402
cf = mf.cf

H, BANDA = 0.5, mf.BANDA_BASE


def correr(caso):
    nombre = f"{caso}_h{H:g}_b{BANDA:g}"
    ref = np.load(mf.WORK / nombre / "campo.npz")
    wd = MACRO / "trabajo_u" / nombre
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    # Igual que mf.correr_caso, sin borrar trabajo/ ni los .txt hasta extraer u.
    Skji, hk, prox, info = mf.envolvente(caso, H)
    S = np.transpose(Skji, (1, 2, 0))
    hs = np.array([hk[1], hk[2], hk[0]])
    if S.shape != ref["S"].shape or not np.array_equal(S, ref["S"]):
        raise RuntimeError(f"{nombre}: la envolvente no coincide con el campo.npz vigente")
    ijk, conn, (fp, fd) = mf.malla_macro(S, 1.0, BANDA, prox)
    zs = np.nonzero(S.any(axis=(0, 1)))[0]
    A_mid = float(S[:, :, (zs.min() + zs.max()) // 2].sum()) * hs[0] * hs[1]
    pr = mf.celda_props(mf.CASOS[caso]["celda"])
    F_ref = mf.EPS_REF * pr["E"] * A_mid
    p = F_ref / (len(fp) * hs[0] * hs[1])
    corr = json.load(open(MACRO / "corridas_macro.json", encoding="utf-8"))
    solver = next((r["solver"] for r in corr if r["caso"] == caso and r["h_mm"] == H and r["banda"] == BANDA and r["ok"]), mf.SOLVER)
    mf.escribir(wd / "m.feb", ijk, conn, fp, fd, hs, pr["E"], pr["nu"], p, prox, solver)   # el mismo solver que la corrida vigente
    t0 = time.time()
    run = cf.correr(wd)
    log = (wd / "m.log").read_text(errors="replace") if (wd / "m.log").exists() else ""
    if "N O R M A L   T E R M I N A T I O N" not in log:
        raise RuntimeError(f"{nombre}: FEBio no terminó bien ({run})")
    u = mf.ultimo_bloque(wd / "u.txt", 3)[:, 1:]
    vm = mf.von_mises(mf.ultimo_bloque(wd / "s.txt", 6)[:, 1:])
    if len(u) != len(ijk):
        raise RuntimeError(f"{nombre}: {len(u)} desplazamientos para {len(ijk)} nodos")
    err = float(np.max(np.abs(vm - ref["vm"])) / max(float(np.max(ref["vm"])), 1e-30))
    if abs(F_ref - float(ref["F_ref"])) > 1e-9 * F_ref or err > 1e-3:
        raise RuntimeError(f"{nombre}: no reproduce campo.npz (F_ref {F_ref} vs {float(ref['F_ref'])}, error VM {err:.2e})")
    np.savez_compressed(wd / "campo_u.npz", S=S, u=u.astype(np.float32), F_ref=F_ref, vm=vm.astype(np.float32), hs=hs, prox=prox)
    for q in ("m.feb", "u.txt", "r.txt", "s.txt", "e.txt"):
        if (wd / q).exists():
            (wd / q).unlink()
    print(f"{nombre}: ok en {time.time() - t0:.0f} s · u_max a F_ref {np.linalg.norm(u, axis=1).max():.3e} mm · error VM {err:.1e}")


if __name__ == "__main__":
    for c in sys.argv[1:] or list(mf.CASOS):
        correr(c)
