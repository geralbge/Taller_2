import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import macro_febio as m
out=[]
for R in (0.75, 1.75, 2.25):
    m.R_APERTURA_MM=R
    for c in ("C_Ti","P2_Ti"):
        r=m.correr_caso(c,0.7)
        d=dict(R_mm=R,caso=c,ok=r['ok'],aborto=r['aborto'],t_s=r['t_s'],V0=r['geom']['V_antes_apertura_mm3'],V=r['V_mm3'],L=r['L_mm'],A_mid=r['A_mid_mm2'],solver=r['solver'])
        if r['ok']: d.update(u_por_N=r['u_max_ref_mm']/r['F_ref_N'],vm_p95_por_N=r['vm_p95_ref']/r['F_ref_N'],eps_p95_por_N=r['Ez_p95_ref']/r['F_ref_N'])
        print(d,flush=True); out.append(d)
json.dump(out,open(HERE / "sensibilidad_apertura.json","w"),indent=1)
