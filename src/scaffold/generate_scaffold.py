"""
Genera el andamio poroso TPMS dentro del defecto caracterizado y calcula las
métricas de arquitectura porosa y verificación geométrica.

Entrada: máscara del defecto en la grilla de la ROI orientada (NIfTI), p. ej.
  results/pathological/<alias>/<alias>_defect.nii.gz         (casos reales)
  o generada por src/defect/export_defect_models.py           (defectos virtuales)

Diseño base (parámetros y referencias: docs/parametros_andamio.md):
  giroide tipo lámina (sheet), celda 2,5 mm, pared 0,3 mm
  -> porosidad ~62 %, poro (espesor local) ~0,75 mm, pared ~0,3 mm
  Coincide con el andamio giroide-lámina "poro pequeño" medido por Kelly et al.
  (2021) en defecto femoral crítico (poro 739 µm, porosidad 62,8 %, pared 300 µm)
  y con los rangos de porosidad 50–80 % y poro 300–800 µm de la literatura.

Uso (desde la raíz del repositorio):
  .\\.venv_nnunet\\Scripts\\python.exe src\\scaffold\\generate_scaffold.py --defect <mascara.nii.gz> --out <carpeta>
      [--cell 2.5] [--wall 0.3 | --porosity 0.62] [--type gyroid|diamond|primitive]
      [--mode sheet|network] [--clearance 0.0] [--h 0.1] [--gt <defecto_verdadero.nii.gz>]
      F4 (opcionales; sin ellos el andamio es idéntico al de antes):
      [--material Ti6Al4V_LPBF|PCL_bTCP_extrusion] [--no-fab]
      [--bone-raw <segmentacion_cruda.nii.gz> --plug 8 [--plug-clearance 0.0]]
      [--ct <roi_ct.nii.gz> --nail auto [--nail-diameter 10] [--nail-clearance 0.5] [--hu-metal 2000]]
      [--grad t_ext t_int p1 p2]      (gradiente radial de pared, mm; solo sheet)

Salidas en --out:
  <nombre>_scaffold.stl         andamio (malla fina, h_malla 0,15 mm)
  <nombre>_scaffold_light.stl   andamio liviano (h_malla ~0,25 mm) para CAD/FEA
  <nombre>_envelope.stl         envolvente externa del andamio
  <nombre>_scaffold_report.json parámetros + métricas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "reconstruction"))
import scaffold_core as sc          # noqa: E402
import scaffold_f4 as f4            # noqa: E402

H_MESH = 0.15


def read_mask(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    sp = np.array(img.GetSpacing()[::-1], float)
    return arr, sp, img


def write_stl(path, v, f, name):
    import reconstruct_tibia as rt
    rt.write_stl(path, v, f, name)


def surface_distances(va, vb):
    import reconstruct_tibia as rt
    return rt.surface_distances(va, vb)


def make_field(defect, sp, h, params, extras=None):
    """Campo implícito con las opciones F4 (gradiente radial y canal para clavo)."""
    ex = extras or {}
    F = f4.build_field(defect, sp, h, params["cell_mm"], params["kind"], params["mode"], params["w"],
                       params["clearance_mm"], grad=ex.get("grad"))
    if ex.get("nail"):
        F["canal"] = f4.apply_channel(F, sp, ex["nail"], ex.get("nail_clearance", 0.5), ex.get("nail_diameter"))
    return F


def mesh_at(defect, sp, img, params, h_mesh, use_env=False, solid_keep=None, extras=None):
    F = make_field(defect, sp, h_mesh, params, extras)
    if use_env:
        f = F["env"]
    else:
        f = F["f"]
        if solid_keep is not None:
            S, _ = sc.clean_fragments(f <= 0)
            f = sc.apply_mask_to_field(F, S)
    return sc.field_to_mesh(f, F, sp, img.GetOrigin(), img.GetDirection())


def generate(defect_path, out_dir, cell=2.5, wall=0.3, porosity=None, kind="gyroid", mode="sheet",
             clearance=0.0, h=0.1, gt_path=None, name=None, meshes="all",
             material="Ti6Al4V_LPBF", fab=True, bone_raw_path=None, plug_mm=0.0, plug_clearance=0.0,
             ct_path=None, nail="none", nail_diameter=None, nail_clearance=0.5, hu_metal=2000.0, grad=None,
             fab_h=None, defect_refine=None, refine_reference=None, refine_hu=600.0, refine_bone=None,
             refine_mass=None, refine_metal_excl=1.0):
    """meshes: 'all' (fina + liviana + envolvente), 'light' (liviana + envolvente) o 'none'.
    Opciones F4 (ver scaffold_f4.py): material, fab (chequeo de fabricabilidad), tapón
    intramedular (bone_raw_path + plug_mm), canal para clavo (ct_path + nail='auto' o
    nail_diameter) y gradiente radial grad=dict(t_ext, t_int, p1, p2).
    fab_h (mm): chequeo de fabricabilidad con el campo reconstruido a ese paso.
    defect_refine ('sector', 'voxel' o 'sector+voxel'): corrige el defecto con la TC (ct_path) antes
    de construir el andamio (src/defect/defect_refine.py); 'sector' necesita refine_reference.
    refine_bone (hueso segmentado): solo cuenta la densidad contigua al huésped; refine_mass (mm): quita
    masas densas de grosor > 2·r; refine_metal_excl (mm): exclusión alrededor del metal.
    La máscara corregida se guarda en out_dir/defecto_corregido.nii.gz."""
    t0 = time.time()
    defect, sp, img = read_mask(defect_path)
    if not defect.any():
        raise ValueError(f"{defect_path}: la máscara del defecto está vacía")
    name = name or Path(defect_path).name.replace(".nii.gz", "").replace("_defect_est", "").replace("_defect", "")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    refine_rep = None
    if defect_refine:
        import SimpleITK as sitk
        sys.path.insert(0, str(ROOT / "src" / "defect"))
        import defect_core as dc
        import defect_refine as dr
        if ct_path is None:
            raise ValueError("--defect-refine necesita --ct (TC de la ROI en HU, misma grilla)")
        ref_mask = read_mask(refine_reference)[0] if refine_reference else None
        ct_r = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        bone_r = read_mask(refine_bone)[0] if refine_bone else None
        defect, refine_rep = dr.refine(defect, ct_r, sp, defect_refine, ref_mask, refine_hu, bone_r, refine_mass,
                                       refine_metal_excl)
        del ct_r, ref_mask, bone_r
        if not defect.any():
            raise ValueError("La corrección dejó el defecto vacío")
        dc.write_like(defect, img, out_dir / "defecto_corregido.nii.gz")
        defect_path = out_dir / "defecto_corregido.nii.gz"

    w = sc.w_for_porosity(kind, mode, porosity) if porosity is not None else sc.w_for_wall(kind, mode, cell, wall)
    params = dict(cell_mm=cell, kind=kind, mode=mode, w=w, clearance_mm=clearance)
    if material not in f4.MATERIALS:
        raise ValueError(f"material desconocido: {material} (opciones: {list(f4.MATERIALS)})")

    # 0) F4: envolvente (tapón intramedular) y clavo
    f4rep = dict(material=dict(clave=material, **f4.MATERIALS[material]))
    if refine_rep:
        f4rep["correccion_defecto"] = refine_rep
    env_mask = defect
    if plug_mm and plug_mm > 0:
        if bone_raw_path is None:
            raise ValueError("El tapón intramedular necesita --bone-raw (segmentación cruda, canal sin rellenar)")
        bone_raw, sp_b, _ = read_mask(bone_raw_path)
        if bone_raw.shape != defect.shape or not np.allclose(sp_b, sp):
            raise RuntimeError("La segmentación cruda no está en la grilla del defecto")
        plug, f4rep["tapon"] = f4.intramedullary_plugs(defect, bone_raw & ~defect, sp, plug_mm, plug_clearance)
        env_mask = defect | plug
        f4rep["tapon"]["volumen_mm3"] = float(plug.sum() * np.prod(sp))
        del bone_raw
    extras = dict(grad=grad, nail_clearance=nail_clearance, nail_diameter=nail_diameter)
    if nail == "auto" or nail_diameter:
        if ct_path is None:
            raise ValueError("El canal para clavo necesita --ct (TC de la ROI en HU, misma grilla)")
        import SimpleITK as sitk
        cimg = sitk.ReadImage(str(ct_path))
        ct = sitk.GetArrayFromImage(cimg).astype(np.float32)
        if ct.shape != defect.shape:
            raise RuntimeError("La TC no está en la grilla del defecto")
        nl = f4.detect_nail(ct, defect, sp, hu_metal=hu_metal)
        del ct
        f4rep["clavo"] = nl if nl else dict(detectado=False)
        extras["nail"] = nl
    if extras.get("nail") and "tapon" in f4rep and f4rep["tapon"].get("activo"):
        # Con clavo, el canal está ocupado (y en la práctica fresado al diámetro del clavo):
        # si el anillo que queda para el tapón es más delgado que una celda, el tapón sería
        # solo astillas de retícula. Se desactiva y se deja constancia.
        r_canal = (nail_diameter / 2 if nail_diameter else extras["nail"]["radio_mm"]) + nail_clearance
        r_plug = min(f4rep["tapon"][s]["diametro_equivalente_mm"] for s in ("proximal", "distal")) / 2
        if r_plug - r_canal < cell:
            env_mask = defect
            f4rep["tapon"].update(activo=False, motivo=f"anillo canal-clavo de {r_plug - r_canal:.1f} mm < celda {cell} mm")
    if grad is not None:
        f4rep["gradiente_radial"] = dict(grad)

    # 1) campo fino para métricas
    F = make_field(env_mask, sp, h, params, extras)
    if "canal" in F:
        f4rep["canal_clavo"] = F["canal"]
    solid, frag = sc.clean_fragments(sc.solid_mask(F))
    dfine = sc.defect_on_fine_grid(defect, sp, F)
    met = sc.architecture_metrics(F, solid, cell, dfine)
    n_sol = solid.sum()
    frag["vol_quitado_mm3"] = float(frag["removed_frac"] / (1 - frag["removed_frac"]) * n_sol * np.prod(F["h_axes"]))
    frag["vol_quitado_mayor_mm3"] = float(frag["vox_quitados_mayor"] * np.prod(F["h_axes"]))
    met["fragmentos"] = frag
    if env_mask is not defect or extras.get("nail"):
        env_base = sc.defect_on_fine_grid(defect, sp, F, offset_mm=clearance)
        met["coincidencia_geometrica_cuerpo"] = sc.jaccard(env_base & (F["env"] <= 0), dfine)
        del env_base
    met.update(f4.interconnectivity(F, solid))
    lat, core = sc.lattice_cube(F["params"], float(np.mean(F["h_axes"])))
    met["superficie_especifica_ideal_mm-1"] = _lattice_specific_surface(lat, float(np.mean(F["h_axes"])))
    if fab:
        fkw = dict(fab_h=fab_h, poro_mm=met["poro_mm"]["mean"]) if fab_h else {}
        f4rep["fabricabilidad"] = f4.fabricability(F, solid, material, frag.get("n_kept", frag["n_components"]), **fkw)
    del F, solid, dfine, lat, core

    # 2) mallas
    if meshes == "all":
        v, f = mesh_at(env_mask, sp, img, params, H_MESH, solid_keep=True, extras=extras)
        write_stl(out_dir / f"{name}_scaffold.stl", v, f, f"{name}_scaffold")
        met["malla"] = dict(caras=int(len(f)), volumen_mm3=abs(sc._signed_volume(v, f)), h_malla_mm=H_MESH,
                            area_mm2=f4.mesh_area(v, f))
        met["superficie_especifica_mm-1"] = met["malla"]["area_mm2"] / met["V_envolvente_mm3"]
        del v, f
    h_light = float(min(0.25, 0.8 * wall)) if porosity is None else 0.25
    if meshes in ("all", "light"):
        vl, fl = mesh_at(env_mask, sp, img, params, h_light, solid_keep=True, extras=extras)
        write_stl(out_dir / f"{name}_scaffold_light.stl", vl, fl, f"{name}_scaffold_light")
        met["malla_liviana"] = dict(caras=int(len(fl)), volumen_mm3=abs(sc._signed_volume(vl, fl)), h_malla_mm=h_light)
        del vl, fl
    ve, fe = mesh_at(env_mask, sp, img, params, 0.25, use_env=True, extras=extras)
    if meshes != "none":
        write_stl(out_dir / f"{name}_envelope.stl", ve, fe, f"{name}_envelope")

    # 3) distancia superficial envolvente <-> defecto
    p0 = dict(params, clearance_mm=0.0)
    vd, fd = mesh_at(defect, sp, img, p0, 0.25, use_env=True)
    met["distancia_superficial_envolvente_defecto_mm"] = surface_distances(ve, vd)

    # 4) contra el defecto verdadero (solo defectos virtuales)
    if gt_path is not None:
        gt, sp_gt, _ = read_mask(gt_path)
        if gt.shape != defect.shape or not np.allclose(sp_gt, sp):
            raise RuntimeError("El defecto verdadero no está en la misma grilla que el estimado")
        lo = np.minimum(np.argwhere(gt).min(0), np.argwhere(defect).min(0))
        hi = np.maximum(np.argwhere(gt).max(0), np.argwhere(defect).max(0))
        union = np.zeros_like(defect); union[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
        F2 = sc.build_scaffold_field(union, sp, h=h, **params)        # misma caja para ambos
        env_e = sc.defect_on_fine_grid(defect, sp, F2, offset_mm=clearance)
        env_g = sc.defect_on_fine_grid(gt, sp, F2)
        vox = float(np.prod(F2["h_axes"]))
        met["vs_defecto_verdadero"] = dict(
            V_verdadero_mm3=float(env_g.sum() * vox), V_envolvente_mm3=float(env_e.sum() * vox),
            ocupacion_pct=100.0 * float(env_e.sum()) / float(env_g.sum()),
            coincidencia_geometrica=sc.jaccard(env_e, env_g),
            relleno_del_defecto_verdadero_pct=100.0 * float((env_e & env_g).sum()) / float(env_g.sum()))
        vg, fg = mesh_at(gt, sp, img, p0, 0.25, use_env=True)
        met["vs_defecto_verdadero"]["distancia_superficial_mm"] = surface_distances(ve, vg)

    report = dict(defecto=str(defect_path), parametros=dict(tipo=kind, modo=mode, celda_mm=cell,
                  pared_objetivo_mm=None if porosity is not None else wall,
                  porosidad_objetivo=porosity, umbral_w=w, holgura_mm=clearance, h_metricas_mm=h),
                  metricas=met, f4=f4rep, tiempo_s=round(time.time() - t0, 1))
    with open(out_dir / f"{name}_scaffold_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    return report


def _lattice_specific_surface(lat, h):
    """Área/volumen de la retícula ideal (mm^-1) en el núcleo periódico del cubo (marching cubes)."""
    from skimage.measure import marching_cubes
    v, fc, _, _ = marching_cubes(lat.astype(np.float32), level=0.5, spacing=(h, h, h))
    return f4.mesh_area(v, fc) / (np.prod(lat.shape) * h ** 3)


def summary_lines(rep):
    m = rep["metricas"]
    L = [f"porosidad {m['porosidad_pct']:.1f} %   poro (espesor local) media {m['poro_mm']['mean']:.2f} mm "
         f"[p10–p90 {m['poro_mm']['p10']:.2f}–{m['poro_mm']['p90']:.2f}]   pared media {m['pared_mm']['mean']:.2f} mm",
         f"ocupación {m['ocupacion_pct']:.1f} %   coincidencia geométrica {m['coincidencia_geometrica']:.3f}   "
         f"dist. sup. media {m['distancia_superficial_envolvente_defecto_mm']['mean_mm']:.3f} mm",
         (f"malla {m['malla']['caras']:,} caras   " if "malla" in m else "")
         + (f"liviana {m['malla_liviana']['caras']:,} caras   " if "malla_liviana" in m else "")
         + f"fragmentos eliminados {m['fragmentos'].get('removed_components', 0)} "
         f"({100 * m['fragmentos'].get('removed_frac', 0):.2f} % vol.)   tiempo {rep['tiempo_s']:.0f} s"]
    fr = m["fragmentos"]
    if fr.get("n_relevantes_antes", 1) > 1:
        L.append(f"una pieza: {fr['n_relevantes_antes']} componentes sólidas relevantes -> se conserva la mayor; "
                 f"quitado {fr.get('vol_quitado_mayor_mm3', 0):.1f} mm3 ({100 * fr.get('frac_quitada_mayor', 0):.2f} % del sólido)")
    if fr.get("aviso_qc"):
        L.append(f"AVISO QC: se quitó {100 * fr['removed_frac']:.1f} % del sólido (> 5 %) al dejar una sola pieza; revisar el defecto")
    F4 = rep.get("f4", {})
    if "interconectividad_pct" in m:
        L.append(f"interconectividad {m['interconectividad_pct']:.1f} %   poros cerrados {m.get('poros_cerrados_n', 0)}   "
                 f"superficie específica (retícula ideal) {m['superficie_especifica_ideal_mm-1']:.2f} mm^-1")
    if "fabricabilidad" in F4:
        fb = F4["fabricabilidad"]
        L.append(f"fabricabilidad [{fb['material']}]: pared real p5 {fb['pared_real']['p5']:.2f} mm "
                 f"(mín. {fb['pared_min_material_mm']}), poro real p5 {fb['poro_real']['p5']:.2f} mm (mín. {fb['poro_min_mm']}), "
                 f"componentes {fb['componentes_solidas']} -> {'OK' if fb['fabricable'] else 'NO CUMPLE' if fb['fabricable'] is False else 'sin criterio'}")
    if "tapon" in F4 and not F4["tapon"].get("activo"):
        L.append(f"tapón intramedular desactivado: {F4['tapon'].get('motivo', '')}")
    elif "tapon" in F4:
        t = F4["tapon"]
        L.append(f"tapón intramedular: prox {t['proximal']['longitud_real_mm']:.1f} mm (Ø {t['proximal']['diametro_equivalente_mm']:.1f}), "
                 f"dist {t['distal']['longitud_real_mm']:.1f} mm (Ø {t['distal']['diametro_equivalente_mm']:.1f})")
    if "canal_clavo" in F4:
        L.append(f"canal para clavo: radio {F4['canal_clavo']['radio_canal_mm']:.2f} mm ({F4['canal_clavo']['diametro_origen']})")
    elif F4.get("clavo", {}).get("detectado") is False:
        L.append("clavo: no se detectó metal en el tramo del defecto")
    if "vs_defecto_verdadero" in m:
        g = m["vs_defecto_verdadero"]
        L.append(f"vs. defecto verdadero: ocupación {g['ocupacion_pct']:.1f} %   CG {g['coincidencia_geometrica']:.3f}   "
                 f"relleno {g['relleno_del_defecto_verdadero_pct']:.1f} %   dist. sup. media "
                 f"{g['distancia_superficial_mm']['mean_mm']:.2f} mm (p95 {g['distancia_superficial_mm']['p95_mm']:.2f})")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defect", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=2.5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--wall", type=float, default=0.3)
    g.add_argument("--porosity", type=float, default=None)
    ap.add_argument("--type", default="gyroid", choices=list(sc.TPMS))
    ap.add_argument("--mode", default="sheet", choices=["sheet", "network"])
    ap.add_argument("--clearance", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--material", default="Ti6Al4V_LPBF", choices=list(f4.MATERIALS))
    ap.add_argument("--no-fab", action="store_true", help="no correr el chequeo de fabricabilidad")
    ap.add_argument("--bone-raw", default=None, help="segmentación cruda (canal sin rellenar) en la grilla del defecto")
    ap.add_argument("--plug", type=float, default=0.0, help="longitud del tapón intramedular (mm); 0 = sin tapón")
    ap.add_argument("--plug-clearance", type=float, default=0.0)
    ap.add_argument("--ct", default=None, help="TC de la ROI (HU) en la grilla del defecto")
    ap.add_argument("--nail", default="none", choices=["none", "auto"])
    ap.add_argument("--nail-diameter", type=float, default=None)
    ap.add_argument("--nail-clearance", type=float, default=0.5)
    ap.add_argument("--hu-metal", type=float, default=2000.0)
    ap.add_argument("--grad", type=float, nargs=4, default=None, metavar=("T_EXT", "T_INT", "P1", "P2"))
    ap.add_argument("--fab-h", type=float, default=None,
                    help="paso fino (mm) del chequeo de fabricabilidad, p. ej. 0.025; sin él, se mide en la grilla --h")
    ap.add_argument("--defect-refine", default=None, choices=["sector", "voxel", "sector+voxel"],
                    help="corrige el defecto con la TC antes del andamio (requiere --ct); 'sector' requiere --refine-reference")
    ap.add_argument("--refine-reference", default=None, help="forma de referencia del sistema (referencia.nii.gz)")
    ap.add_argument("--refine-hu", type=float, default=600.0, help="umbral HU de la corrección (fijado a priori en 600)")
    ap.add_argument("--refine-bone", default=None, help="hueso segmentado (tibia_segmentada.nii.gz): solo densidad contigua al huésped")
    ap.add_argument("--refine-mass", type=float, default=None, help="radio (mm): quita masas densas más gruesas que 2·r (espaciador)")
    ap.add_argument("--refine-metal-excl", type=float, default=1.0, help="exclusión alrededor del metal (mm)")
    a = ap.parse_args()
    grad = dict(zip(("t_ext", "t_int", "p1", "p2"), a.grad)) if a.grad else None
    rep = generate(a.defect, a.out, a.cell, a.wall, a.porosity, a.type, a.mode, a.clearance, a.h, a.gt,
                   material=a.material, fab=not a.no_fab, bone_raw_path=a.bone_raw, plug_mm=a.plug,
                   plug_clearance=a.plug_clearance, ct_path=a.ct, nail=a.nail, nail_diameter=a.nail_diameter,
                   nail_clearance=a.nail_clearance, hu_metal=a.hu_metal, grad=grad, fab_h=a.fab_h,
                   defect_refine=a.defect_refine, refine_reference=a.refine_reference, refine_hu=a.refine_hu,
                   refine_bone=a.refine_bone, refine_mass=a.refine_mass, refine_metal_excl=a.refine_metal_excl)
    print("\n".join(summary_lines(rep)))
    print(f"Salida: {a.out}")


if __name__ == "__main__":
    main()
