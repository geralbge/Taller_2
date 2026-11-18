"""
F6. Módulo scripted de 3D Slicer que ENVUELVE el pipeline del prototipo.

No reimplementa ni modifica nada del pipeline: arma los archivos de entrada que
espera src/pipeline/run_case.py, lo ejecuta como proceso externo con el Python de
.venv_nnunet (torch / nnU-Net no se instalan en el Python de Slicer), muestra el
registro en vivo y, al terminar, carga en la escena los resultados que el script
escribe en la carpeta de salida.

Entradas que prepara (las mismas que run_case.py documenta):
  <salida>/clicks/Tibia_run1.mrk.json         P1 (proximal), P2 (distal), tibia afectada
  <salida>/clicks/Defect_run1.mrk.json        D1, D2: hueso sano justo encima/debajo del defecto
  <salida>/clicks/Tibia_contra_run1.mrk.json  (opcional) P1, P2 sobre la tibia contralateral
  TC: el archivo NIfTI de origen si el volumen se cargó de uno y no se modificó;
      si no, el volumen se exporta a <salida>/ct_slicer.nii.gz (geometría de Slicer, LPS en disco).

Ejecución en dos pasos (ambos son scripts existentes, sin cambios):
  1) run_case.py --no-scaffold     TC -> ROI -> segmentación -> reconstrucción -> defecto
  2) generate_scaffold.py          andamio sobre <salida>/work/andamio.nii.gz (copia de defecto.nii.gz,
                                   así las salidas se llaman andamio_*), con las opciones que run_case.py
                                   no expone: chequeo fino de fabricabilidad (--fab-h) y la corrección
                                   opcional del defecto por densidad (--defect-refine, modo declarado).
  Con las mismas opciones, el paso 2 hace la misma llamada a generate() que run_case.py.

Material y etapa: el material va a generate_scaffold.py (--material). La etapa de degradación NO
cambia la geometría: solo define las propiedades del sólido para el FEA (E y fluencia por los
factores de scaffold_f4.MATERIALS[...]["etapas_degradacion"]); se guardan en propiedades_fea.json.

Salidas que carga (nombres fijados por run_case.py y generate_scaffold.py):
  tibia.stl, defecto.stl, referencia.stl, andamio_scaffold_light.stl,
  andamio_scaffold.stl, andamio_envelope.stl (STL en LPS mm, cabecera SPACE=LPS)
  tibia_segmentada.nii.gz, defecto.nii.gz, defecto_corregido.nii.gz (máscaras, grilla de la ROI orientada)
  reporte.json, andamio_scaffold_report.json

Caché: run_case.py reutiliza work/roi_*.nii.gz y work/pred/*.nii.gz si existen.
Si cambian la TC, los clics P1/P2 (o los contralaterales), el modelo, los folds o
TTA, el módulo borra esos intermedios antes de ejecutar, para que no se usen
segmentaciones de una configuración anterior. Si solo cambian D1/D2 o los
parámetros del andamio, la segmentación se reutiliza y la corrida es mucho más corta.

Prototipo computacional de investigación: sin uso clínico.
"""

from __future__ import annotations

import csv
import hashlib
import json
import ast
import os
import shutil
from pathlib import Path

import numpy as np
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

DEFAULT_ROOT = str(Path(__file__).resolve().parents[2])  # raíz del repositorio
SETTINGS = "AndamioTibial/"
CLICK_FILES = {"tibia": "Tibia_run1.mrk.json", "defect": "Defect_run1.mrk.json",
               "contra": "Tibia_contra_run1.mrk.json"}
# Celda / pared de diseño por material (decisión del proyecto, hallazgos 06/11/2026):
# Ti 2,5 / 0,30 mm; PCL/β-TCP 4,2 / 0,50 mm (pared sobre el mínimo fabricable y porosidad ~64 %).
DESIGN_DEFAULTS = {"Ti6Al4V_LPBF": (2.5, 0.30), "PCL_bTCP_extrusion": (4.2, 0.50)}
FAB_H_DEFAULT = 0.025                           # paso del chequeo fino usado en los resultados finales
# Corrección por densidad en el contorno: variante CONGELADA en virtuales y aplicada así a los pacientes
# (tanda_corregidos.py). No se exponen sus parámetros para no ajustarlos caso a caso.
REFINE_FROZEN = dict(rule="sector", hu=600, mass_mm=3, metal_excl_mm=3)
REFINE_MODES = [("no", "No (salida automática)"),
                ("aviso", "Solo si salta el aviso de densidad (compuerta)"),
                ("siempre", "Siempre")]
SCAFFOLD_NAME = "andamio"                       # mismo nombre que usa run_case.py
RESULT_FILES = ["tibia.stl", "tibia_cad.stl", "tibia_segmentada.nii.gz", "defecto.nii.gz",
                "referencia.nii.gz", "defecto.stl", "referencia.stl", "reporte.json",
                "defecto_corregido.nii.gz", "propiedades_fea.json", "slicer_corrida.json",
                f"{SCAFFOLD_NAME}_scaffold.stl", f"{SCAFFOLD_NAME}_scaffold_light.stl",
                f"{SCAFFOLD_NAME}_envelope.stl", f"{SCAFFOLD_NAME}_scaffold_report.json"]
SIGNATURE_FILE = "_slicer_firma.json"

# (etiqueta, ruta en reporte.json, formato). Las claves salen de run_case.py,
# defect_core.characterize y generate_scaffold.generate; si una falta, la fila se omite.
MAIN_METRICS = [
    ("Segmentación", "Modelo", "modelo", "{}"),
    ("Segmentación", "Dispositivo", "dispositivo", "{}"),
    ("Reconstrucción", "Volumen tibia (cm³)", "tibia.volumen_cm3", "{:.1f}"),
    ("Reconstrucción", "Fragmentos de la tibia", "tibia.fragmentos", "{}"),
    ("Reconstrucción", "Malla tibia cerrada (watertight)", "tibia.malla.watertight", "{}"),
    ("Defecto", "Referencia usada", "referencia_usada", "{}"),
    ("Defecto", "RMS registro contralateral (mm)", "registro_contralateral.rms_mm", "{:.2f}"),
    ("Defecto", "Dice interpolación vs contralateral", "dice_referencias_interp_vs_contra", "{:.3f}"),
    ("Defecto", "Defecto detectado", "defecto.defect_detected", "{}"),
    ("Defecto", "Longitud (mm)", "defecto.length_mm", "{:.1f}"),
    ("Defecto", "Volumen (mm³)", "defecto.volume_mm3", "{:.0f}"),
    ("Defecto", "Pérdida cortical máx. (%)", "defecto.cortical_loss_max_pct", "{:.0f}"),
    ("Defecto", "Pérdida cortical media (%)", "defecto.cortical_loss_mean_pct", "{:.0f}"),
    ("Defecto", "Segmentos", "defecto.n_segments", "{}"),
    ("Defecto", "Diafisario", "defecto.diaphyseal", "{}"),
    ("Defecto", "Cumple criterios del proyecto", "defecto.meets_criteria", "{}"),
    ("Defecto", "Aviso de control de calidad", "defecto.qc_warning", "{}"),
    ("Defecto", "Aviso de densidad en el defecto", "defecto.qc_density_warning", "{}"),
    ("Corrección del defecto", "Aplicada", "_slicer.correccion_aplicada", "{}"),
    ("Corrección del defecto", "Motivo", "_slicer.correccion_motivo", "{}"),
    ("Corrección del defecto", "Volumen inicial (mm³)", "andamio.f4.correccion_defecto.V_inicial_mm3", "{:.0f}"),
    ("Corrección del defecto", "Volumen quitado por sectores (mm³)", "andamio.f4.correccion_defecto.V_quitado_sector_mm3", "{:.0f}"),
    ("Corrección del defecto", "Volumen final (mm³)", "andamio.f4.correccion_defecto.V_final_mm3", "{:.0f}"),
    ("Andamio", "Material", "andamio.f4.material.clave", "{}"),
    ("Andamio", "Celda (mm)", "andamio.parametros.celda_mm", "{}"),
    ("Andamio", "Pared objetivo (mm)", "andamio.parametros.pared_objetivo_mm", "{}"),
    ("Andamio", "Porosidad (%)", "andamio.metricas.porosidad_pct", "{:.1f}"),
    ("Andamio", "Poro medio (mm)", "andamio.metricas.poro_mm.mean", "{:.2f}"),
    ("Andamio", "Pared media (mm)", "andamio.metricas.pared_mm.mean", "{:.2f}"),
    ("Andamio", "Interconectividad (%)", "andamio.metricas.interconectividad_pct", "{:.1f}"),
    ("Andamio", "Superficie específica (mm⁻¹)", "andamio.metricas.superficie_especifica_mm-1", "{:.2f}"),
    ("Verificación geométrica", "Ocupación (%)", "andamio.metricas.ocupacion_pct", "{:.1f}"),
    ("Verificación geométrica", "Coincidencia geométrica", "andamio.metricas.coincidencia_geometrica", "{:.3f}"),
    ("Verificación geométrica", "Dist. superficial media envolvente–defecto (mm)",
     "andamio.metricas.distancia_superficial_envolvente_defecto_mm.mean_mm", "{:.3f}"),
    ("Fabricabilidad", "Fabricable", "andamio.f4.fabricabilidad.fabricable", "{}"),
    ("Fabricabilidad", "Chequeo fino, paso (mm)", "andamio.f4.fabricabilidad.medicion_fina.h_fina_mm", "{}"),
    ("Fabricabilidad", "Pared real p1 (mm)", "andamio.f4.fabricabilidad.pared_real.p1", "{:.3f}"),
    ("Fabricabilidad", "Pared real p5 (mm)", "andamio.f4.fabricabilidad.pared_real.p5", "{:.2f}"),
    ("Fabricabilidad", "Pared mínima del material (mm)", "andamio.f4.fabricabilidad.pared_min_material_mm", "{}"),
    ("Fabricabilidad", "Poro real p5 (mm)", "andamio.f4.fabricabilidad.poro_real.p5", "{:.2f}"),
    ("Fabricabilidad", "Componentes sólidas", "andamio.f4.fabricabilidad.componentes_solidas", "{}"),
    ("Propiedades FEA", "Etapa", "_slicer.fea.etapa", "{}"),
    ("Propiedades FEA", "E del sólido (MPa)", "_slicer.fea.E_MPa", "{:.0f}"),
    ("Propiedades FEA", "ν", "_slicer.fea.nu", "{}"),
    ("Propiedades FEA", "Fluencia del sólido (MPa)", "_slicer.fea.fluencia_MPa", "{:.1f}"),
    ("Tiempo", "Segmentación + defecto (s)", "tiempo_total_s", "{:.0f}"),
    ("Tiempo", "Andamio (s)", "andamio.tiempo_s", "{:.0f}"),
]


# ============================================================================ módulo
class AndamioTibial(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Andamio tibial (prototipo)"
        parent.categories = ["Taller de Grado"]
        parent.dependencies = []
        parent.contributors = ["Geraldy B. Guzmán Espinoza (UCB, La Paz)"]
        parent.helpText = (
            "Interfaz del prototipo computacional TC → segmentación → defecto → andamio TPMS. "
            "Ejecuta src/pipeline/run_case.py con el Python de .venv_nnunet y carga sus resultados. "
            "Prototipo de investigación: no es un dispositivo clínico.")
        parent.acknowledgementText = "Trabajo de Grado II, Ingeniería Biomédica, Universidad Católica Boliviana “San Pablo”."


# ============================================================================ utilidades puras
def get_path(d, path):
    """Valor en un dict anidado por ruta 'a.b.c'; None si falta."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def flatten(d, prefix=""):
    """dict anidado -> lista [(ruta, valor)] con valores escalares."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d) and len(d) <= 8:
            out.append((prefix, d))
        else:
            for i, v in enumerate(d):
                out += flatten(v, f"{prefix}[{i}]")
    else:
        out.append((prefix, d))
    return out


def es(text):
    """Coma decimal para mostrar en pantalla (los archivos exportados no cambian)."""
    import re
    return re.sub(r"(?<=\d)\.(?=\d)", ",", str(text))


def case_name_from(volume_name):
    """Nombre del caso a partir del nombre de la TC: 'pat_02_CT_HU' -> 'pat_02'."""
    import re
    n = re.sub(r"(?i)([_ -]?(ct|tc)([_ -]?hu)?)$", "", (volume_name or "").strip())
    n = re.sub(r"[^\w-]+", "_", n).strip("_")
    return n or "caso_01"


def fmt(value, f="{}"):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sí" if value else "no"
    try:
        return f.format(value)
    except (ValueError, TypeError):
        return str(value)


def build_args(p):
    """Parámetros de la interfaz -> argumentos de run_case.py (solo opciones que el script define)."""
    a = [str(Path(p["root"]) / "src" / "pipeline" / "run_case.py"),
         "--ct", str(p["ct"]), "--clicks", str(p["clicks"]), "--out", str(p["out"]),
         "--model", p["model"], "--reference", p["reference"],
         "--cell", f"{p['cell']:g}", "--wall", f"{p['wall']:g}", "--material", p["material"]]
    a += ["--folds"] + [str(int(f)) for f in p["folds"]]
    if p.get("tta"):
        a.append("--tta")
    if p.get("no_scaffold"):
        a.append("--no-scaffold")
    if p.get("nail") == "auto":
        a += ["--nail", "auto"]
    if p.get("nail_diameter"):
        a += ["--nail-diameter", f"{p['nail_diameter']:g}"]
    if p.get("plug"):
        a += ["--plug", f"{p['plug']:g}"]
    if p.get("grad"):
        a += ["--grad"] + [f"{x:g}" for x in p["grad"]]
    return a


def build_scaffold_args(p):
    """Paso 2: argumentos de generate_scaffold.py (opciones que el script define en su argparse)."""
    out, work = Path(p["out"]), Path(p["out"]) / "work"
    a = [str(Path(p["root"]) / "src" / "scaffold" / "generate_scaffold.py"),
         "--defect", str(work / f"{SCAFFOLD_NAME}.nii.gz"), "--out", str(out),
         "--cell", f"{p['cell']:g}", "--wall", f"{p['wall']:g}", "--material", p["material"]]
    if p.get("fab_h"):
        a += ["--fab-h", f"{p['fab_h']:g}"]
    nail = p.get("nail") == "auto" or bool(p.get("nail_diameter"))
    if nail or p.get("refine"):
        a += ["--ct", str(work / "roi_0000.nii.gz")]          # TC de la ROI en HU (misma grilla)
    if p.get("nail") == "auto":
        a += ["--nail", "auto"]
    if p.get("nail_diameter"):
        a += ["--nail-diameter", f"{p['nail_diameter']:g}"]
    if p.get("plug"):
        a += ["--bone-raw", str(work / "pred" / "roi.nii.gz"), "--plug", f"{p['plug']:g}"]
    if p.get("grad"):
        a += ["--grad"] + [f"{x:g}" for x in p["grad"]]
    if p.get("refine"):
        f = REFINE_FROZEN
        a += ["--defect-refine", f["rule"], "--refine-hu", f"{f['hu']:g}",
              "--refine-reference", str(out / "referencia.nii.gz"),
              "--refine-bone", str(out / "tibia_segmentada.nii.gz"),
              "--refine-mass", f"{f['mass_mm']:g}", "--refine-metal-excl", f"{f['metal_excl_mm']:g}"]
    return a


def refine_decision(mode, report):
    """(aplicar, motivo) según el modo elegido y el aviso de densidad de defect_core."""
    warn = bool(get_path(report or {}, "defecto.qc_density_warning"))
    if mode == "siempre":
        return True, "modo 'siempre'"
    if mode == "aviso":
        return (True, "saltó el aviso de densidad") if warn else (False, "sin aviso de densidad: no se corrige")
    return False, "modo desactivado"


def load_materials(root):
    """Lee el dict MATERIALS de src/scaffold/scaffold_f4.py sin importarlo (el Python de Slicer no
    tiene sus dependencias). Solo se evalúa esa asignación, con dict y round como únicos nombres."""
    path = Path(root) / "src" / "scaffold" / "scaffold_f4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MATERIALS" for t in node.targets):
            code = compile(ast.Expression(node.value), str(path), "eval")
            return eval(code, {"__builtins__": {}, "dict": dict, "round": round})  # noqa: S307
    raise RuntimeError(f"No se encontró MATERIALS en {path}")


def stages(material_dict):
    """Etapas de degradación del material; si no tiene, solo t0."""
    return material_dict.get("etapas_degradacion") or [dict(etapa="t0", meses=0, factor_E=1.0,
                                                            factor_fluencia=1.0, fuente="propiedades iniciales")]


def stage_properties(key, material_dict, etapa):
    """Propiedades del sólido para el FEA en la etapa. Un factor None -> propiedad sin dato (no se inventa)."""
    st = next((s for s in stages(material_dict) if s["etapa"] == etapa), None)
    if st is None:
        raise ValueError(f"Etapa {etapa} no definida para {key}")
    mul = lambda v, f: None if v is None or f is None else float(v) * float(f)
    return dict(material=key, nombre=material_dict.get("nombre"), etapa=etapa, meses=st.get("meses"),
                E_MPa=mul(material_dict.get("E_MPa"), st.get("factor_E")), nu=material_dict.get("nu"),
                fluencia_MPa=mul(material_dict.get("fluencia_MPa"), st.get("factor_fluencia")),
                factor_E=st.get("factor_E"), factor_fluencia=st.get("factor_fluencia"), fuente_etapa=st.get("fuente"),
                E_inicial_MPa=material_dict.get("E_MPa"), fluencia_inicial_MPa=material_dict.get("fluencia_MPa"),
                supuesto="factores medidos sobre el andamio aplicados al material macizo (supuesto del proyecto)"
                if material_dict.get("etapas_degradacion") else None)


def parse_folds(text):
    folds = [int(t) for t in text.replace(",", " ").split()]
    if not folds:
        raise ValueError("Indica al menos un fold (p. ej. 0).")
    return folds


def segmentation_signature(ct_sig, tibia_pts, contra_pts, model, folds, tta):
    """Todo lo que determina work/roi_* y work/pred/*. D1/D2 y el andamio NO entran."""
    r = lambda pts: None if pts is None else [[round(float(x), 3) for x in q] for q in pts]
    return dict(ct=ct_sig, tibia=r(tibia_pts), contra=r(contra_pts), model=model,
                folds=[int(f) for f in folds], tta=bool(tta))


def invalidate_cache(out, new_sig, log=print):
    """Borra intermedios de segmentación si la firma cambió. Devuelve True si la firma cambió
    (o no había firma), False si es la misma configuración que la corrida anterior."""
    work = Path(out) / "work"
    sig_path = work / SIGNATURE_FILE
    old = None
    if sig_path.is_file():
        try:
            old = json.loads(sig_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
    has_cache = work.is_dir() and (any(work.glob("roi*.nii.gz")) or (work / "pred").is_dir())
    changed = old != new_sig
    if has_cache and changed:
        for f in list(work.glob("roi*.nii.gz")) + list(work.glob("roi*.json")) + [work / "ct_HU.nii.gz"]:
            if f.is_file():
                f.unlink()
        if (work / "pred").is_dir():
            shutil.rmtree(work / "pred")
        log("Configuración de segmentación distinta a la anterior: se borraron ROI y predicciones en caché.")
    elif has_cache:
        log("Misma TC, clics P1/P2 y modelo que la corrida anterior: se reutiliza la segmentación en caché.")
    work.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(new_sig, indent=1), encoding="utf-8")
    return changed


def clear_previous_results(out):
    """Evita cargar resultados de una corrida anterior si la nueva no los regenera."""
    for name in RESULT_FILES:
        f = Path(out) / name
        if f.is_file():
            f.unlink()


def clean_child_environment(env):
    """Entorno para el Python externo sin las variables que Slicer define para su propio Python."""
    env = dict(env)
    for k in list(env):
        if k.upper() in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE"):
            env.pop(k)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ============================================================================ visor de resultados ya calculados
# Muestra casos precalculados por etapa, variante y material, sin ejecutar nada (para demostración).
VIEW_STAGES = ["1. Hueso", "2. Defecto", "3. Envolvente", "4. Andamio"]
VIEW_MATERIALS = {"Ti-6Al-4V": "ti", "PCL/β-TCP": "pcl"}
VIEW_VARIANTS = {"Automático": "sis", "Corregido (opcional)": "siscorr", "Referencia manual": "ref"}


def viewer_cases(root):
    """{etiqueta: (tipo, carpeta)} de los casos disponibles en disco."""
    root = Path(root)
    out = {}
    base = root / "results" / "pathological" / "comparacion_materiales_v2"
    for pat in ("pat_01", "pat_02"):
        if any(base.glob(f"{pat}_*")):
            out[f"Paciente {pat}"] = ("paciente", pat)
    for d in sorted((root / "results" / "evaluation" / "scaffold" / "demostrativos_v2").glob("tipo_*")):
        out[f"Virtual tipo {d.name.split('_')[1]} ({d.name.split('__')[-1]})"] = ("virtual", str(d))
    return out


def viewer_folder(root, kind, ref, variant, mat):
    if kind == "virtual":
        return Path(ref) / mat
    return Path(root) / "results" / "pathological" / "comparacion_materiales_v2" / f"{ref}_{variant}_{mat}"


def viewer_items(root, kind, ref, variant, mat, stage):
    """Mallas de la etapa: lista de (ruta, color, opacidad, etiqueta) y avisos."""
    root = Path(root)
    folder = viewer_folder(root, kind, ref, variant, mat)
    first = lambda pat: next(iter(sorted(folder.glob(pat))), None)
    env, scaf = first("*_envelope.stl"), first("*_scaffold_light.stl")
    items, notes = [], []
    ft = root / "results" / "pathological" / f"{ref}_ft"
    if kind == "paciente" and (ft / "tibia.stl").is_file():
        items.append((ft / "tibia.stl", (0.89, 0.85, 0.79), 1.0 if stage == 0 else 0.3, "tibia"))
    elif kind == "virtual":
        notes.append("Caso virtual: no hay malla de tibia guardada; se muestra solo defecto y andamio.")
    if stage == 1:
        if kind == "paciente" and variant == "sis" and (ft / "defecto.stl").is_file():
            items.append((ft / "defecto.stl", (0.85, 0.25, 0.25), 0.7, "defecto"))
        elif env:
            items.append((env, (0.85, 0.25, 0.25), 0.7, "defecto"))
    if stage == 2 and env:
        items.append((env, (0.95, 0.75, 0.2), 0.6, "envolvente"))
    if stage == 3:
        if env:
            items.append((env, (0.95, 0.75, 0.2), 0.12, "envolvente"))
        if scaf:
            items.append((scaf, (0.2, 0.7, 0.55), 1.0, "andamio"))
        else:
            notes.append(f"Sin andamio en {folder}")
    return items, notes


def viewer_metrics(folder):
    """Resumen desde el reporte del andamio."""
    reps = sorted(Path(folder).glob("*_scaffold_report.json"))
    if not reps:
        return []
    r = json.loads(reps[0].read_text(encoding="utf-8"))
    m, fb = r["metricas"], r["f4"]["fabricabilidad"]
    rows = [f"Material: {fb['material']}", f"Celda / pared: {r['parametros']['celda_mm']} / {r['parametros']['pared_objetivo_mm']} mm",
            f"Porosidad: {m['porosidad_pct']:.1f} %", f"Pared mediana: {fb['pared_real']['mediana']:.3f} mm",
            f"Poro mediano: {fb['poro_real']['mediana']:.3f} mm", f"Fabricable: {'sí' if fb['fabricable'] else 'no'}"]
    vs = m.get("vs_defecto_verdadero")
    if vs:
        rows.append(f"Coincidencia con la referencia: {vs['coincidencia_geometrica']:.3f}")
    return rows


# ============================================================================ progreso por etapa
# Etapas reales: las 6 primeras las anuncia run_case.py con "[n] ..." en su salida; la 7 es el paso 2
# (generate_scaffold.py) y la 8 la carga en la escena. No se estiman porcentajes dentro de una etapa.
STAGES = [("Lectura de TC", "[1]"), ("ROI orientada", "[2]"), ("Segmentación nnU-Net", "[3]"),
          ("Reconstrucción 3D", "[4]"), ("Caracterización del defecto", "[5]"), ("Modelo 3D del defecto", "[6]"),
          ("Andamio TPMS y verificaciones", None), ("Carga de resultados", None)]
STAGE_ANDAMIO, STAGE_CARGA = 6, 7
ICON = {"pendiente": "○", "en curso": "◐", "hecho": "✓", "error": "✗", "omitido": "–"}


class StageTracker:
    """Estado de cada etapa (pendiente, en curso, hecho, error, omitido) con sus tiempos."""

    def __init__(self, now):
        self.state = ["pendiente"] * len(STAGES)
        self.t0 = [None] * len(STAGES)
        self.dt = [None] * len(STAGES)
        self.start_time = now

    def current(self):
        return next((i for i, st in enumerate(self.state) if st == "en curso"), None)

    def start(self, i, now):
        c = self.current()
        if c is not None and c != i:
            self.finish(c, now)
        for j in range(i):                             # etapas que el pipeline no anunció (p. ej. sin defecto)
            if self.state[j] == "pendiente":
                self.state[j] = "omitido"
        self.state[i] = "en curso"; self.t0[i] = now

    def finish(self, i, now, state="hecho"):
        if self.state[i] == "en curso" and self.t0[i] is not None:
            self.dt[i] = now - self.t0[i]
        self.state[i] = state

    def fail(self, now):
        c = self.current()
        if c is not None:
            self.finish(c, now, "error")

    def skip_rest(self):
        for j, st in enumerate(self.state):
            if st == "pendiente":
                self.state[j] = "omitido"

    def feed_line(self, line, now):
        """Detecta el anuncio '[n] ...' de run_case.py."""
        t = line.strip()
        for i, (_, tag) in enumerate(STAGES):
            if tag and t.startswith(tag):
                self.start(i, now)
                return i
        return None

    def done_count(self):
        return sum(st in ("hecho", "omitido") for st in self.state)

    def lines(self, now):
        out = []
        for i, (name, _) in enumerate(STAGES):
            st = self.state[i]
            t = (now - self.t0[i]) if st == "en curso" and self.t0[i] is not None else self.dt[i]
            out.append(f"{ICON[st]}  {name}" + (f"  ({t:.0f} s)" if t is not None else "")
                       + ("  en curso" if st == "en curso" else ""))
        return out


# ============================================================================ control de calidad
def qc_items(rep, fea=None):
    """[(estado, texto)] solo con criterios que el proyecto ya definió. estado: ok, aviso, error, info.
    Valores sin umbral definido (p. ej. coincidencia geométrica) se informan como 'info', sin veredicto."""
    rep = rep or {}
    it = []
    fr = get_path(rep, "tibia.fragmentos")
    if fr is not None:
        it.append(("ok" if fr == 1 else "aviso", f"Segmentación: tibia en {fr} fragmento(s)"))
    det = get_path(rep, "defecto.defect_detected")
    if det is False:
        it.append(("error", "No se detectó defecto entre D1 y D2"))
    meets = get_path(rep, "defecto.meets_criteria")
    if meets is not None:
        it.append(("ok" if meets else "aviso", "Defecto " + ("cumple" if meets else "NO cumple")
                   + " los criterios del proyecto (2,5–7 cm, pérdida cortical > 50 %, un segmento, sin articulación)"))
    if get_path(rep, "defecto.qc_density_warning"):
        it.append(("aviso", "Aviso de densidad: posible sub-segmentación dentro del defecto"))
    elif get_path(rep, "defecto.qc_warning"):
        it.append(("aviso", "Aviso de control de calidad del defecto: revisar antes de usar el andamio"))
    ap = get_path(rep, "_slicer.correccion_aplicada")
    if ap is not None:
        it.append(("info", f"Corrección por densidad: {'aplicada' if ap else 'no aplicada'} "
                           f"({get_path(rep, '_slicer.correccion_motivo') or ''})"))
    cg = get_path(rep, "andamio.metricas.coincidencia_geometrica")
    if cg is not None:
        it.append(("info", f"Coincidencia geométrica envolvente–defecto: {cg:.3f} (sin umbral definido en el proyecto)"))
    vs = get_path(rep, "andamio.metricas.vs_defecto_verdadero.coincidencia_geometrica")
    if vs is not None:
        it.append(("info", f"Coincidencia con la referencia: {vs:.3f} (sin umbral definido en el proyecto)"))
    fb = get_path(rep, "andamio.f4.fabricabilidad") or {}
    ch = fb.get("chequeos") or {}
    if "pared" in ch:
        it.append(("ok" if ch["pared"] else "error", f"Pared real p1 {fmt(get_path(fb, 'pared_real.p1'), '{:.3f}')} mm "
                   f"vs. mínimo del material {fmt(fb.get('pared_min_material_mm'))} mm"))
    if "poro" in ch:
        it.append(("ok" if ch["poro"] else "error", f"Poro real p5 {fmt(get_path(fb, 'poro_real.p5'), '{:.3f}')} mm "
                   f"vs. mínimo {fmt(fb.get('poro_min_mm'))} mm"))
    if "una_componente" in ch:
        it.append(("ok" if ch["una_componente"] else "error",
                   f"Andamio en {fmt(fb.get('componentes_solidas'))} pieza(s) sólida(s)"))
    if fea:
        if fea.get("fluencia_MPa") is None:
            it.append(("info", f"FEA, etapa {fea.get('etapa')}: sin dato de fluencia (la fuente no lo reporta)"))
        if fea.get("supuesto"):
            it.append(("info", "FEA: " + fea["supuesto"]))
    return it


QC_ICON = {"ok": "✓", "aviso": "⚠", "error": "✗", "info": "ℹ"}
QC_COLOR = {"ok": "#2e7d32", "aviso": "#b26a00", "error": "#c62828", "info": "#455a64"}


def qc_html(items):
    if not items:
        return "<i>Sin resultados todavía.</i>"
    return "<br>".join(f'<span style="color:{QC_COLOR[e]}">{QC_ICON[e]}</span> {t}' for e, t in items)


# ============================================================================ vista 3D y cortes (Slicer)
VIEW_LABELS = [("tibia", "Tibia"), ("defecto", "Defecto"), ("referencia", "Referencia"),
               ("envolvente", "Envolvente"), ("andamio", "Andamio")]
CT_WINDOW_LEVEL = (1800.0, 400.0)               # ventana ósea para mostrar la TC en los cortes


def viewer_ct_path(root, kind, ref):
    """TC de la ROI orientada del caso (misma geometría física que las mallas), si existe."""
    if kind != "paciente":
        return None
    p = Path(root) / "results" / "pathological" / f"{ref}_ft" / "work" / "roi_0000.nii.gz"
    return p if p.is_file() else None


def union_bounds(nodes):
    inf = float("inf")
    b = [inf, -inf, inf, -inf, inf, -inf]
    for n in nodes:
        if n is None:
            continue
        nb = [0.0] * 6
        n.GetRASBounds(nb)
        if nb[0] > nb[1]:
            continue
        for i in (0, 2, 4):
            b[i] = min(b[i], nb[i]); b[i + 1] = max(b[i + 1], nb[i + 1])
    return b if b[0] <= b[1] else None


def tidy_3d_views():
    """Sin la caja ni las letras de orientación de la vista 3D (distraen en la presentación)."""
    lm = slicer.app.layoutManager()
    for i in range(lm.threeDViewCount):
        vn = lm.threeDWidget(i).mrmlViewNode()
        vn.SetBoxVisible(False)
        vn.SetAxisLabelsVisible(False)


def focus_camera(nodes, axis=None):
    """Encuadra la vista 3D en los nodos indicados; axis = ctk.ctkAxesWidget.Anterior, etc."""
    b = union_bounds(nodes)
    if b is None:
        return
    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    if axis is not None:
        view.lookFromAxis(axis)
    ren = view.renderWindow().GetRenderers().GetFirstRenderer()
    ren.ResetCamera(b)
    ren.ResetCameraClippingRange()
    view.forceRender()


def isolate_scene(keep_folder, show_markups):
    """Deja visible solo la carpeta indicada ("Visor" o "Resultados <caso>"): oculta las demás carpetas
    del módulo y, en el visor, los puntos de la corrida, para no mezclar casos en la misma escena."""
    import vtk
    sh = slicer.mrmlScene.GetSubjectHierarchyNode()
    ids = vtk.vtkIdList(); sh.GetItemChildren(sh.GetSceneItemID(), ids)
    for i in range(ids.GetNumberOfIds()):
        it = ids.GetId(i); name = sh.GetItemName(it)
        if name == "Visor" or name.startswith("Resultados "):
            sh.SetItemDisplayVisibility(it, int(name == keep_folder))
    for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
        d = n.GetDisplayNode()
        if d is not None:
            d.SetVisibility(show_markups)


def show_ct_in_slices(volume, center_nodes):
    """TC de fondo en los tres cortes, ventana ósea, cortes centrados en la región de interés."""
    if volume is None:
        return
    d = volume.GetDisplayNode()
    if d is not None:
        d.SetAutoWindowLevel(False); d.SetWindowLevel(*CT_WINDOW_LEVEL)
    slicer.util.setSliceViewerLayers(background=volume, fit=True)
    b = union_bounds(center_nodes)
    if b is not None:
        c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        slicer.modules.markups.logic().JumpSlicesToLocation(c[0], c[1], c[2], True)


# ---------------------------------------------------------------------------- FEA (solo lectura)
# Resultados de results/fea/macro_febio (FEBio, escala 2). El módulo no corre FEBio: lee campo.npz,
# que guarda la envolvente S (x, y, z) = (j, i, k) y el Von Mises por elemento a la carga de referencia
# F_ref, y resultados_macro.csv. Todo escala lineal: VM ~ F; u ~ F / E (eso hace informe_macro.py).
FEA_SUBDIR = Path("results") / "fea" / "macro_febio"
FEA_CASES = {"C": "Caso C (demostrativo)", "P2": "pat_02"}
FEA_MATS = {"Ti": "Ti6Al4V_LPBF", "PCL": "PCL_bTCP_extrusion"}
FEA_FINAL = (0.5, 0.25)                 # malla y banda de los resultados finales (H_FINA, BANDA_BASE)
FEA_U_SUBDIR = "trabajo_u"              # campos con desplazamiento, de fea_desplazamientos.py
FEA_EXAG = [1, 10, 50, 100, 200, 500, 1000]


def fea_runs(root, final_only=True):
    """{etiqueta: dict(tag, caso, mat, h, banda, npz, npz_u)} de las corridas con campo.npz."""
    import re
    base = Path(root) / FEA_SUBDIR
    out = {}
    for d in sorted((base / "trabajo").glob("*")):
        m = re.fullmatch(r"(C|P2)_(Ti|PCL)_h([\d.]+)_b([\d.]+)", d.name)
        if not m or not (d / "campo.npz").is_file():
            continue
        h, b = float(m.group(3)), float(m.group(4))
        if final_only and (h, b) != FEA_FINAL:
            continue
        u = base / FEA_U_SUBDIR / d.name / "campo_u.npz"
        lab = f"{FEA_CASES[m.group(1)]} · {'Ti6Al4V' if m.group(2) == 'Ti' else 'PCL/β-TCP'}"
        if not final_only:
            lab += f" · malla {h:g} mm · banda {b:g}"
        out[lab] = dict(tag=f"{m.group(1)}_{m.group(2)}", caso=m.group(1), mat=m.group(2), h=h, banda=b,
                        npz=d / "campo.npz", npz_u=u if u.is_file() else None)
    return out


def fea_table(root):
    path = Path(root) / FEA_SUBDIR / "resultados_macro.csv"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fea_loads(rows):
    """Cargas presentes en resultados_macro.csv: [(F_N, nombre)]."""
    seen = {}
    for r in rows:
        seen[float(r["F_N"])] = r["carga"]
    return sorted(seen.items())


def fea_row(rows, tag, F, etapa, h):
    for r in rows:
        if r["caso"] == tag and abs(float(r["F_N"]) - F) < 1e-6 and r["etapa"] == etapa and abs(float(r["h_mm"]) - h) < 1e-9:
            return r
    return None


def fea_factor_E(materials, mat, etapa):
    st = next((s for s in stages(materials.get(FEA_MATS[mat], {})) if s["etapa"] == etapa), None)
    return None if st is None or st.get("factor_E") is None else float(st["factor_E"])


def hex_mesh(S):
    """Nodos (índices) y conectividad HEX8 con el mismo orden que celda_febio.malla (0-based)."""
    idx = np.argwhere(S).astype(np.int64)
    nx = np.array(S.shape, np.int64) + 1
    off = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.int64)
    c = np.stack([(idx[:, 0] + o[0]) * nx[1] * nx[2] + (idx[:, 1] + o[1]) * nx[2] + idx[:, 2] + o[2] for o in off], axis=1)
    un, inv = np.unique(c, return_inverse=True)
    ijk = np.stack([un // (nx[1] * nx[2]), (un // nx[2]) % nx[1], un % nx[2]], axis=1)
    return ijk, inv.reshape(c.shape)


def fea_field(npz_path, F, factor_E=1.0, npz_u=None):
    """Malla y campos escalados a la carga F (N) y la etapa (factor_E). Coordenadas en mm, propias de la
    malla del FEA (no registradas a la TC); se gira 180° alrededor de x si hace falta para dejar proximal
    arriba (+z). u es None si el campo de desplazamiento no está guardado."""
    z = np.load(npz_path)
    S, hs, prox, F_ref = z["S"], np.asarray(z["hs"], float), int(z["prox"]), float(z["F_ref"])
    s = F / F_ref
    ijk, conn = hex_mesh(S)
    pts = ijk * hs
    vm = np.asarray(z["vm"], float) * s
    if len(vm) != len(conn):
        raise ValueError(f"{npz_path}: {len(vm)} valores de Von Mises para {len(conn)} elementos")
    u = None
    u_max_ref = None
    if npz_u is not None:
        zu = np.load(npz_u)
        if zu["S"].shape != S.shape or not np.array_equal(zu["S"], S):
            raise ValueError(f"{npz_u}: la envolvente no coincide con {npz_path}")
        u = np.asarray(zu["u"], float) * (F / float(zu["F_ref"])) / factor_E
        if u.shape != pts.shape:
            raise ValueError(f"{npz_u}: {u.shape[0]} desplazamientos para {len(pts)} nodos")
        u_max_ref = float(np.linalg.norm(zu["u"], axis=1).max())
    if prox < 0:                                      # rotación propia: (x, y, z) -> (x, -y, -z)
        rot = np.array([1.0, -1.0, -1.0])
        pts = pts * rot
        u = None if u is None else u * rot
    pts = pts - pts.mean(axis=0)
    return dict(points=pts, conn=conn, vm=vm, u=u, prox=prox, F_ref=F_ref, u_max_ref=u_max_ref,
                vm_p99=float(np.percentile(vm, 99)), vm_max=float(vm.max()),
                u_max=None if u is None else float(np.linalg.norm(u, axis=1).max()))


def fea_suggested_exag(field, frac=0.1):
    """Mayor factor de FEA_EXAG con el que el desplazamiento máximo dibujado no pasa del 10 % del largo de
    la pieza. Es solo una ayuda de visualización (para que la deformada se lea), no un dato del FEA."""
    if field["u"] is None or not field["u_max"]:
        return 1
    L = float(np.ptp(field["points"], axis=0).max())
    ok = [f for f in FEA_EXAG if f * field["u_max"] <= frac * L]
    return max(ok) if ok else 1


def fea_notes(info, row, fE_ok=True):
    """Texto bajo el visor. Todo sale de resultados_macro.csv o de campo.npz; nada se estima aquí."""
    g = lambda k, f="{:.2f}": fmt(None if row is None or row.get(k) in (None, "") else float(row[k]), f)
    L = ["<b>Von Mises:</b> es el esfuerzo <b>aparente</b> del andamio homogenizado (el andamio como sólido "
         "continuo con E efectivo). <b>No</b> es el esfuerzo en las paredes de la celda."]
    if row is not None:
        L.append(f"σ nominal {g('sigma_nom_MPa')} MPa · VM macro media {g('vm_macro_media_MPa')}, p95 {g('vm_macro_p95_MPa')}, "
                 f"máx {g('vm_macro_max_MPa', '{:.1f}')} MPa (el máximo es puntual, en esquinas de vóxel y empotramiento).")
        L.append(f"<b>Esfuerzo local en el sólido</b> (celda, con ε p95): p99 = {g('vm_solido_p99_eps_p95_MPa', '{:.1f}')} MPa · "
                 f"fluencia {g('fluencia_MPa', '{:.1f}')} MPa · <b>FS = {g('FS_p99_eps_p95')}</b>.")
        L.append(f"Desplazamiento máximo (escalado): {g('u_max_mm', '{:.3f}')} mm · ε aparente p95 {fmt(None if not row.get('eps_ap_p95') else 100 * float(row['eps_ap_p95']), '{:.3f}')} %.")
        if row.get("fuera_rango_lineal") == "True":
            L.append("<span style='color:#b45309'><b>Fuera del rango lineal</b> (ε p95 &gt; 1 %): los desplazamientos son "
                     "extrapolación y no son valores físicos; el material no soporta este escenario.</span>")
    else:
        L.append("No hay fila en resultados_macro.csv para esta combinación.")
    if not fE_ok:
        L.append("Esta etapa no tiene factor de E en MATERIALS: el desplazamiento queda sin dato.")
    if info["mat"] == "Ti":
        L.append("Ti6Al4V: E = 16,2 GPa es el módulo <b>ideal de diseño</b>; en piezas impresas reales sería 5–8 GPa (Luo 2023).")
    if info["caso"] == "P2":
        L.append("pat_02 es el caso menos fiable: la carga pasa por cuellos de ~3 mm, del orden de una celda.")
    L.append("Malla del FEA en coordenadas propias (no registrada a la TC), proximal arriba. Cara distal empotrada, "
             "carga axial uniforme en la cara proximal; sin fijación, hueso ni biología. Sin uso clínico.")
    return "<br>".join(L)


def fea_polydata(field):
    """Superficie de la malla HEX8 con VonMises_MPa (celda) y, si hay, Desplazamiento_mm (punto)."""
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints(); pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(field["points"]), deep=True))
    conn = field["conn"].astype(np.int64)
    n = len(conn)
    ca = vtk.vtkCellArray()
    ca.SetData(ns.numpy_to_vtkIdTypeArray(np.arange(0, 8 * n + 1, 8, dtype=np.int64), deep=True),
               ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn.ravel()), deep=True))
    ug = vtk.vtkUnstructuredGrid(); ug.SetPoints(pts); ug.SetCells(vtk.VTK_HEXAHEDRON, ca)
    a = ns.numpy_to_vtk(np.ascontiguousarray(field["vm"], dtype=np.float32), deep=True); a.SetName("VonMises_MPa")
    ug.GetCellData().AddArray(a)
    if field["u"] is not None:
        u = ns.numpy_to_vtk(np.ascontiguousarray(field["u"], dtype=np.float32), deep=True); u.SetName("Desplazamiento_mm")
        ug.GetPointData().AddArray(u)
        m = ns.numpy_to_vtk(np.linalg.norm(field["u"], axis=1).astype(np.float32), deep=True); m.SetName("Desplazamiento_mag_mm")
        ug.GetPointData().AddArray(m)
    sf = vtk.vtkDataSetSurfaceFilter(); sf.SetInputData(ug); sf.PassThroughPointIdsOn(); sf.Update()
    poly = vtk.vtkPolyData(); poly.DeepCopy(sf.GetOutput())
    return poly


def fea_warp(poly, field, factor):
    """Mueve los puntos de la superficie a X0 + factor·u (deformada exagerada; no es una animación)."""
    from vtk.util import numpy_support as ns
    if field["u"] is None:
        return
    ids = ns.vtk_to_numpy(poly.GetPointData().GetArray("vtkOriginalPointIds")).astype(np.int64)
    new = field["points"][ids] + float(factor) * field["u"][ids]
    poly.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(new), deep=True))
    poly.GetPoints().Modified(); poly.Modified()


# ============================================================================ lógica (Slicer)
class AndamioTibialLogic(ScriptedLoadableModuleLogic):

    # ------------------------------------------------------------ validaciones y entradas
    @staticmethod
    def points_lps(node, n_expected=2):
        """Posiciones de los puntos definidos, en LPS mm (coordenadas de mundo)."""
        if node is None:
            return None
        if node.GetParentTransformNode() is not None:
            raise ValueError(f"'{node.GetName()}' tiene una transformación aplicada; endurécela (Harden) antes de ejecutar.")
        pts = []
        for i in range(node.GetNumberOfControlPoints()):
            if hasattr(node, "GetNthControlPointPositionStatus") and \
                    node.GetNthControlPointPositionStatus(i) != node.PositionDefined:
                continue
            ras = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(i, ras)
            pts.append([-ras[0], -ras[1], ras[2]])
        if len(pts) != n_expected:
            raise ValueError(f"'{node.GetName()}' tiene {len(pts)} puntos definidos; se necesitan exactamente {n_expected}.")
        return pts

    @staticmethod
    def _export(node, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(slicer.util, "exportNode"):          # Slicer >= 5.2: no cambia el storage del nodo
            slicer.util.exportNode(node, str(path))
        elif not slicer.util.saveNode(node, str(path)):
            raise RuntimeError(f"No se pudo guardar {path}")
        if not path.is_file():
            raise RuntimeError(f"No se pudo guardar {path}")

    def write_clicks(self, clicks_dir, tibia, defect, contra):
        clicks_dir = Path(clicks_dir)
        clicks_dir.mkdir(parents=True, exist_ok=True)
        for key, node in (("tibia", tibia), ("defect", defect), ("contra", contra)):
            f = clicks_dir / CLICK_FILES[key]
            if f.is_file():
                f.unlink()                                 # sin clic contralateral -> sin archivo
            if node is not None:
                self._export(node, f)

    @staticmethod
    def ct_signature(volume):
        """Geometría + muestra de vóxeles: identifica la TC sin depender del ID del nodo."""
        arr = slicer.util.arrayFromVolume(volume)
        m = vtk_matrix_to_list(volume)
        h = hashlib.md5(np.ascontiguousarray(arr[::3, ::3, ::3]).tobytes()).hexdigest()
        return dict(shape=list(arr.shape), ijk_to_ras=[round(x, 4) for x in m], md5_muestra=h)

    def ct_path(self, volume, out, reuse_export=False):
        """Archivo NIfTI de origen si existe y no se modificó; si no, exporta el volumen
        (o reutiliza la exportación anterior si la TC no cambió)."""
        if volume.GetParentTransformNode() is not None:
            raise ValueError("La TC tiene una transformación aplicada; endurécela (Harden) antes de ejecutar.")
        st = volume.GetStorageNode()
        if st is not None and st.GetFileName():
            src = Path(st.GetFileName())
            if src.is_file() and src.name.lower().endswith((".nii", ".nii.gz")) and not volume.GetModifiedSinceRead():
                return src
        dst = Path(out) / "ct_slicer.nii.gz"
        if not (reuse_export and dst.is_file()):
            self._export(volume, dst)
        return dst

    # ------------------------------------------------------------ proceso externo
    @staticmethod
    def child_environment():
        try:
            base = slicer.util.startupEnvironment()     # entorno previo al lanzador de Slicer
        except AttributeError:
            base = dict(os.environ)
        return clean_child_environment(base)

    # ------------------------------------------------------------ resultados
    @staticmethod
    def read_report(out):
        """reporte.json (paso 1) + andamio_scaffold_report.json (paso 2) como 'andamio'
        + slicer_corrida.json como '_slicer'. No reescribe ningún archivo del pipeline."""
        out = Path(out)
        f = out / "reporte.json"
        if not f.is_file():
            return None
        rep = json.loads(f.read_text(encoding="utf-8"))
        sf = out / f"{SCAFFOLD_NAME}_scaffold_report.json"
        if "andamio" not in rep and sf.is_file():
            rep["andamio"] = json.loads(sf.read_text(encoding="utf-8"))
        cf = out / "slicer_corrida.json"
        if cf.is_file():
            rep["_slicer"] = json.loads(cf.read_text(encoding="utf-8"))
        return rep

    @staticmethod
    def metric_rows(report, all_rows=False):
        if report is None:
            return []
        if all_rows:
            return [("reporte.json", k, fmt(v)) for k, v in flatten(report)]
        rows = []
        for sec, label, path, f in MAIN_METRICS:
            v = get_path(report, path)
            if v is not None:
                rows.append((sec, label, fmt(v, f)))
        return rows

    def load_view(self, items):
        """Reemplaza la vista anterior del visor por las mallas indicadas."""
        sh = slicer.mrmlScene.GetSubjectHierarchyNode()
        old = sh.GetItemChildWithName(sh.GetSceneItemID(), "Visor")
        if old:
            sh.RemoveItem(old)
        folder = sh.CreateFolderItem(sh.GetSceneItemID(), "Visor")
        self.shown = {}
        for path, color, opacity, label in items:
            n = slicer.util.loadModel(str(path))
            n.SetName(f"visor_{label}")
            d = n.GetDisplayNode()
            d.SetColor(*color); d.SetOpacity(opacity); d.SetVisibility2D(True)
            sh.SetItemParent(sh.GetItemByDataNode(n), folder)
            self.shown[label] = n
        return self.shown

    def load_ct(self, path):
        """Carga la TC una sola vez por archivo (se reutiliza al cambiar de etapa o material)."""
        name = "TC_" + Path(path).parent.parent.name
        vol = slicer.mrmlScene.GetFirstNodeByName(name)
        if vol is None:
            vol = slicer.util.loadVolume(str(path))
            vol.SetName(name)
        return vol

    def load_results(self, out, case, opts, ct_node=None):
        """Carga en una carpeta de Subject Hierarchy; reemplaza la carga anterior del mismo caso."""
        out = Path(out)
        sh = slicer.mrmlScene.GetSubjectHierarchyNode()
        folder_name = f"Resultados {case}"
        old = sh.GetItemChildWithName(sh.GetSceneItemID(), folder_name)
        if old:
            sh.RemoveItem(old)
        folder = sh.CreateFolderItem(sh.GetSceneItemID(), folder_name)
        loaded = []
        self.shown = {}

        def model(fname, color, opacity, label):
            f = out / fname
            if not f.is_file():
                return None
            n = slicer.util.loadModel(str(f))
            n.SetName(f"{case}_{label}")
            d = n.GetDisplayNode()
            d.SetColor(*color); d.SetOpacity(opacity); d.SetVisibility2D(True)
            sh.SetItemParent(sh.GetItemByDataNode(n), folder)
            loaded.append(n.GetName())
            self.shown[label] = n
            return n

        has_scaffold = (out / f"{SCAFFOLD_NAME}_scaffold_light.stl").is_file()
        if opts.get("tibia", True):
            model("tibia.stl", (0.89, 0.85, 0.79), 0.35 if has_scaffold else 1.0, "tibia")
        if opts.get("defecto", True):
            model("defecto.stl", (0.85, 0.25, 0.25), 0.4, "defecto")
        if opts.get("referencia", False):
            model("referencia.stl", (0.4, 0.6, 0.9), 0.25, "referencia")
        if opts.get("envolvente", False):
            model(f"{SCAFFOLD_NAME}_envelope.stl", (0.95, 0.75, 0.2), 0.3, "envolvente")
        sc_file = f"{SCAFFOLD_NAME}_scaffold.stl" if opts.get("andamio_fino") else f"{SCAFFOLD_NAME}_scaffold_light.stl"
        if opts.get("andamio", True):
            model(sc_file, (0.2, 0.7, 0.55), 1.0, "andamio")

        if opts.get("mascaras", True):
            seg = self._masks_to_segmentation(out, case)
            if seg is not None:
                sh.SetItemParent(sh.GetItemByDataNode(seg), folder)
                loaded.append(seg.GetName())
        return loaded

    @staticmethod
    def _masks_to_segmentation(out, case):
        """Máscaras en su propia grilla (ROI orientada): sin remuestrear a la TC."""
        masks = [("tibia_segmentada.nii.gz", "Tibia", (0.89, 0.85, 0.79)),
                 ("defecto.nii.gz", "Defecto", (0.85, 0.25, 0.25)),
                 ("defecto_corregido.nii.gz", "Defecto corregido", (0.95, 0.6, 0.1))]
        masks = [m for m in masks if (out / m[0]).is_file()]
        if not masks:
            return None
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{case}_segmentacion")
        seg.CreateDefaultDisplayNodes()
        logic = slicer.modules.segmentations.logic()
        for fname, sname, color in masks:
            lab = slicer.util.loadLabelVolume(str(out / fname))
            before = seg.GetSegmentation().GetNumberOfSegments()
            logic.ImportLabelmapToSegmentationNode(lab, seg)
            segm = seg.GetSegmentation()
            for i in range(before, segm.GetNumberOfSegments()):
                s = segm.GetNthSegment(i)
                s.SetName(sname); s.SetColor(*color)
            slicer.mrmlScene.RemoveNode(lab)
        seg.GetDisplayNode().SetOpacity2DFill(0.3)
        return seg

    @staticmethod
    def export(out, dest, rows_all, log_text):
        """Copia STL + reportes a 'dest' y escribe métricas en CSV y el registro de la corrida."""
        out, dest = Path(out), Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in RESULT_FILES:
            if name.endswith((".stl", ".json")) and (out / name).is_file():
                shutil.copy2(out / name, dest / name)
                copied.append(name)
        with open(dest / "metricas.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["seccion", "metrica", "valor"])
            w.writerows(rows_all)
        copied.append("metricas.csv")
        (dest / "registro_corrida.txt").write_text(log_text, encoding="utf-8")
        copied.append("registro_corrida.txt")
        return copied


def vtk_matrix_to_list(volume):
    import vtk
    m = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(m)
    return [m.GetElement(r, c) for r in range(4) for c in range(4)]


# ============================================================================ interfaz
class AndamioTibialWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = AndamioTibialLogic()
        self.proc = None
        self.timer = qt.QTimer(); self.timer.setInterval(250); self.timer.connect("timeout()", self._poll)
        self._buf = ""
        self._run_ctx = None
        s = qt.QSettings()

        # ---------------------------------------------------- modo
        mw = qt.QWidget(); ml = qt.QHBoxLayout(mw); ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(qt.QLabel("Modo:"))
        self.modeBasic = qt.QRadioButton("Básico"); self.modeAdv = qt.QRadioButton("Avanzado")
        ml.addWidget(self.modeBasic); ml.addWidget(self.modeAdv); ml.addStretch(1)
        self.layout.addWidget(mw)
        self._advanced_widgets = []

        # ---------------------------------------------------- configuración
        cfg = ctk.ctkCollapsibleButton(); cfg.text = "Configuración"; cfg.collapsed = True
        self._advanced_widgets.append(cfg)
        self.layout.addWidget(cfg); f = qt.QFormLayout(cfg)
        self.rootEdit = ctk.ctkPathLineEdit(); self.rootEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.rootEdit.currentPath = s.value(SETTINGS + "root", DEFAULT_ROOT)
        self.rootEdit.toolTip = "Carpeta del proyecto (contiene src/ y .venv_nnunet/)."
        f.addRow("Proyecto:", self.rootEdit)
        self.pyEdit = ctk.ctkPathLineEdit(); self.pyEdit.filters = ctk.ctkPathLineEdit.Files
        self.pyEdit.currentPath = s.value(SETTINGS + "python", str(Path(DEFAULT_ROOT) / ".venv_nnunet" / "Scripts" / "python.exe"))
        self.pyEdit.toolTip = "Python con torch y nnU-Net (no el de Slicer)."
        f.addRow("Python del pipeline:", self.pyEdit)
        self.outEdit = ctk.ctkPathLineEdit(); self.outEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.outEdit.currentPath = s.value(SETTINGS + "out", str(Path(DEFAULT_ROOT) / "results" / "slicer"))
        f.addRow("Carpeta de resultados:", self.outEdit)
        self.caseEdit = qt.QLineEdit(s.value(SETTINGS + "case", "caso_01"))
        self.caseEdit.toolTip = "Subcarpeta del caso dentro de la carpeta de resultados."

        # ---------------------------------------------------- entradas
        inp = ctk.ctkCollapsibleButton(); inp.text = "1. TC y puntos"
        self.layout.addWidget(inp); f = qt.QFormLayout(inp)
        self.ctSel = slicer.qMRMLNodeComboBox()
        self.ctSel.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSel.addEnabled = False; self.ctSel.removeEnabled = False; self.ctSel.noneEnabled = True
        self.ctSel.setMRMLScene(slicer.mrmlScene)
        self.ctSel.toolTip = "TC en HU (DICOM cargado con el módulo DICOM, o NIfTI/NRRD)."
        f.addRow("TC:", self.ctSel)
        f.addRow("Nombre del caso:", self.caseEdit)
        self.ctSel.connect("currentNodeChanged(vtkMRMLNode*)",
                           lambda n: setattr(self.caseEdit, "text", case_name_from(n.GetName())) if n is not None else None)
        self.mkTibia = self._markups_widget("Tibia_P1P2", "P%d", (0.2, 0.6, 1.0))
        f.addRow(qt.QLabel("Tibia afectada: P1 = proximal, P2 = distal (2 puntos sobre el eje)"))
        f.addRow(self.mkTibia)
        self.mkDefect = self._markups_widget("Defecto_D1D2", "D%d", (1.0, 0.3, 0.3))
        f.addRow(qt.QLabel("Defecto: D1 / D2 = hueso sano justo por encima y por debajo"))
        f.addRow(self.mkDefect)
        self.useContra = qt.QCheckBox("Usar tibia contralateral (opcional, P1/P2 de la otra pierna)")
        self._advanced_widgets.append(self.useContra)
        f.addRow(self.useContra)
        self.mkContra = self._markups_widget("Tibia_contra_P1P2", "P%d", (0.6, 0.6, 0.6))
        self.mkContra.enabled = False
        self.useContra.connect("toggled(bool)", lambda on: setattr(self.mkContra, "enabled", on))
        self._advanced_widgets.append(self.mkContra)
        f.addRow(self.mkContra)

        # ---------------------------------------------------- parámetros
        par = ctk.ctkCollapsibleButton(); par.text = "2. Parámetros"
        self.layout.addWidget(par); pv = qt.QVBoxLayout(par)
        f = qt.QFormLayout(); pv.addLayout(f)
        self.materials = {}
        self.matCombo = qt.QComboBox(); f.addRow("Material:", self.matCombo)
        self.stageCombo = qt.QComboBox()
        self.stageCombo.toolTip = "Etapa de degradación: no cambia la geometría, solo las propiedades para el FEA."
        f.addRow("Etapa (FEA):", self.stageCombo)
        self.propLabel = qt.QLabel(); self.propLabel.wordWrap = True
        f.addRow("Propiedades del sólido:", self.propLabel)

        adv = ctk.ctkCollapsibleButton(); adv.text = "Opciones avanzadas"; adv.collapsed = True
        pv.addWidget(adv); f = qt.QFormLayout(adv)
        self._advanced_widgets.append(adv)
        self.modelCombo = qt.QComboBox(); self.modelCombo.addItems(["ft", "base"])
        self.modelCombo.toolTip = "ft = Dataset503 ajustado con defectos v2 (vigente); base = Dataset502."
        f.addRow("Modelo nnU-Net:", self.modelCombo)
        self.foldsEdit = qt.QLineEdit("0"); f.addRow("Folds:", self.foldsEdit)
        self.ttaCheck = qt.QCheckBox("TTA (espejado; más lento)"); f.addRow("", self.ttaCheck)
        self.refCombo = qt.QComboBox(); self.refCombo.addItems(["auto", "interp", "contra"])
        self.refCombo.toolTip = "Forma de referencia del defecto. 'contra' requiere los puntos contralaterales."
        f.addRow("Referencia del defecto:", self.refCombo)
        self.noScaffold = qt.QCheckBox("Solo segmentación y defecto (sin andamio)"); f.addRow("", self.noScaffold)
        self.cellSpin = self._spin(0.5, 10.0, 0.1, 2.5, " mm"); f.addRow("Celda TPMS:", self.cellSpin)
        self.wallSpin = self._spin(0.05, 3.0, 0.05, 0.30, " mm"); f.addRow("Pared:", self.wallSpin)
        self.fabFine = qt.QCheckBox(f"Chequeo fino de fabricabilidad (paso {FAB_H_DEFAULT:g} mm; más lento)")
        self.fabFine.checked = True; f.addRow("", self.fabFine)
        self.refineCombo = qt.QComboBox()
        for _, lab in REFINE_MODES:
            self.refineCombo.addItem(lab)
        self.refineCombo.toolTip = ("Modo opcional declarado: rescata como hueso los sectores con densidad >= 600 HU "
                                    "contigua al huésped (masas > 3 mm y zona a 3 mm del metal excluidas). "
                                    "Parámetros congelados; no es la salida automática.")
        f.addRow("Corrección del defecto por densidad:", self.refineCombo)
        self.matCombo.connect("currentIndexChanged(int)", lambda _: self._on_material())
        self.stageCombo.connect("currentIndexChanged(int)", lambda _: self._on_stage())
        self.rootEdit.connect("currentPathChanged(QString)", lambda _: self._load_materials())
        self.rootEdit.connect("currentPathChanged(QString)", lambda _: self._fill_fea() if hasattr(self, "feaRun") else None)
        self.nailCombo = qt.QComboBox(); self.nailCombo.addItems(["none", "auto"])
        self.nailCombo.toolTip = "auto = detectar clavo intramedular por HU de metal y restar un canal."
        f.addRow("Canal para clavo:", self.nailCombo)
        self.nailDiam = self._spin(0.0, 20.0, 0.5, 0.0, " mm"); self.nailDiam.specialValueText = "detectado"
        f.addRow("Diámetro del clavo:", self.nailDiam)
        self.plugSpin = self._spin(0.0, 30.0, 1.0, 0.0, " mm"); self.plugSpin.specialValueText = "sin tapón"
        f.addRow("Tapón intramedular:", self.plugSpin)
        self.gradCheck = qt.QCheckBox("Gradiente radial de pared (modo lámina)"); f.addRow("", self.gradCheck)
        gw = qt.QWidget(); gl = qt.QHBoxLayout(gw); gl.setContentsMargins(0, 0, 0, 0)
        self.gExt = self._spin(0.05, 3, 0.05, 0.40, ""); self.gInt = self._spin(0.05, 3, 0.05, 0.25, "")
        self.gP1 = self._spin(0, 30, 0.5, 2.0, ""); self.gP2 = self._spin(0, 30, 0.5, 6.0, "")
        for lab, w in (("t_ext", self.gExt), ("t_int", self.gInt), ("p1", self.gP1), ("p2", self.gP2)):
            gl.addWidget(qt.QLabel(lab)); gl.addWidget(w)
        gw.enabled = False; self.gradCheck.connect("toggled(bool)", lambda on: setattr(gw, "enabled", on))
        gw.toolTip = "Pared t_ext hasta la profundidad p1, interpolación lineal hasta t_int en p2 (mm)."
        f.addRow("t_ext, t_int, p1, p2 (mm):", gw)

        # ---------------------------------------------------- ejecución
        run = ctk.ctkCollapsibleButton(); run.text = "3. Ejecutar"
        self.layout.addWidget(run); v = qt.QVBoxLayout(run)
        h = qt.QHBoxLayout(); v.addLayout(h)
        self.runBtn = qt.QPushButton("Ejecutar pipeline"); self.runBtn.connect("clicked()", self.onRun)
        self.cancelBtn = qt.QPushButton("Cancelar"); self.cancelBtn.enabled = False
        self.cancelBtn.connect("clicked()", self.onCancel)
        h.addWidget(self.runBtn); h.addWidget(self.cancelBtn)
        self.stageBar = qt.QProgressBar(); self.stageBar.setRange(0, len(STAGES)); self.stageBar.value = 0
        self.stageBar.setFormat("%v de %m etapas"); v.addWidget(self.stageBar)
        self.stageLabel = qt.QLabel(); self.stageLabel.setTextFormat(qt.Qt.PlainText); v.addWidget(self.stageLabel)
        self.tracker = None
        self._render_stages()
        self.status = qt.QLabel("Listo."); self.status.wordWrap = True; v.addWidget(self.status)
        self.logView = qt.QPlainTextEdit(); self.logView.readOnly = True
        self._advanced_widgets.append(self.logView)
        self.logView.setMinimumHeight(180); self.logView.setMaximumBlockCount(5000)
        v.addWidget(self.logView)

        # ---------------------------------------------------- resultados
        res = ctk.ctkCollapsibleButton(); res.text = "4. Resultados"
        self.layout.addWidget(res); v = qt.QVBoxLayout(res)
        g = qt.QGridLayout(); v.addLayout(g)
        self.optChecks = {}
        for i, (k, lab, on) in enumerate([("tibia", "Tibia", True), ("defecto", "Defecto", True),
                                          ("andamio", "Andamio", True), ("mascaras", "Máscaras en cortes", True),
                                          ("referencia", "Referencia", False), ("envolvente", "Envolvente", False),
                                          ("andamio_fino", "Andamio malla fina (pesada)", False)]):
            c = qt.QCheckBox(lab); c.checked = on; self.optChecks[k] = c; g.addWidget(c, i // 3, i % 3)
            self._advanced_widgets.append(c)
        h = qt.QHBoxLayout(); v.addLayout(h)
        self.loadBtn = qt.QPushButton("Cargar resultados del caso"); self.loadBtn.connect("clicked()", self.onLoad)
        self.exportBtn = qt.QPushButton("Exportar STL y reporte…"); self.exportBtn.connect("clicked()", self.onExport)
        self.openBtn = qt.QPushButton("Abrir carpeta"); self.openBtn.connect("clicked()", self.onOpenFolder)
        h.addWidget(self.loadBtn); h.addWidget(self.exportBtn); h.addWidget(self.openBtn)
        self.allMetrics = qt.QCheckBox("Mostrar todas las métricas de reporte.json")
        self.allMetrics.connect("toggled(bool)", lambda _: self.refreshTable())
        v.addWidget(self.allMetrics)
        self._advanced_widgets.append(self.allMetrics)
        qcb = qt.QGroupBox("Control de calidad"); ql = qt.QVBoxLayout(qcb)
        self.qcLabel = qt.QLabel(); self.qcLabel.wordWrap = True; self.qcLabel.setTextFormat(qt.Qt.RichText)
        ql.addWidget(self.qcLabel); v.addWidget(qcb)
        self.table = qt.QTableWidget(); self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Sección", "Métrica", "Valor"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(260)
        v.addWidget(self.table)

        # ---------------------------------------------------- visor (sin ejecutar nada)
        vw = ctk.ctkCollapsibleButton(); vw.text = "5. Visor de casos calculados"; vw.collapsed = True
        self.layout.addWidget(vw); f = qt.QFormLayout(vw)
        self.viewCase = qt.QComboBox(); f.addRow("Caso:", self.viewCase)
        self.viewVariant = qt.QComboBox(); self.viewVariant.addItems(list(VIEW_VARIANTS)); f.addRow("Defecto:", self.viewVariant)
        self.viewMat = qt.QComboBox(); self.viewMat.addItems(list(VIEW_MATERIALS)); f.addRow("Material:", self.viewMat)
        self.viewStage = qt.QComboBox(); self.viewStage.addItems(VIEW_STAGES); self.viewStage.currentIndex = 3
        f.addRow("Etapa:", self.viewStage)
        self.viewInfo = qt.QLabel(""); self.viewInfo.wordWrap = True; f.addRow(self.viewInfo)
        self.viewQC = qt.QLabel(""); self.viewQC.wordWrap = True; self.viewQC.setTextFormat(qt.Qt.RichText)
        f.addRow(self.viewQC)
        self.refreshView = qt.QPushButton("Actualizar lista de casos"); f.addRow(self.refreshView)
        self.refreshView.connect("clicked()", self._fill_view_cases)
        self._fill_view_cases()
        self.viewCT = qt.QCheckBox("Mostrar la TC en los cortes (la primera vez tarda unos segundos)")
        self.viewCT.checked = True; f.addRow(self.viewCT)
        for c in (self.viewCase, self.viewVariant, self.viewMat, self.viewStage):
            c.connect("currentIndexChanged(int)", lambda _: self.onView())

        # ---------------------------------------------------- vista 3D
        vis = ctk.ctkCollapsibleButton(); vis.text = "Vista 3D"
        self.layout.addWidget(vis); g = qt.QGridLayout(vis)
        self.visRows = {}
        for r, (key, lab) in enumerate(VIEW_LABELS):
            chk = qt.QCheckBox(lab); chk.checked = True
            sl = ctk.ctkSliderWidget(); sl.minimum = 0.0; sl.maximum = 1.0; sl.singleStep = 0.05; sl.decimals = 2
            sl.toolTip = "Opacidad"
            chk.connect("toggled(bool)", lambda on, k=key: self._set_visible(k, on))
            sl.connect("valueChanged(double)", lambda v, k=key: self._set_opacity(k, v))
            g.addWidget(chk, r, 0); g.addWidget(sl, r, 1)
            self.visRows[key] = (chk, sl)
        bw = qt.QWidget(); bl = qt.QHBoxLayout(bw); bl.setContentsMargins(0, 0, 0, 0)
        for lab, axis in (("Centrar", None), ("Anterior", ctk.ctkAxesWidget.Anterior),
                          ("Posterior", ctk.ctkAxesWidget.Posterior), ("Izquierda", ctk.ctkAxesWidget.Left),
                          ("Derecha", ctk.ctkAxesWidget.Right)):
            b = qt.QPushButton(lab); b.connect("clicked()", lambda a=axis: self.onFocus(a)); bl.addWidget(b)
        g.addWidget(bw, len(VIEW_LABELS), 0, 1, 2)
        self._sync_vis_panel()

        # ---------------------------------------------------- resultados FEA (solo lectura)
        fe = ctk.ctkCollapsibleButton(); fe.text = "6. Resultados FEA (FEBio)"; fe.collapsed = True
        self.layout.addWidget(fe); f = qt.QFormLayout(fe)
        self.feaRun = qt.QComboBox(); f.addRow("Caso y material:", self.feaRun)
        self.feaLoad = qt.QComboBox(); f.addRow("Carga axial:", self.feaLoad)
        self.feaStage = qt.QComboBox(); self.feaStage.toolTip = "La etapa no cambia el Von Mises macro; el desplazamiento escala con 1/factor_E."
        f.addRow("Etapa:", self.feaStage)
        self.feaColor = qt.QComboBox(); self.feaColor.addItems(["Von Mises aparente (MPa)", "Desplazamiento (mm)"])
        f.addRow("Color:", self.feaColor)
        ew = qt.QWidget(); el = qt.QHBoxLayout(ew); el.setContentsMargins(0, 0, 0, 0)
        self.feaExag = qt.QSlider(qt.Qt.Horizontal); self.feaExag.minimum = 0; self.feaExag.maximum = len(FEA_EXAG) - 1
        self.feaExagLabel = qt.QLabel("×1"); el.addWidget(self.feaExag); el.addWidget(self.feaExagLabel)
        self.feaExag.toolTip = ("Al mostrar, se elige el mayor factor con el que la deformada no pasa del 10 % del largo "
                                "de la pieza (ayuda visual). Los mm de la barra de color son los reales, sin exagerar.")
        self.feaExagRow = ew
        f.addRow("Deformada exagerada:", ew)
        self.feaFull = qt.QCheckBox("Escala de color hasta el máximo (por defecto hasta el p99, como en las figuras)")
        f.addRow(self.feaFull); self._advanced_widgets.append(self.feaFull)
        self.feaHide = qt.QCheckBox("Ocultar el resto de la escena"); self.feaHide.checked = True; f.addRow(self.feaHide)
        self.feaShow = qt.QPushButton("Mostrar FEA"); f.addRow(self.feaShow)
        self.feaNotes = qt.QLabel(""); self.feaNotes.wordWrap = True; self.feaNotes.setTextFormat(qt.Qt.RichText)
        f.addRow(self.feaNotes)
        self._fea = None; self._fea_hidden = []
        self.feaShow.connect("clicked()", self.onFea)
        self.feaRun.connect("currentIndexChanged(int)", lambda _: self._fea_stages())
        self.feaColor.connect("currentIndexChanged(int)", lambda _: self._fea_style())
        self.feaFull.connect("toggled(bool)", lambda _: self._fea_style())
        self.feaExag.connect("valueChanged(int)", self._fea_exag)
        self.feaHide.connect("toggled(bool)", self._fea_hide)

        self.layout.addStretch(1)
        self._load_materials()
        self.refreshTable()
        self.modeBasic.connect("toggled(bool)", lambda _: self._apply_mode())
        (self.modeAdv if s.value(SETTINGS + "modo", "basico") == "avanzado" else self.modeBasic).checked = True
        self._apply_mode()

    # ---------------------------------------------------- modo y etapas
    def _apply_mode(self):
        """Básico: TC, puntos, material/etapa, ejecutar y resultados. Avanzado: todo."""
        adv = self.modeAdv.checked
        for w in self._advanced_widgets:
            w.visible = adv
        qt.QSettings().setValue(SETTINGS + "modo", "avanzado" if adv else "basico")
        if hasattr(self, "feaRun"):
            self._fill_fea()

    def _now(self):
        import time
        return time.time()

    def _render_stages(self):
        tr = self.tracker
        if tr is None:
            self.stageLabel.text = "\n".join(f"{ICON['pendiente']}  {n}" for n, _ in STAGES)
            self.stageBar.value = 0
            return
        now = self._now()
        self.stageLabel.text = "\n".join(tr.lines(now)) + f"\nTiempo total: {now - tr.start_time:.0f} s"
        self.stageBar.value = tr.done_count()

    # ---------------------------------------------------- visor
    def _fill_view_cases(self):
        self._view_cases = viewer_cases(self.rootEdit.currentPath)
        self.viewCase.blockSignals(True)
        self.viewCase.clear(); self.viewCase.addItems(list(self._view_cases))
        self.viewCase.blockSignals(False)
        self.viewInfo.text = "" if self._view_cases else "No hay casos calculados en este proyecto."

    def onView(self):
        case = self.viewCase.currentText
        if case not in self._view_cases:
            return
        kind, ref = self._view_cases[case]
        var, mat = VIEW_VARIANTS[self.viewVariant.currentText], VIEW_MATERIALS[self.viewMat.currentText]
        root = self.rootEdit.currentPath
        items, notes = viewer_items(root, kind, ref, var, mat, self.viewStage.currentIndex)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.logic.load_view(items)
        finally:
            slicer.app.restoreOverrideCursor()
        isolate_scene("Visor", show_markups=False)
        self.viewInfo.text = es("\n".join(notes + viewer_metrics(viewer_folder(root, kind, ref, var, mat))))
        reps = sorted(viewer_folder(root, kind, ref, var, mat).glob("*_scaffold_report.json"))
        self.viewQC.text = es(qc_html(qc_items({"andamio": json.loads(reps[0].read_text(encoding="utf-8"))}))) if reps else ""
        ct_path = viewer_ct_path(root, kind, ref) if self.viewCT.checked else None
        if ct_path is not None:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                show_ct_in_slices(self.logic.load_ct(ct_path), self._focus_nodes())
            finally:
                slicer.app.restoreOverrideCursor()
        elif self.viewCT.checked:                       # sin TC del caso: no dejar la TC de otro caso de fondo
            slicer.util.setSliceViewerLayers(background=None, foreground=None)
        self._after_load()

    # ---------------------------------------------------- vista 3D
    def _focus_nodes(self):
        """Región de interés: andamio, si no envolvente o defecto; si no hay ninguno, todo."""
        shown = getattr(self.logic, "shown", {}) or {}
        for k in ("andamio", "envolvente", "defecto"):
            if k in shown:
                return [shown[k]]
        return list(shown.values())

    def _after_load(self):
        tidy_3d_views()
        self._sync_vis_panel()
        focus_camera(self._focus_nodes())

    def _sync_vis_panel(self):
        shown = getattr(self.logic, "shown", {}) or {}
        for key, (chk, sl) in self.visRows.items():
            n = shown.get(key)
            d = n.GetDisplayNode() if n is not None else None
            for w in (chk, sl):
                w.blockSignals(True)
            chk.enabled = sl.enabled = d is not None
            if d is not None:
                chk.checked = bool(d.GetVisibility()); sl.value = d.GetOpacity()
            for w in (chk, sl):
                w.blockSignals(False)

    def _display(self, key):
        n = (getattr(self.logic, "shown", {}) or {}).get(key)
        return n.GetDisplayNode() if n is not None else None

    def _set_visible(self, key, on):
        d = self._display(key)
        if d is not None:
            d.SetVisibility(on)

    def _set_opacity(self, key, v):
        d = self._display(key)
        if d is not None:
            d.SetOpacity(v)

    def onFocus(self, axis=None):
        tidy_3d_views()
        focus_camera(self._focus_nodes(), axis)

    # ---------------------------------------------------- resultados FEA
    def _fill_fea(self):
        root = self.rootEdit.currentPath
        cur = self.feaRun.currentText
        self._fea_runs = fea_runs(root, final_only=not self.modeAdv.checked)
        self._fea_rows = fea_table(root)
        for c in (self.feaRun, self.feaLoad):
            c.blockSignals(True); c.clear()
        self.feaRun.addItems(list(self._fea_runs))
        i = self.feaRun.findText(cur); self.feaRun.setCurrentIndex(max(i, 0))
        for F, name in fea_loads(self._fea_rows):
            self.feaLoad.addItem(f"{F:.0f} N ({name.split('_')[0]})", F)
        self.feaLoad.setCurrentIndex(max(self.feaLoad.count - 1, 0))
        for c in (self.feaRun, self.feaLoad):
            c.blockSignals(False)
        self.feaShow.enabled = bool(self._fea_runs) and self.feaLoad.count > 0
        if not self._fea_runs:
            self.feaNotes.text = f"No hay resultados en {Path(root) / FEA_SUBDIR / 'trabajo'}."
        elif not self._fea_rows:
            self.feaNotes.text = "Falta resultados_macro.csv (de informe_macro.py)."
        self._fea_stages()

    def _fea_stages(self):
        info = self._fea_runs.get(self.feaRun.currentText) if hasattr(self, "_fea_runs") else None
        self.feaStage.clear()
        if info is not None:
            for st in stages(getattr(self, "materials", {}).get(FEA_MATS[info["mat"]], {})):
                self.feaStage.addItem(st["etapa"])
        self.feaStage.enabled = self.feaStage.count > 1
        has_u = info is not None and info["npz_u"] is not None
        if not has_u:
            self.feaColor.setCurrentIndex(0)
        self.feaColor.enabled = self.feaExag.enabled = has_u

    def onFea(self):
        info = self._fea_runs.get(self.feaRun.currentText)
        if info is None:
            return
        F = float(self.feaLoad.currentData)
        etapa = self.feaStage.currentText or "t0"
        fE = fea_factor_E(getattr(self, "materials", {}), info["mat"], etapa)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            field = fea_field(info["npz"], F, fE or 1.0, info["npz_u"] if fE else None)
            poly = fea_polydata(field)
        except Exception as e:  # noqa: BLE001
            slicer.app.restoreOverrideCursor()
            slicer.util.errorDisplay(f"No se pudo leer el resultado del FEA:\n{e}")
            return
        name = f"FEA {self.feaRun.currentText} · {F:.0f} N · {etapa}"
        node = self._fea["node"] if self._fea and slicer.mrmlScene.IsNodePresent(self._fea["node"]) else None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            node.CreateDefaultDisplayNodes()
        node.SetName(name); node.SetAndObservePolyData(poly)
        self._fea = dict(node=node, poly=poly, field=field, info=info)
        row = fea_row(self._fea_rows, info["tag"], F, etapa, info["h"])
        notes = fea_notes(info, row, fE_ok=fE is not None)
        if info["npz_u"] is None:
            notes += ("<br><i>El campo de desplazamiento no se guardó en campo.npz (solo el máximo, arriba). Para colorear por "
                      "desplazamiento y exagerar la deformada hay que generarlo con fea_desplazamientos.py.</i>")
        self.feaNotes.text = es(notes)
        self._fea_style()
        self.feaExag.blockSignals(True)
        self.feaExag.value = FEA_EXAG.index(fea_suggested_exag(field))
        self.feaExag.blockSignals(False)
        self._fea_exag(self.feaExag.value, refocus=False)
        self._fea_hide(self.feaHide.checked)
        slicer.app.restoreOverrideCursor()
        tidy_3d_views()
        focus_camera([node], ctk.ctkAxesWidget.Anterior)

    def _fea_style(self):
        if not self._fea or not slicer.mrmlScene.IsNodePresent(self._fea["node"]):
            return
        import vtk
        d = self._fea["node"].GetDisplayNode(); fld = self._fea["field"]
        by_u = self.feaColor.currentIndex == 1 and fld["u"] is not None
        if by_u:
            d.SetActiveScalar("Desplazamiento_mag_mm", vtk.vtkAssignAttribute.POINT_DATA); hi, title = fld["u_max"], "Desplazamiento (mm)"
        else:
            d.SetActiveScalar("VonMises_MPa", vtk.vtkAssignAttribute.CELL_DATA)
            hi = fld["vm_max"] if self.feaFull.checked else fld["vm_p99"]
            title = "Von Mises aparente (MPa)" + ("" if self.feaFull.checked else ", hasta p99")
        for cid in ("vtkMRMLColorTableNodeFileViridis.txt", "vtkMRMLColorTableNodeRainbow"):
            if slicer.mrmlScene.GetNodeByID(cid) is not None:
                d.SetAndObserveColorNodeID(cid); break
        d.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        d.SetScalarRange(0.0, float(hi) if hi else 1.0)
        d.SetScalarVisibility(True); d.SetVisibility2D(False); d.SetBackfaceCulling(False); d.SetVisibility(True)
        try:
            leg = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(self._fea["node"])
            leg.SetTitleText(title); leg.SetLabelFormat("%.3g"); leg.SetVisibility(True)
        except Exception:  # noqa: BLE001  (barra de color: Slicer >= 5.0)
            pass

    def _fea_exag(self, i, refocus=True):
        f = FEA_EXAG[int(i)]
        self.feaExagLabel.text = f"×{f}"
        if self._fea and self._fea["field"]["u"] is not None:
            fea_warp(self._fea["poly"], self._fea["field"], f)
            self._fea["node"].Modified()
            if refocus:
                focus_camera([self._fea["node"]])

    def _fea_hide(self, on):
        """Oculta (o restaura) las demás mallas y segmentaciones: la malla del FEA no está registrada a la TC."""
        own = self._fea["node"].GetDisplayNode() if self._fea else None
        if on:
            for cls in ("vtkMRMLModelDisplayNode", "vtkMRMLSegmentationDisplayNode"):
                for d in slicer.util.getNodesByClass(cls):
                    if d is not own and d.GetVisibility() and not d.IsA("vtkMRMLColorLegendDisplayNode"):
                        d.SetVisibility(False); self._fea_hidden.append(d)
        else:
            for d in self._fea_hidden:
                if slicer.mrmlScene.IsNodePresent(d):
                    d.SetVisibility(True)
            self._fea_hidden = []
            if own is not None:
                own.SetVisibility(False)
        self._sync_vis_panel()

    # ---------------------------------------------------- material y etapa
    def _load_materials(self):
        """MATERIALS se lee de scaffold_f4.py del proyecto: una sola fuente para valores y citas."""
        try:
            self.materials = load_materials(self.rootEdit.currentPath)
        except Exception as e:  # noqa: BLE001
            self.materials = {}
            self.propLabel.text = f"No se pudo leer MATERIALS de scaffold_f4.py: {e}"
        cur = self.matCombo.currentText
        self.matCombo.blockSignals(True); self.matCombo.clear()
        self.matCombo.addItems(list(self.materials) or list(DESIGN_DEFAULTS))
        i = self.matCombo.findText(cur); self.matCombo.setCurrentIndex(max(i, 0))
        self.matCombo.blockSignals(False)
        self._on_material(set_defaults=i < 0)

    def _on_material(self, set_defaults=True):
        key = self.matCombo.currentText
        if set_defaults and key in DESIGN_DEFAULTS:
            self.cellSpin.value, self.wallSpin.value = DESIGN_DEFAULTS[key]
        self.stageCombo.blockSignals(True); self.stageCombo.clear()
        for st in stages(self.materials.get(key, {})):
            self.stageCombo.addItem(st["etapa"])
        self.stageCombo.blockSignals(False)
        self.stageCombo.enabled = self.stageCombo.count > 1
        self._on_stage()

    def fea_properties(self):
        key = self.matCombo.currentText
        if key not in self.materials:
            return None
        return stage_properties(key, self.materials[key], self.stageCombo.currentText)

    def _on_stage(self):
        pr = self.fea_properties()
        if pr is None:
            return
        txt = (f"E = {fmt(pr['E_MPa'], '{:.0f}')} MPa · ν = {fmt(pr['nu'])} · "
               f"fluencia = {fmt(pr['fluencia_MPa'], '{:.1f}')} MPa")
        if pr["etapa"] != "t0":
            txt += f"  (factor E {fmt(pr['factor_E'])}, factor fluencia {fmt(pr['factor_fluencia'])}; {pr['fuente_etapa']})"
        self.propLabel.text = es(txt)

    # ---------------------------------------------------- ayudas de interfaz
    def _markups_widget(self, base_name, label_fmt, color):
        w = slicer.qSlicerSimpleMarkupsWidget()
        w.setMRMLScene(slicer.mrmlScene)
        w.markupsSelectorComboBox().baseName = base_name
        w.setDefaultNodeColor(qt.QColor.fromRgbF(*color))
        w.enterPlaceModeOnNodeChange = True
        w.tableWidget().setMinimumHeight(70)

        def configure(node):
            if node is None:
                return
            if hasattr(node, "SetMaximumNumberOfControlPoints"):
                node.SetMaximumNumberOfControlPoints(2)
            if hasattr(node, "SetControlPointLabelFormat"):
                node.SetControlPointLabelFormat(label_fmt)
            elif hasattr(node, "SetMarkupLabelFormat"):
                node.SetMarkupLabelFormat(label_fmt)
        w.connect("markupsNodeChanged()", lambda: configure(w.currentNode()))
        return w

    @staticmethod
    def _spin(lo, hi, step, val, suffix):
        s = qt.QDoubleSpinBox(); s.minimum = lo; s.maximum = hi; s.singleStep = step; s.decimals = 2
        s.value = val; s.suffix = suffix
        return s

    def out_dir(self):
        case = self.caseEdit.text.strip()
        if not case or any(c in case for c in '\\/:*?"<>|'):
            raise ValueError("Nombre de caso vacío o con caracteres no válidos.")
        return Path(self.outEdit.currentPath) / case, case

    def log(self, text):
        self.logView.appendPlainText(text)

    def _save_settings(self):
        s = qt.QSettings()
        s.setValue(SETTINGS + "root", self.rootEdit.currentPath)
        s.setValue(SETTINGS + "python", self.pyEdit.currentPath)
        s.setValue(SETTINGS + "out", self.outEdit.currentPath)
        s.setValue(SETTINGS + "case", self.caseEdit.text)

    def _check_order(self, node, pts):
        """P1 debe ser proximal (S mayor). Si no, ofrece intercambiar; el pipeline usa el orden P1 -> P2."""
        if pts[0][2] >= pts[1][2]:
            return pts
        r = qt.QMessageBox.question(
            slicer.util.mainWindow(), "Orden de puntos",
            f"En '{node.GetName()}', P1 está por debajo de P2 (S {pts[0][2]:.1f} < {pts[1][2]:.1f} mm).\n"
            "El pipeline espera P1 proximal y P2 distal. ¿Intercambiar los puntos?",
            qt.QMessageBox.Yes | qt.QMessageBox.No | qt.QMessageBox.Cancel)
        if r == qt.QMessageBox.Cancel:
            raise RuntimeError("Cancelado por el usuario.")
        if r == qt.QMessageBox.Yes:
            a, b = [0.0] * 3, [0.0] * 3
            node.GetNthControlPointPositionWorld(0, a); node.GetNthControlPointPositionWorld(1, b)
            node.SetNthControlPointPositionWorld(0, b); node.SetNthControlPointPositionWorld(1, a)
            return [pts[1], pts[0]]
        return pts

    # ---------------------------------------------------- acciones
    def onRun(self):
        try:
            self._start()
        except Exception as e:  # noqa: BLE001 - se muestra al usuario
            self.status.text = f"No se pudo iniciar: {e}"
            slicer.util.errorDisplay(str(e), windowTitle="Andamio tibial")

    def _start(self):
        if self.proc is not None:
            raise RuntimeError("Ya hay una corrida en curso.")
        self._save_settings()
        root = Path(self.rootEdit.currentPath); py = Path(self.pyEdit.currentPath)
        if not (root / "src" / "pipeline" / "run_case.py").is_file():
            raise FileNotFoundError(f"No se encuentra src/pipeline/run_case.py en {root}")
        if not py.is_file():
            raise FileNotFoundError(f"No se encuentra el Python del pipeline: {py}")
        ct = self.ctSel.currentNode()
        if ct is None:
            raise ValueError("Selecciona la TC.")
        tib, dfn = self.mkTibia.currentNode(), self.mkDefect.currentNode()
        con = self.mkContra.currentNode() if self.useContra.checked else None
        if tib is None or dfn is None:
            raise ValueError("Faltan los puntos de la tibia (P1/P2) o del defecto (D1/D2).")
        if self.useContra.checked and con is None:
            raise ValueError("Marcaste 'usar contralateral' pero no hay puntos contralaterales.")
        if self.refCombo.currentText == "contra" and con is None:
            raise ValueError("La referencia 'contra' requiere los puntos contralaterales.")
        tib_pts = self._check_order(tib, self.logic.points_lps(tib))
        self.logic.points_lps(dfn)
        con_pts = self._check_order(con, self.logic.points_lps(con)) if con is not None else None
        folds = parse_folds(self.foldsEdit.text)
        out, case = self.out_dir()
        out.mkdir(parents=True, exist_ok=True)

        self.logView.clear()
        self.tracker = StageTracker(self._now()); self._render_stages()
        self.status.text = "Preparando entradas…"; slicer.app.processEvents()
        sig = segmentation_signature(self.logic.ct_signature(ct), tib_pts, con_pts,
                                     self.modelCombo.currentText, folds, self.ttaCheck.checked)
        changed = invalidate_cache(out, sig, log=self.log)
        clear_previous_results(out)
        ct_file = self.logic.ct_path(ct, out, reuse_export=not changed)
        self.logic.write_clicks(out / "clicks", tib, dfn, con)

        params = dict(root=root, ct=ct_file, clicks=out / "clicks", out=out, model=self.modelCombo.currentText,
                      reference=self.refCombo.currentText, cell=self.cellSpin.value, wall=self.wallSpin.value,
                      material=self.matCombo.currentText, folds=folds, tta=self.ttaCheck.checked,
                      no_scaffold=True, nail=self.nailCombo.currentText,
                      nail_diameter=self.nailDiam.value or None, plug=self.plugSpin.value or None,
                      grad=[self.gExt.value, self.gInt.value, self.gP1.value, self.gP2.value] if self.gradCheck.checked else None,
                      fab_h=FAB_H_DEFAULT if self.fabFine.checked else None,
                      refine_mode=REFINE_MODES[self.refineCombo.currentIndex][0])
        fea = self.fea_properties()
        if fea is not None:
            (out / "propiedades_fea.json").write_text(json.dumps(fea, indent=1, ensure_ascii=False), encoding="utf-8")
        self._run_ctx = dict(out=out, case=case, ct=ct, py=py, root=root, params=params, step="pipeline",
                             scaffold=not self.noScaffold.checked, cancelled=False,
                             corrida=dict(parametros={k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
                                          fea=fea, correccion_aplicada=False, correccion_motivo="modo desactivado"))
        self._write_corrida()
        self._launch(build_args(params), "Paso 1/2: segmentación y defecto… (en CPU puede tardar varios minutos)")

    def _write_corrida(self):
        ctx = self._run_ctx
        (ctx["out"] / "slicer_corrida.json").write_text(json.dumps(ctx["corrida"], indent=1, ensure_ascii=False,
                                                                   default=str), encoding="utf-8")

    def _launch(self, args, status):
        ctx = self._run_ctx
        self.log(f"> {ctx['py']} " + " ".join(f'"{x}"' if " " in x else x for x in args))
        self.proc = qt.QProcess()
        env = qt.QProcessEnvironment()
        for k, v in self.logic.child_environment().items():
            env.insert(k, v)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(ctx["root"]))
        self.proc.setProcessChannelMode(qt.QProcess.MergedChannels)
        self._buf = ""
        self.proc.start(str(ctx["py"]), args)
        if not self.proc.waitForStarted(10000):
            self.proc = None
            raise RuntimeError("No se pudo lanzar el Python del pipeline.")
        self.runBtn.enabled = False; self.cancelBtn.enabled = True
        self.status.text = status
        self.timer.start()

    def _next_step(self):
        """Tras el paso 1: decide si sigue el andamio y con qué opciones. Devuelve True si lanzó el paso 2."""
        ctx = self._run_ctx
        if ctx["step"] != "pipeline" or not ctx["scaffold"] or ctx["cancelled"]:
            return False
        out = ctx["out"]
        rep = self.logic.read_report(out)
        if not get_path(rep or {}, "defecto.defect_detected"):
            self.log("No se detectó defecto entre D1 y D2: no se genera andamio.")
            return False
        p = dict(ctx["params"])
        apply, why = refine_decision(p["refine_mode"], rep)
        p["refine"] = apply
        ctx["corrida"].update(correccion_aplicada=apply, correccion_motivo=why,
                              correccion_parametros=REFINE_FROZEN if apply else None)
        self._write_corrida()
        self.log(f"Corrección del defecto por densidad: {'sí' if apply else 'no'} ({why}).")
        shutil.copy2(out / "defecto.nii.gz", out / "work" / f"{SCAFFOLD_NAME}.nii.gz")
        ctx["step"] = "andamio"
        try:
            self._launch(build_scaffold_args(p), "Paso 2/2: andamio TPMS y métricas…")
        except Exception as e:  # noqa: BLE001
            self.status.text = f"No se pudo lanzar el andamio: {e}"
            return False
        return True

    def _read_output(self):
        qba = self.proc.readAll()
        try:
            data = qba.data()
        except AttributeError:
            data = bytes(qba)
        text = data if isinstance(data, str) else bytes(data).decode("utf-8", "replace")
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            ln = ln.rstrip("\r").split("\r")[-1]       # barras de progreso (tqdm) -> último estado
            if ln.strip():
                self.log(ln)
                if self.tracker is not None:
                    self.tracker.feed_line(ln, self._now())
        if "\r" in self._buf:                           # progreso en curso: solo en la etiqueta
            self.status.text = self._buf.split("\r")[-2 if self._buf.endswith("\r") else -1].strip()[:160]

    def _poll(self):
        if self.proc is None:
            self.timer.stop(); return
        self._read_output()
        self._render_stages()
        if self.proc.state() != qt.QProcess.NotRunning:
            return
        self._read_output()
        if self._buf.strip():
            self.log(self._buf.split("\r")[-1]); self._buf = ""
        code, crashed = self.proc.exitCode(), self.proc.exitStatus() != qt.QProcess.NormalExit
        self.timer.stop(); self.proc = None
        self.runBtn.enabled = True; self.cancelBtn.enabled = False
        ctx = self._run_ctx
        try:
            (ctx["out"] / "registro_slicer.txt").write_text(self.logView.toPlainText(), encoding="utf-8")
        except OSError:
            pass
        tr, now = self.tracker, self._now()
        if ctx["cancelled"] or crashed or code != 0:
            if tr is not None:
                tr.fail(now); self._render_stages()
            self.status.text = ("Corrida cancelada." if ctx["cancelled"] else
                                f"El paso '{ctx['step']}' terminó con error (código {code}). Revisa el registro (modo avanzado).")
            return
        if self._next_step():
            if tr is not None:
                tr.start(STAGE_ANDAMIO, now); self._render_stages()
            return
        if tr is not None:
            c = tr.current()
            if c is not None:
                tr.finish(c, now)
            tr.start(STAGE_CARGA, now); self._render_stages()
        self.status.text = "Terminado. Cargando resultados…"; slicer.app.processEvents()
        self._load(ctx["out"], ctx["case"], ctx["ct"])
        if tr is not None:
            tr.finish(STAGE_CARGA, self._now()); tr.skip_rest(); self._render_stages()

    def onCancel(self):
        if self.proc is not None:
            self._run_ctx["cancelled"] = True
            self.proc.kill()
            self.log("Corrida cancelada por el usuario.")

    def _load(self, out, case, ct):
        try:
            loaded = self.logic.load_results(out, case, {k: c.checked for k, c in self.optChecks.items()}, ct)
            self.refreshTable()
            isolate_scene(f"Resultados {case}", show_markups=True)
            show_ct_in_slices(ct, self._focus_nodes())
            self._after_load()
            rep = self.logic.read_report(out)
            warn = ""
            if rep and get_path(rep, "defecto.defect_detected") is False:
                warn = " No se detectó defecto entre D1 y D2: no hay andamio."
            elif rep and get_path(rep, "defecto.qc_warning"):
                warn = " AVISO de control de calidad: revisar el defecto antes de usar el andamio."
            self.status.text = f"Cargado: {', '.join(loaded) or 'nada'}.{warn}"
        except Exception as e:  # noqa: BLE001
            self.status.text = f"Error al cargar resultados: {e}"

    def onLoad(self):
        try:
            out, case = self.out_dir()
        except ValueError as e:
            slicer.util.errorDisplay(str(e)); return
        if not (out / "reporte.json").is_file():
            slicer.util.errorDisplay(f"No hay reporte.json en {out}"); return
        self._load(out, case, self.ctSel.currentNode())

    def refreshTable(self):
        try:
            out, _ = self.out_dir()
            rep = self.logic.read_report(out)
        except (ValueError, OSError):
            rep = None
        self.qcLabel.text = es(qc_html(qc_items(rep, (rep or {}).get("_slicer", {}).get("fea"))))
        rows = self.logic.metric_rows(rep, self.allMetrics.checked)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.table.setItem(i, j, qt.QTableWidgetItem(es(val)))
        self.table.resizeColumnsToContents()

    def onExport(self):
        try:
            out, case = self.out_dir()
        except ValueError as e:
            slicer.util.errorDisplay(str(e)); return
        rep = self.logic.read_report(out)
        if rep is None:
            slicer.util.errorDisplay(f"No hay reporte.json en {out}"); return
        dest = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Carpeta de exportación", str(out.parent))
        if not dest:
            return
        dest = Path(dest) / f"{case}_export"
        rows = self.logic.metric_rows(rep, all_rows=False) + self.logic.metric_rows(rep, all_rows=True)
        log_file = out / "registro_slicer.txt"
        log_text = log_file.read_text(encoding="utf-8") if log_file.is_file() else self.logView.toPlainText()
        copied = self.logic.export(out, dest, rows, log_text)
        self.status.text = f"Exportado a {dest}: {', '.join(copied)}"

    def onOpenFolder(self):
        try:
            out, _ = self.out_dir()
        except ValueError:
            return
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(out if out.is_dir() else out.parent)))

    def cleanup(self):
        if self.proc is not None:
            self.proc.kill()
        self.timer.stop()


# ============================================================================ prueba mínima
class AndamioTibialTest(ScriptedLoadableModuleTest):
    """Prueba sin pipeline: argumentos, caché y tabla de métricas (Reload and Test)."""

    def runTest(self):
        import tempfile
        a = build_args(dict(root="R", ct="ct.nii.gz", clicks="c", out="o", model="ft", reference="auto",
                            cell=2.5, wall=0.3, material="Ti6Al4V_LPBF", folds=[0], tta=False, no_scaffold=False,
                            nail="auto", nail_diameter=None, plug=8.0, grad=None))
        assert a[a.index("--plug") + 1] == "8" and "--nail-diameter" not in a and "--tta" not in a
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "work"; (w / "pred").mkdir(parents=True); (w / "roi_0000.nii.gz").write_bytes(b"x")
            s1 = segmentation_signature({"md5": "a"}, [[0, 0, 1], [0, 0, 0]], None, "ft", [0], False)
            assert invalidate_cache(d, s1, log=lambda *_: None)            # caché sin firma -> se borra
            assert not (w / "roi_0000.nii.gz").exists()
            (w / "pred").mkdir(); assert not invalidate_cache(d, s1, log=lambda *_: None)
            s2 = dict(s1, model="base"); assert invalidate_cache(d, s2, log=lambda *_: None)
        g = build_scaffold_args(dict(root="R", out="o", cell=4.2, wall=0.5, material="PCL_bTCP_extrusion",
                                     fab_h=FAB_H_DEFAULT, nail="none", refine=True))
        assert g[g.index("--defect-refine") + 1] == "sector" and "--ct" in g and "--nail" not in g
        assert refine_decision("aviso", {"defecto": {"qc_density_warning": False}})[0] is False
        assert refine_decision("aviso", {"defecto": {"qc_density_warning": True}})[0] is True
        pr = stage_properties("X", dict(E_MPa=100.0, nu=0.4, fluencia_MPa=10.0, etapas_degradacion=[
            dict(etapa="t0", factor_E=1, factor_fluencia=1), dict(etapa="6m", factor_E=0.5, factor_fluencia=None)]), "6m")
        assert pr["E_MPa"] == 50.0 and pr["fluencia_MPa"] is None
        tr = StageTracker(0.0)
        for t, ln in ((1, "[1] TC"), (2, "[2] ROI orientada"), (9, "[5] Caracterización del defecto")):
            tr.feed_line(ln, t)
        assert tr.state[:5] == ["hecho", "hecho", "omitido", "omitido", "en curso"] and tr.dt[1] == 7
        tr.start(STAGE_CARGA, 12); assert tr.state[5] == "pendiente" or tr.state[5] == "omitido"
        qi = qc_items({"tibia": {"fragmentos": 1}, "defecto": {"meets_criteria": False},
                       "andamio": {"f4": {"fabricabilidad": {"chequeos": {"pared": True}, "pared_real": {"p1": 0.27},
                                                             "pared_min_material_mm": 0.2}}}})
        assert [e for e, _ in qi] == ["ok", "aviso", "ok"]
        rows = AndamioTibialLogic.metric_rows({"defecto": {"length_mm": 50.04, "meets_criteria": True}})
        assert ("Defecto", "Longitud (mm)", "50.0") in rows and ("Defecto", "Cumple criterios del proyecto", "sí") in rows
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pf = root / "results" / "pathological"
            (pf / "pat_02_ft").mkdir(parents=True); (pf / "pat_02_ft" / "tibia.stl").write_bytes(b"x")
            (pf / "pat_02_ft" / "defecto.stl").write_bytes(b"x")
            c = pf / "comparacion_materiales_v2" / "pat_02_sis_pcl"; c.mkdir(parents=True)
            (c / "d_envelope.stl").write_bytes(b"x"); (c / "d_scaffold_light.stl").write_bytes(b"x")
            assert viewer_cases(root) == {"Paciente pat_02": ("paciente", "pat_02")}
            it, _ = viewer_items(root, "paciente", "pat_02", "sis", "pcl", 3)
            assert [x[3] for x in it] == ["tibia", "envolvente", "andamio"]
            it, _ = viewer_items(root, "paciente", "pat_02", "sis", "pcl", 1)
            assert [x[3] for x in it] == ["tibia", "defecto"] and it[1][0].name == "defecto.stl"
        with tempfile.TemporaryDirectory() as d:                         # FEA: lectura y escalado
            root = Path(d); w = root / FEA_SUBDIR / "trabajo" / "C_Ti_h0.5_b0.25"; w.mkdir(parents=True)
            S = np.zeros((3, 2, 4), bool); S[1, :, :] = True; S[0, 0, 1:3] = True
            np.savez(w / "campo.npz", S=S, vm=np.arange(S.sum(), dtype=np.float32), hs=np.array([0.5, 0.5, 0.5]),
                     prox=-1, F_ref=2.0, lr_col=1, ap_col=0)
            (w.parent / "C_Ti_h1_b0.25").mkdir(); np.savez(w.parent / "C_Ti_h1_b0.25" / "campo.npz", S=S)
            (root / FEA_SUBDIR / "resultados_macro.csv").write_text(
                "caso,carga,F_N,etapa,h_mm,vm_macro_max_MPa,fuera_rango_lineal\nC_Ti,moderada_1432N,1432.0,t0,0.5,9,False\n", encoding="utf-8")
            runs = fea_runs(root); assert list(runs) == ["Caso C (demostrativo) · Ti6Al4V"] and len(fea_runs(root, False)) == 2
            rows = fea_table(root); assert fea_loads(rows) == [(1432.0, "moderada_1432N")]
            assert fea_row(rows, "C_Ti", 1432.0, "t0", 0.5)["vm_macro_max_MPa"] == "9"
            ijk, conn = hex_mesh(S); assert conn.shape == (S.sum(), 8) and conn.max() == len(ijk) - 1
            u = root / FEA_SUBDIR / FEA_U_SUBDIR / "C_Ti_h0.5_b0.25"; u.mkdir(parents=True)
            np.savez(u / "campo_u.npz", S=S, u=np.tile([0.0, 0.0, 1.0], (len(ijk), 1)), F_ref=4.0)
            fld = fea_field(runs["Caso C (demostrativo) · Ti6Al4V"]["npz"], 4.0, 0.5, fea_runs(root)["Caso C (demostrativo) · Ti6Al4V"]["npz_u"])
            assert abs(fld["vm_max"] - 2 * (S.sum() - 1)) < 1e-6           # VM ~ F/F_ref
            assert np.allclose(fld["u"], [0, 0, -2.0])                       # u ~ F/(F_ref·fE), girado: proximal arriba
        assert es("E = 110000.0 MPa, p99 0.446 mm, roi_0000.nii.gz") == "E = 110000,0 MPa, p99 0,446 mm, roi_0000.nii.gz"
        assert case_name_from("pat_02_CT_HU") == "pat_02" and case_name_from("Paciente 3") == "Paciente_3"
        pts = np.array([[0.0, 0, 0], [0, 0, 50.0]])
        assert fea_suggested_exag(dict(points=pts, u=np.zeros((2, 3)), u_max=0.06)) == 50
        assert fea_suggested_exag(dict(points=pts, u=np.zeros((2, 3)), u_max=18.5)) == 1
        self.delayDisplay("Prueba AndamioTibial OK")
