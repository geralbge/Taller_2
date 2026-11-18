# Prototipo computacional de andamios tibiales a partir de TC

Código del Trabajo de Grado II de Ingeniería Biomédica, Universidad Católica Boliviana "San Pablo", La Paz:

> **Prototipo de sistema computacional para el diseño y verificación estructural de andamios para defectos óseos segmentarios críticos tibiales a partir de imágenes de TC.**
> Geraldy Briseika Guzmán Espinoza.

**Aviso.** Es un prototipo computacional de investigación. No es un dispositivo médico, no está validado para uso clínico y sus resultados no deben usarse para decisiones de tratamiento. El análisis por elementos finitos usa cargas y condiciones simplificadas y es una verificación computacional, no una demostración de seguridad mecánica.

## Flujo del sistema

```
TC (DICOM / NIfTI, HU)
  └─ [1] ROI orientada de la tibia (puntos P1, P2 en 3D Slicer)
      └─ [2] Segmentación con nnU-Net v2 (modelo base Dataset502 o ajustado Dataset503)
          └─ [3] Post-procesamiento y reconstrucción 3D (STL, LPS, mm)
              └─ [4] Caracterización del defecto entre D1 y D2
                     (referencia por interpolación o contralateral, criterios del proyecto)
                  └─ [5] Andamio poroso TPMS (giroide-lámina) dentro del defecto
                         + métricas de porosidad, espesor y verificación geométrica
                      └─ [6] Verificación estructural con FEBio en dos escalas
                             (escala 1: celda → E_eff; escala 2: andamio homogenizado)
```

Criterios del defecto usados: longitud aproximada 2,5–7 cm, pérdida cortical > 50 %, sin compromiso articular y no multisegmentario.

## Contenido

Los archivos conservan las rutas relativas del proyecto original, porque los scripts se importan entre sí y calculan sus rutas desde la raíz del repositorio. La tabla los agrupa como en el anexo del documento.

### A. Sistema (pipeline)

| Archivo | Función |
|---|---|
| `src/pipeline/run_case.py` | Orquestador: TC + puntos → segmentación → defecto → andamio, con reporte JSON |
| `src/FISCHER/create_oriented_tibia_roi.py` | ROI orientada según el eje largo de la tibia |
| `src/reconstruction/reconstruct_tibia.py` | Post-procesamiento de la segmentación y superficie 3D |
| `src/defect/defect_core.py` | Forma de referencia por interpolación y caracterización del defecto |
| `src/defect/contralateral.py` | Referencia contralateral reflejada y registrada |
| `src/defect/defect_refine.py` | Corrección opcional del defecto por densidad (TC) |
| `src/scaffold/generate_scaffold.py` | Generación del andamio y métricas |
| `src/scaffold/scaffold_core.py` | Superficies TPMS y métricas de arquitectura porosa |
| `src/scaffold/scaffold_f4.py` | Envolvente, gradiente radial, materiales y fabricabilidad |

### B. Módulo de 3D Slicer

| Archivo | Función |
|---|---|
| `slicer/AndamioTibial/AndamioTibial.py` | Interfaz en 3D Slicer que ejecuta el pipeline y muestra resultados y FEA |
| `slicer/AndamioTibial/herramientas/fea_desplazamientos.py` | Campo de desplazamientos para la deformada en el visor |

### C. Verificación estructural (FEA, FEBio 4)

| Archivo | Función |
|---|---|
| `results/fea/celda_febio/celda_febio.py` | Escala 1: módulo efectivo de la celda |
| `results/fea/macro_febio/macro_febio.py` | Escala 2: andamio homogenizado y esfuerzo local |
| `results/fea/macro_febio/informe_macro.py` | Tablas y figuras del FEA |
| `results/fea/macro_febio/sens_apertura.py` | Sensibilidad al radio de apertura del canal |
| `results/fea/celda_febio/benchmark_luo/benchmark_luo.py` | Comparación de la celda con Luo et al. (2023) |

### D. Evaluación

| Archivo | Resultado |
|---|---|
| `src/DRYAD/evaluate_filled_geometry.py`, `src/DRYAD/evaluate_dryad_external.py` | Segmentación: validación cruzada (Fischer) y prueba externa (DRYAD) |
| `src/defect/evaluate_contralateral.py` (usa `src/defect/virtual_defects.py`) | Interpolación vs contralateral |
| `src/finetune/benchmark_v2.py`, `src/finetune/eval_f3.py` (usan `src/finetune/defect_core.py`) | Delimitación del defecto, modelo base vs ajustado, Wilcoxon |
| `src/FISCHER/audit_nnunet_training.py` | Auditoría de sobreajuste del entrenamiento |
| `src/defect/eval_manual_reference.py` | Casos patológicos vs segmentación manual |
| `results/pathological/pat_02_manual_eval/final/p9_relleno.py`, `results/pathological/pat_02_manual_eval/p4_7_analisis.py` | Verificación independiente del caso pat_02 |
| `results/pathological/comparacion_materiales_v2/resumen_v2.py`, `tanda_corregidos.py` | Andamios de los casos patológicos por material |
| `results/evaluation/scaffold/demostrativos_v2/tanda_demo.py` | Casos demostrativos virtuales |

### E. Datos y entrenamiento

| Archivo | Función |
|---|---|
| `src/FISCHER/prepare_preprocess/prepare_fischer_gt_rois.py` | ROI y etiquetas del dataset de Fischer et al. |
| `src/FISCHER/prepare_nnunet/build_dataset502.py` | Dataset nnU-Net 502 |
| `src/FISCHER/prepare_nnunet/create_subject_splits.py` | Partición en 5 folds por sujeto |
| `runpod/train_runpod.sh` | Entrenamiento base en GPU (RunPod) |
| `src/finetune/virtual_defects_v2.py` | Generador de defectos virtuales realistas (escenarios A–D) |
| `src/finetune/make_defect_dataset_v2.py` | Dataset 503 para el ajuste fino |
| `src/finetune/nnUNetTrainer_FT.py` | Trainer de ajuste fino (250 épocas, lr 1e-3) |
| `resultados_ft/pod_finetune/runpod_finetune.sh` | Ajuste fino y benchmark en GPU |

Las etiquetas F1 a F4 que aparecen en comentarios y nombres de archivo corresponden a fases internas del desarrollo (F1–F2 generador y ajuste fino, F3 forma del defecto, F4 parámetros del andamio).

## Qué no incluye

- **Ninguna imagen ni dato de pacientes.** `pat_01` y `pat_02` son seudónimos de dos casos clínicos usados en la demostración; sus TC, máscaras y resultados no se publican.
- **Ningún dato de terceros.** Las TC y segmentaciones del dataset de Fischer et al. y de DRYAD se descargan de sus fuentes originales, bajo sus propias licencias.
- **Pesos de los modelos nnU-Net** (Dataset502 y Dataset503). Se esperan en `data/nnunet/nnUNet_results/` y `models/` respectivamente.

## Instalación

Probado en Windows 10/11 con Python 3.12.

```powershell
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\pip install -r requirements.txt
```

Herramientas externas:

- **3D Slicer** (5.x), para colocar los puntos y usar el módulo. Agregar `slicer/AndamioTibial` en *Edit → Application Settings → Modules → Additional module paths*.
- **FEBio 4** (FEBio Studio). Si no está en `C:\Program Files\FEBioStudio\bin\febio4.exe`, definir la variable de entorno `FEBIO_EXE`.
- Para el entrenamiento: una GPU con CUDA. El entrenamiento del trabajo se hizo en RunPod.

## Uso básico

```powershell
.\.venv_nnunet\Scripts\python.exe src\pipeline\run_case.py --ct <TC.nii.gz | carpeta DICOM> --clicks <carpeta de puntos> --out <carpeta de salida>
```

La carpeta de puntos contiene los archivos de Markups de 3D Slicer `Tibia_run1.mrk.json` (P1 proximal, P2 distal) y `Defect_run1.mrk.json` (D1 y D2, hueso sano por encima y por debajo del defecto); `Tibia_contra_run1.mrk.json` es opcional. Las opciones de material, celda, pared, clavo y gradiente están descritas al inicio de `run_case.py`.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). La licencia cubre solo el código; los datasets de terceros mantienen sus propias licencias y no se incluyen datos de pacientes.

## Cita

Guzmán Espinoza, G. B. (2026). *Prototipo de sistema computacional para el diseño y verificación estructural de andamios para defectos óseos segmentarios críticos tibiales a partir de imágenes de TC* [Trabajo de Grado, Universidad Católica Boliviana "San Pablo"].
