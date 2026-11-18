#!/usr/bin/env bash
# Ajuste fino del fold 0 con defectos REALISTAS (generador v2) + benchmark base vs ajustado, en un pod
# RunPod (RTX 4090). Supone el mismo volumen usado para el entrenamiento original:
#   /workspace/nnUNet_raw, /workspace/nnUNet_preprocessed, /workspace/nnUNet_results
# y que esta carpeta (finetune/) se copió a /workspace/finetune con:
#   make_defect_dataset_v2.py  benchmark_v2.py  virtual_defects_v2.py  defect_core.py  nnUNetTrainer_FT.py
# Costo estimado de GPU: preprocesado ~40 min + 250 épocas x ~45 s ~ 3,1 h + inferencia del benchmark
# (2 modelos) ~30 min  ->  ~4-4,5 h.  La generación del benchmark corre en CPU durante el entrenamiento.
set -euo pipefail

export nnUNet_raw=/workspace/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export nnUNet_results=/workspace/nnUNet_results
FOLD=0
SRC=502
DST=503
DST_NAME=Dataset503_TibiaDefectsF${FOLD}
PLANS=nnUNetResEncUNetMPlans
CKPT=$nnUNet_results/Dataset502_TibiaFischer/nnUNetTrainer__${PLANS}__3d_fullres/fold_${FOLD}/checkpoint_final.pth
NCPU=$(nproc)

cd /workspace/finetune
python -c "import nnunetv2, torch; print('nnunetv2 OK; GPU:', torch.cuda.is_available())"
test -f "$CKPT" || { echo "Falta $CKPT"; exit 1; }
for f in make_defect_dataset_v2.py benchmark_v2.py virtual_defects_v2.py defect_core.py nnUNetTrainer_FT.py; do
  test -f "$f" || { echo "Falta $f en /workspace/finetune"; exit 1; }
done

# 1) dataset con defectos realistas (CPU): intacta + 6 defectos por tibia, mezcla A/B/C/D
python make_defect_dataset_v2.py --fold $FOLD --n 6 --workers $((NCPU > 2 ? NCPU - 1 : 1)) | tee make_dataset.log

# 2) mismos planes que el modelo original (arquitectura y normalización idénticas)
nnUNetv2_extract_fingerprint -d $DST -np 8
nnUNetv2_move_plans_between_datasets -s $SRC -t $DST -sp $PLANS -tp $PLANS
nnUNetv2_preprocess -d $DST -plans_name $PLANS -c 3d_fullres -np 8
cp splits_final_Dataset503.json $nnUNet_preprocessed/$DST_NAME/splits_final.json

# 3) trainer de ajuste fino
TR_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer'))")
cp nnUNetTrainer_FT.py "$TR_DIR/"

# 4) benchmark: se genera en CPU (en segundo plano) mientras la GPU entrena
nice -n 19 python benchmark_v2.py generate --fold $FOLD > benchmark_generate.log 2>&1 &
BG=$!

# 5) entrenamiento desde los pesos del fold original
nnUNetv2_train $DST 3d_fullres $FOLD -p $PLANS -tr nnUNetTrainer_FT -pretrained_weights "$CKPT" --npz

# 6) inferencia del benchmark con ambos modelos (mismas entradas) y evaluación
wait $BG || { echo "Falló la generación del benchmark: ver benchmark_generate.log"; exit 1; }
python benchmark_v2.py predict  --fold $FOLD --models base ft
python benchmark_v2.py evaluate --fold $FOLD --models base ft | tee benchmark_evaluate.log

echo "Listo:"
echo "  modelo: $nnUNet_results/$DST_NAME/nnUNetTrainer_FT__${PLANS}__3d_fullres/fold_${FOLD}/checkpoint_final.pth"
echo "  benchmark: /workspace/finetune/benchmark_v2_f${FOLD}/results_summary.json y results_rows.csv"
echo "Apaga el pod después de descargar esos dos archivos y la carpeta fold_${FOLD}."
