"""
Trainer de ajuste fino (fine-tuning) para nnU-Net v2.

Igual que nnUNetTrainer, pero:
  - 250 épocas (en vez de 1000)
  - tasa de aprendizaje inicial 1e-3 (en vez de 1e-2), con el mismo
    decaimiento polinomial -> no "olvida" lo aprendido con tibias intactas.

Copiar a: <site-packages>/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_FT.py
Uso: nnUNetv2_train 503 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_FT
         -pretrained_weights <.../Dataset502.../fold_0/checkpoint_final.pth>
"""

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_FT(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.initial_lr = 1e-3
