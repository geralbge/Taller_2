"""
Auditoría de entrenamiento nnU-Net v2 (Dataset502_TibiaFischer).

Extrae, de cada checkpoint .pth local, el historial que nnU-Net guarda dentro
(clave 'logging'): pérdida de entrenamiento, pérdida de validación, pseudo-Dice
de validación y EMA del pseudo-Dice, por época. Con eso se evalúa si hubo
sobreajuste (validación empeora mientras entrenamiento sigue bajando).

No modifica ningún checkpoint. Solo lee.

Uso (PowerShell, desde la raíz del repositorio, con el entorno de nnU-Net):
    .\\.venv_nnunet\\Scripts\\python.exe src\\FISCHER\\audit_nnunet_training.py

Dependencias: torch, matplotlib (si falta: pip install matplotlib)

Salidas en results/evaluation/segmentation/training_audit/:
    - fold_X_<archivo>_log.csv   historial por época
    - fold_X_<archivo>_curves.png curvas train/val loss y pseudo-Dice
    - resumen_auditoria.csv        una fila por checkpoint
"""

from pathlib import Path
import csv
import gc

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("AVISO: matplotlib no instalado; se generarán CSV sin gráficos.")

ROOT = Path(__file__).resolve().parents[2]  # raíz del repositorio
RUNPOD = ROOT / "runpod"
OUT = ROOT / "results" / "evaluation" / "segmentation" / "training_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Checkpoints locales encontrados (fold 0 = el de la raíz de runpod/)
CANDIDATES = [
    ("0", RUNPOD / "checkpoint_best.pth"),
    ("1", RUNPOD / "fold_1" / "checkpoint_best.pth"),
    ("2", RUNPOD / "fold_2" / "checkpoint_best.pth"),
    ("3", RUNPOD / "fold_3" / "checkpoint_best.pth"),
    ("3", RUNPOD / "fold_3" / "checkpoint_latest.pth"),
    ("4", RUNPOD / "fold_4" / "checkpoint_latest.pth"),
]
# Si existen checkpoint_final.pth en cualquier carpeta, también se auditan
for p in sorted(RUNPOD.rglob("checkpoint_final.pth")):
    fold = p.parent.name.replace("fold_", "") if p.parent.name.startswith("fold_") else "0"
    CANDIDATES.append((fold, p))

LOG_KEYS = ["train_losses", "val_losses", "mean_fg_dice", "ema_fg_dice", "lrs"]


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return float(x.item())
        except Exception:
            return None


def window_mean(vals, a, b):
    seg = [v for v in vals[a:b] if v is not None]
    return sum(seg) / len(seg) if seg else None


summary_rows = []

for fold, path in CANDIDATES:
    if not path.is_file():
        print(f"[fold {fold}] no existe: {path}")
        continue

    print(f"\n[fold {fold}] leyendo {path.relative_to(ROOT)} ...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # liberar pesos de la red, solo necesitamos el historial
    for k in ("network_weights", "optimizer_state", "grad_scaler_state"):
        ckpt.pop(k, None)
    gc.collect()

    current_epoch = ckpt.get("current_epoch")
    best_ema = ckpt.get("_best_ema")
    trainer = ckpt.get("trainer_name")
    logging = ckpt.get("logging", {}) or {}

    series = {k: [to_float(v) for v in logging.get(k, [])] for k in LOG_KEYS}
    n = max((len(v) for v in series.values()), default=0)
    print(f"  trainer={trainer}  current_epoch={current_epoch}  _best_ema={best_ema}  épocas_en_log={n}")

    tag = f"fold_{fold}_{path.stem}"
    with open(OUT / f"{tag}_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + LOG_KEYS)
        for i in range(n):
            w.writerow([i] + [series[k][i] if i < len(series[k]) else "" for k in LOG_KEYS])

    tr, va, dice = series["train_losses"], series["val_losses"], series["mean_fg_dice"]
    ema = series["ema_fg_dice"]

    # Indicadores simples de sobreajuste
    best_val_epoch = None
    if va:
        valid = [(i, v) for i, v in enumerate(va) if v is not None]
        best_val_epoch, best_val = min(valid, key=lambda t: t[1])
    else:
        best_val = None
    best_ema_epoch = None
    if ema:
        valid = [(i, v) for i, v in enumerate(ema) if v is not None]
        best_ema_epoch = max(valid, key=lambda t: t[1])[0]

    last50 = slice(max(0, n - 50), n)
    row = {
        "fold": fold,
        "checkpoint": path.name,
        "current_epoch": current_epoch,
        "epochs_logged": n,
        "best_ema_pseudo_dice": round(best_ema, 4) if best_ema is not None else "",
        "epoch_best_ema": best_ema_epoch,
        "epoch_min_val_loss": best_val_epoch,
        "min_val_loss": round(best_val, 4) if best_val is not None else "",
        "train_loss_last50": round(window_mean(tr, last50.start, last50.stop), 4) if tr else "",
        "val_loss_last50": round(window_mean(va, last50.start, last50.stop), 4) if va else "",
        "pseudo_dice_last50": round(window_mean(dice, last50.start, last50.stop), 4) if dice else "",
    }
    if tr and va and n > 0:
        row["gap_val_minus_train_last50"] = round(row["val_loss_last50"] - row["train_loss_last50"], 4)
    summary_rows.append(row)
    for k, v in row.items():
        print(f"  {k}: {v}")

    if HAS_MPL and n > 0:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ep = range(n)
        if tr:
            ax1.plot(ep[: len(tr)], tr, label="train loss", lw=1)
        if va:
            ax1.plot(ep[: len(va)], va, label="val loss", lw=1)
        ax1.set_xlabel("época")
        ax1.set_ylabel("loss (Dice+CE, negativa = mejor)")
        ax2 = ax1.twinx()
        if dice:
            ax2.plot(ep[: len(dice)], dice, label="pseudo-Dice val", lw=0.6, alpha=0.4, color="green")
        if ema:
            ax2.plot(ep[: len(ema)], ema, label="EMA pseudo-Dice", lw=1.5, color="darkgreen")
        ax2.set_ylabel("pseudo-Dice")
        ax2.set_ylim(0, 1)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center right")
        ax1.set_title(f"Dataset502 – fold {fold} – {path.name} (época {current_epoch})")
        fig.tight_layout()
        fig.savefig(OUT / f"{tag}_curves.png", dpi=130)
        plt.close(fig)

    del ckpt
    gc.collect()

if summary_rows:
    keys = list(summary_rows[0].keys())
    for r in summary_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / "resumen_auditoria.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nListo. Resultados en: {OUT}")
else:
    print("\nNo se encontró ningún checkpoint.")
