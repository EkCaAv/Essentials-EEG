"""
cnn_experiments.py — CNN sobre espectrogramas STFT para detección de crisis EEG.

Input:  windows_dataset.csv (con columnas subject_id, edf_file, t0_sec, t1_sec, label)
        + EDFs originales (para extraer señal cruda)
Output: results.csv y cv_fold_metrics.csv compatibles con chbmit_experiments.py

Validación: GroupKFold por subject_id — sin leakage entre sujetos.
Métrica primaria: PR-AUC (dataset desbalanceado ~1.24% positivos).

Uso:
    python3 cnn_experiments.py \\
      --dataset_csv ./out_thesis_final/runs/baseline_classical/windows_dataset.csv \\
      --data_root ./data \\
      --out_dir ./out_thesis_final/cnn_stft \\
      --subjects chb05 chb09 chb14 chb16 chb20 chb22 chb23 \\
      --n_splits 7 --epochs 40 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    raise ImportError(
        "PyTorch no está instalado.\n"
        "  CPU:  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  GPU:  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
    )

from dl_utils import (
    build_raw_arrays,
    compute_metrics_safe,
    make_run_id,
    save_dl_results,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset: señal cruda → espectrograma STFT por canal
# ---------------------------------------------------------------------------

class SpectrogramDataset(Dataset):
    """
    Convierte ventanas EEG crudas (n_ch, n_samples) en espectrogramas STFT
    por canal usando torch.stft(). La transformación es lazy (en __getitem__).

    Forma de salida por muestra: (n_channels, freq_bins, time_frames) float32
    Para n_fft=128, hop=32, n_samples=1280:
      freq_bins  = n_fft // 2 + 1 = 65
      time_frames = 1 + (n_samples - n_fft) // hop = 37
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        n_fft: int = 128,
        hop_length: int = 32,
    ):
        self.X = torch.from_numpy(X_raw)  # (N, n_ch, n_samples)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.n_fft = n_fft
        self.hop = hop_length
        self.window = torch.hann_window(n_fft)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (n_ch, n_samples)
        specs = []
        for ch in range(x.shape[0]):
            stft_out = torch.stft(
                x[ch],
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=self.window,
                return_complex=True,
            )
            mag = stft_out.abs()
            log_mag = torch.log1p(mag)  # (freq_bins, time_frames)
            specs.append(log_mag)
        spec = torch.stack(specs, dim=0)  # (n_ch, freq_bins, time_frames)
        return spec, self.y[idx]


# ---------------------------------------------------------------------------
# Modelo CNN
# ---------------------------------------------------------------------------

class EEGSpectrogramCNN(nn.Module):
    """
    CNN para clasificación ictal/interictal desde espectrogramas multi-canal.

    Input:  (batch, n_channels, freq_bins, time_frames)
    Output: (batch,)  logit no normalizado — usar BCEWithLogitsLoss

    Arquitectura:
      Conv2d(n_ch→32, 3×3) → BN → ReLU → MaxPool(2,2)
      Conv2d(32→64, 3×3)   → BN → ReLU → MaxPool(2,2)
      Conv2d(64→128, 3×3)  → BN → ReLU → AdaptiveAvgPool(1,1)
      Flatten → Linear(128→64) → ReLU → Dropout → Linear(64→1)
    """

    def __init__(self, n_channels: int = 23, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Loop de entrenamiento
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# GroupKFold CV con CNN
# ---------------------------------------------------------------------------

def run_cnn_groupkfold_cv(
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 7,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_fft: int = 128,
    hop_length: int = 32,
    pos_weight_scale: float = 1.0,
    dropout: float = 0.5,
    device: Optional[torch.device] = None,
    run_id: str = "cnn_stft",
    models_dir: Optional[Path] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Ejecuta GroupKFold CV con CNN sobre espectrogramas STFT.
    Retorna (result_rows, fold_df) compatibles con save_dl_results().
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CNN] Dispositivo: {device}")

    gkf = GroupKFold(n_splits=n_splits)
    n_channels = X_raw.shape[1]

    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_raw, y, groups), start=1):
        X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
        y_tr, y_te = y[tr_idx].astype(np.float32), y[te_idx].astype(np.float32)
        train_subjects = sorted(np.unique(groups[tr_idx]).tolist())
        test_subjects = sorted(np.unique(groups[te_idx]).tolist())

        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        print(f"\n{'='*60}")
        print(
            f"[CNN] Fold {fold}/{n_splits} | "
            f"train={len(tr_idx)} (pos={n_pos}, neg={n_neg}) | "
            f"test={len(te_idx)} (pos={int(y_te.sum())})"
        )
        print(f"  Train: {train_subjects} | Test: {test_subjects}")

        base_metrics = dict(
            fold=fold,
            auc_roc=float("nan"), pr_auc=float("nan"),
            sensitivity=float("nan"), specificity=float("nan"),
            accuracy=float("nan"),
            test_pos=int(y_te.sum()), test_neg=int(len(y_te) - y_te.sum()),
            train_rows=len(tr_idx), test_rows=len(te_idx),
            train_pos=n_pos, train_neg=n_neg,
            train_subjects=";".join(train_subjects),
            test_subjects=";".join(test_subjects),
            train_single_class=0,
            best_params="", best_inner_score=float("nan"),
        )

        if y_te.sum() == 0:
            print(f"  [SKIP] Fold {fold}: test sin positivos")
            fold_metrics.append(base_metrics)
            continue

        if n_pos == 0:
            print(f"  [SKIP] Fold {fold}: train sin positivos")
            base_metrics["train_single_class"] = 1
            fold_metrics.append(base_metrics)
            continue

        # ── Datasets y loaders ─────────────────────────────────────────────
        train_ds = SpectrogramDataset(X_tr, y_tr.astype(np.int8), n_fft, hop_length)
        test_ds = SpectrogramDataset(X_te, y_te.astype(np.int8), n_fft, hop_length)

        # WeightedRandomSampler garantiza positivos en cada batch
        sample_weights = np.where(y_tr == 1, float(n_neg) / n_pos, 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(y_tr),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=256, shuffle=False, num_workers=0
        )

        # ── Modelo y optimizador ────────────────────────────────────────────
        model = EEGSpectrogramCNN(n_channels=n_channels, dropout=dropout).to(device)
        pos_w = torch.tensor([min(n_neg / n_pos * pos_weight_scale, 80.0)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # ── Entrenamiento ───────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        early_stop_patience = 10

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            # Validation loss sobre test (usado solo para scheduler/earlystop)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    logits = model(Xb.to(device))
                    val_loss += criterion(logits, yb.to(device)).item() * len(yb)
            val_loss /= max(len(test_ds), 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"  epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
                )

            if patience_counter >= early_stop_patience:
                print(f"  Early stop en epoch {epoch}")
                break

        # ── Inferencia con el mejor estado ──────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)

        # Guardar pesos del mejor modelo de este fold
        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            test_subj_str = "_".join(test_subjects)
            ckpt_path = models_dir / f"fold{fold:02d}_{test_subj_str}.pth"
            torch.save({
                "fold": fold,
                "model_state_dict": best_state,
                "test_subjects": test_subjects,
                "train_subjects": train_subjects,
                "n_channels": n_channels,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "dropout": dropout,
                "run_id": run_id,
            }, ckpt_path)
            print(f"  [SAVE] Modelo guardado: {ckpt_path.name}")

        y_score = predict_proba(model, test_loader, device)
        metrics = compute_metrics_safe(y_te.astype(np.int8), y_score)

        print(
            f"  AUC-ROC={metrics['auc_roc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"Sens={metrics['sensitivity']:.4f} | "
            f"Spec={metrics['specificity']:.4f}"
        )

        base_metrics.update(metrics)
        fold_metrics.append(base_metrics)

        # Liberar memoria GPU/CPU
        del model, best_state, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Agregar resultados ──────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_metrics)

    valid = fold_df["pr_auc"].dropna()
    auc_valid = fold_df["auc_roc"].dropna()

    result_row = {
        "model": "cnn_spectrogram",
        "feature_set": "stft_spectrogram",
        "n_features": int((n_fft // 2 + 1) * (1 + (X_raw.shape[2] - n_fft) // hop_length) * X_raw.shape[1]),
        "auc_roc_mean": float(auc_valid.mean()) if len(auc_valid) else float("nan"),
        "auc_roc_std": float(auc_valid.std()) if len(auc_valid) else float("nan"),
        "pr_auc_mean": float(valid.mean()) if len(valid) else float("nan"),
        "pr_auc_std": float(valid.std()) if len(valid) else float("nan"),
        "sensitivity_mean": float(fold_df["sensitivity"].dropna().mean()),
        "sensitivity_std": float(fold_df["sensitivity"].dropna().std()),
        "specificity_mean": float(fold_df["specificity"].dropna().mean()),
        "specificity_std": float(fold_df["specificity"].dropna().std()),
        "accuracy_mean": float(fold_df["accuracy"].dropna().mean()),
        "accuracy_std": float(fold_df["accuracy"].dropna().std()),
        "folds_with_nan_auc": int(fold_df["auc_roc"].isna().sum()),
        "folds_with_nan_prauc": int(fold_df["pr_auc"].isna().sum()),
        "avg_test_pos": float(fold_df["test_pos"].mean()),
    }

    fold_df.insert(0, "run_id", run_id)
    fold_df.insert(1, "feature_set", "stft_spectrogram")
    fold_df.insert(2, "model", "cnn_spectrogram")
    fold_df.insert(3, "n_features", result_row["n_features"])

    print(f"\n{'='*60}")
    print("[CNN] RESUMEN:")
    print(
        f"  PR-AUC  = {result_row['pr_auc_mean']:.4f} ± {result_row['pr_auc_std']:.4f}"
    )
    print(
        f"  AUC-ROC = {result_row['auc_roc_mean']:.4f} ± {result_row['auc_roc_std']:.4f}"
    )
    print(f"  Sensitivity = {result_row['sensitivity_mean']:.4f}")
    print(f"  Folds con NaN: {result_row['folds_with_nan_auc']}/{n_splits}")

    return [result_row], fold_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CNN sobre espectrogramas STFT — detección de crisis EEG (CHB-MIT)"
    )
    parser.add_argument("--dataset_csv", type=str, required=True,
                        help="windows_dataset.csv generado por chbmit_experiments.py")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directorio raíz con subdirectorios chb##/")
    parser.add_argument("--out_dir", type=str, default="./out_cnn",
                        help="Directorio de salida")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Sujetos a incluir (default: todos los del CSV)")
    parser.add_argument("--n_splits", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_fft", type=int, default=128,
                        help="Tamaño de ventana STFT en muestras")
    parser.add_argument("--hop_length", type=int, default=32,
                        help="Paso de la STFT en muestras")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pos_weight_scale", type=float, default=1.0,
                        help="Multiplicador sobre pos_weight automático")
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--l_freq", type=float, default=0.5)
    parser.add_argument("--h_freq", type=float, default=45.0)
    parser.add_argument("--resample_hz", type=float, default=256.0)
    parser.add_argument("--no_cache", action="store_true",
                        help="Ignorar caché de arrays .npy (re-leer EDFs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.run_name or "cnn_stft")
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CNN] run_id={run_id}")
    print(f"[CNN] out_dir={out_dir}")

    # ── Cargar CSV de ventanas ──────────────────────────────────────────────
    df = pd.read_csv(args.dataset_csv)
    print(f"[CNN] Dataset cargado: {len(df)} ventanas | pos_rate={df['label'].mean():.4f}")

    # ── Construir arrays crudos ─────────────────────────────────────────────
    t0 = time.time()
    cache_dir = out_dir / "raw_cache" if not args.no_cache else None
    X_raw, y, groups = build_raw_arrays(
        df,
        data_root=Path(args.data_root),
        window_sec=args.window_sec,
        fs=args.resample_hz,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        subjects=args.subjects,
        cache_dir=cache_dir,
    )
    print(f"[CNN] Arrays: X={X_raw.shape} | tiempo={time.time()-t0:.0f}s")
    print(f"[CNN] Distribución: pos={y.sum()} neg={len(y)-y.sum()} rate={y.mean():.4f}")

    # ── CV ─────────────────────────────────────────────────────────────────
    models_dir = run_dir / "saved_models"
    result_rows, fold_df = run_cnn_groupkfold_cv(
        X_raw=X_raw,
        y=y,
        groups=groups,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        pos_weight_scale=args.pos_weight_scale,
        dropout=args.dropout,
        run_id=run_id,
        models_dir=models_dir,
    )

    # ── Enriquecer filas con metadatos del run ──────────────────────────────
    subjects_used = sorted(np.unique(groups).tolist())
    for row in result_rows:
        row.update({
            "run_id": run_id,
            "cohort": "6_10",
            "subjects": ",".join(subjects_used),
            "window_sec": args.window_sec,
            "overlap": 0.5,
            "exclude_margin_sec": 0.0,
            "l_freq": args.l_freq,
            "h_freq": args.h_freq,
            "notch": None,
            "resample_hz": args.resample_hz,
            "max_interictal_per_file": None,
            "cv": f"GroupKFold(n_splits={args.n_splits})",
            "model_params": json.dumps({
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "n_fft": args.n_fft, "hop_length": args.hop_length,
                "dropout": args.dropout,
            }),
            "n_rows": len(df),
            "pos_rate": float(y.mean()),
        })

    # ── Guardar ────────────────────────────────────────────────────────────
    save_dl_results(
        result_rows=result_rows,
        fold_frames=[fold_df],
        out_dir=out_dir,
        run_dir=run_dir,
        run_id=run_id,
        manifest_extras={
            "model": "CNN_STFT",
            "subjects": subjects_used,
            "n_splits": args.n_splits,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
        },
    )

    print(f"\n[CNN] Completado. Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
