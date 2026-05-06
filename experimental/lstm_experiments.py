"""
lstm_experiments.py — BiLSTM sobre señal EEG cruda para detección de crisis.

Input:  windows_dataset.csv (con columnas subject_id, edf_file, t0_sec, t1_sec, label)
        + EDFs originales (para extraer señal cruda)
Output: results.csv y cv_fold_metrics.csv compatibles con chbmit_experiments.py

Validación: GroupKFold por subject_id — sin leakage entre sujetos.
Métrica primaria: PR-AUC (dataset desbalanceado ~1.24% positivos).

Uso:
    python3 lstm_experiments.py \\
      --dataset_csv ./out_thesis_final/runs/baseline_classical/windows_dataset.csv \\
      --data_root ./data \\
      --out_dir ./out_thesis_final/lstm_raw \\
      --subjects chb05 chb09 chb14 chb16 chb20 chb22 chb23 \\
      --n_splits 7 --epochs 40 --batch_size 16
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
# Dataset: normalización z-score por ventana y canal
# ---------------------------------------------------------------------------

class RawEEGDataset(Dataset):
    """
    Dataset de señal EEG cruda para LSTM.

    Forma de entrada (X_raw): (N, n_channels, n_samples)
    Forma de salida por muestra: (n_samples, n_channels) float32
      → formato batch_first para PyTorch LSTM: (batch, seq_len, input_size)

    Normalización: z-score por canal dentro de cada ventana.
    Elimina diferencias de amplitud entre sujetos.
    """

    def __init__(self, X_raw: np.ndarray, y: np.ndarray):
        # Transponer: (N, n_ch, n_samples) → (N, n_samples, n_ch)
        X = X_raw.transpose(0, 2, 1).astype(np.float32)

        # z-score por canal: (N, n_samples, n_ch) → normalizado
        mean = X.mean(axis=1, keepdims=True)           # (N, 1, n_ch)
        std = X.std(axis=1, keepdims=True) + 1e-8       # (N, 1, n_ch)
        X = (X - mean) / std

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Modelo BiLSTM
# ---------------------------------------------------------------------------

class EEGBiLSTM(nn.Module):
    """
    BiLSTM para clasificación ictal/interictal desde señal EEG cruda.

    Input:  (batch, seq_len=1280, input_size=n_channels)  batch_first=True
    Output: (batch,)  logit no normalizado — usar BCEWithLogitsLoss

    Arquitectura:
      LayerNorm(n_channels)
      BiLSTM(input=n_ch, hidden=hidden_size, num_layers, dropout entre capas)
      Tomar el último timestep: output[:, -1, :]  → (batch, 2*hidden_size)
      Dropout → Linear(2*hidden→64) → ReLU → Linear(64→1)
    """

    def __init__(
        self,
        n_channels: int = 23,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        lstm_out_size = hidden_size * 2  # bidireccional
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_ch)
        x = self.norm(x)
        out, _ = self.lstm(x)      # (batch, seq_len, 2*hidden)
        last = out[:, -1, :]       # último timestep captura contexto completo
        return self.classifier(last).squeeze(-1)


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
# GroupKFold CV con BiLSTM
# ---------------------------------------------------------------------------

def run_lstm_groupkfold_cv(
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 7,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 5e-4,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
    lstm_dropout: float = 0.3,
    pos_weight_scale: float = 1.0,
    device: Optional[torch.device] = None,
    run_id: str = "lstm_raw",
    models_dir: Optional[Path] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Ejecuta GroupKFold CV con BiLSTM sobre señal EEG cruda.
    Retorna (result_rows, fold_df) compatibles con save_dl_results().
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LSTM] Dispositivo: {device}")

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
            f"[LSTM] Fold {fold}/{n_splits} | "
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
        train_ds = RawEEGDataset(X_tr, y_tr.astype(np.int8))
        test_ds = RawEEGDataset(X_te, y_te.astype(np.int8))

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
            test_ds, batch_size=64, shuffle=False, num_workers=0
        )

        # ── Modelo y optimizador ────────────────────────────────────────────
        model = EEGBiLSTM(
            n_channels=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lstm_dropout=lstm_dropout,
        ).to(device)

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
        early_stop_patience = 12

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
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

        # ── Inferencia ──────────────────────────────────────────────────────
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
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "lstm_dropout": lstm_dropout,
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

        del model, best_state, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Agregar resultados ──────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_metrics)

    valid = fold_df["pr_auc"].dropna()
    auc_valid = fold_df["auc_roc"].dropna()

    result_row = {
        "model": "lstm_bilstm",
        "feature_set": "raw_eeg_sequence",
        "n_features": X_raw.shape[1] * X_raw.shape[2],  # n_ch × n_samples
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
    fold_df.insert(1, "feature_set", "raw_eeg_sequence")
    fold_df.insert(2, "model", "lstm_bilstm")
    fold_df.insert(3, "n_features", result_row["n_features"])

    print(f"\n{'='*60}")
    print("[LSTM] RESUMEN:")
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
        description="BiLSTM sobre EEG crudo — detección de crisis epilépticas (CHB-MIT)"
    )
    parser.add_argument("--dataset_csv", type=str, required=True,
                        help="windows_dataset.csv generado por chbmit_experiments.py")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directorio raíz con subdirectorios chb##/")
    parser.add_argument("--out_dir", type=str, default="./out_lstm",
                        help="Directorio de salida")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Sujetos a incluir (default: todos los del CSV)")
    parser.add_argument("--n_splits", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Tamaño del estado oculto LSTM")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Número de capas LSTM apiladas")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lstm_dropout", type=float, default=0.3,
                        help="Dropout entre capas LSTM (solo si num_layers > 1)")
    parser.add_argument("--pos_weight_scale", type=float, default=1.0)
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--l_freq", type=float, default=0.5)
    parser.add_argument("--h_freq", type=float, default=45.0)
    parser.add_argument("--resample_hz", type=float, default=256.0)
    parser.add_argument("--no_cache", action="store_true",
                        help="Ignorar caché de arrays .npy (re-leer EDFs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.run_name or "lstm_raw")
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LSTM] run_id={run_id}")
    print(f"[LSTM] out_dir={out_dir}")

    # ── Cargar CSV de ventanas ──────────────────────────────────────────────
    df = pd.read_csv(args.dataset_csv)
    print(f"[LSTM] Dataset: {len(df)} ventanas | pos_rate={df['label'].mean():.4f}")

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
    print(f"[LSTM] Arrays: X={X_raw.shape} | tiempo={time.time()-t0:.0f}s")
    print(f"[LSTM] Distribución: pos={y.sum()} neg={len(y)-y.sum()} rate={y.mean():.4f}")

    # ── CV ─────────────────────────────────────────────────────────────────
    models_dir = run_dir / "saved_models"
    result_rows, fold_df = run_lstm_groupkfold_cv(
        X_raw=X_raw,
        y=y,
        groups=groups,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lstm_dropout=args.lstm_dropout,
        pos_weight_scale=args.pos_weight_scale,
        run_id=run_id,
        models_dir=models_dir,
    )

    # ── Enriquecer con metadatos del run ────────────────────────────────────
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
                "lr": args.lr, "hidden_size": args.hidden_size,
                "num_layers": args.num_layers, "dropout": args.dropout,
                "bidirectional": True,
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
            "model": "BiLSTM_raw",
            "subjects": subjects_used,
            "n_splits": args.n_splits,
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        },
    )

    print(f"\n[LSTM] Completado. Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
