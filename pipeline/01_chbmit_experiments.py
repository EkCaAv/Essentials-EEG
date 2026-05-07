# chbmit_experiments.py
# Requisitos:
#   pip install mne numpy pandas scipy scikit-learn
#
# Ejecutar (PowerShell, una sola línea):
#   python .\chbmit_experiments.py --data_root ".\data" --out_dir ".\out_6_10" --subjects chb05 chb09 chb14 chb16 chb20 chb22 chb23 --window_sec 5 --overlap 0.5 --n_splits 7

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import mne
from mne.time_frequency import psd_array_welch


def integrate_trapezoid(y: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    """Compatibilidad NumPy 1.x/2.x para integración por trapecios."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


# ----------------------------
# Configuración por defecto
# ----------------------------
DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class WindowConfig:
    window_sec: float = 5.0
    overlap: float = 0.5
    exclude_margin_sec: float = 0.0


@dataclass(frozen=True)
class PreprocessConfig:
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch: Optional[float] = None
    resample_hz: Optional[float] = 256.0
    picks: str = "eeg"
    verbose: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    bands: Dict[str, Tuple[float, float]] = None
    use_bandpower_rel: bool = True
    use_rms: bool = True
    use_kurtosis: bool = True
    use_skewness: bool = True
    channel_agg: Tuple[str, ...] = ("mean", "std")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: Dict
    # param_grid: espacio de búsqueda para RandomizedSearchCV (nested CV).
    # Puede ser un dict o una lista de dicts (igual que en GridSearchCV).
    # Las claves NO deben llevar el prefijo "clf__" — se añade automáticamente.
    # Si es None, no se hace búsqueda de hiperparámetros.
    param_grid: Optional[object] = field(default=None, hash=False, compare=False)
    n_iter_search: int = 30


def build_estimator(model_cfg: ModelConfig):
    if model_cfg.name == "logreg":
        return LogisticRegression(**model_cfg.params)
    if model_cfg.name == "random_forest":
        return RandomForestClassifier(**model_cfg.params)
    if model_cfg.name == "svm":
        return SVC(**model_cfg.params)
    if model_cfg.name == "gradient_boosting":
        return GradientBoostingClassifier(**model_cfg.params)
    raise ValueError(f"Modelo no soportado: {model_cfg.name}")


def make_run_id(user_run_name: Optional[str] = None) -> str:
    if user_run_name:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", user_run_name.strip())
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def summarize_dataset(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_files": int(df["edf_file"].nunique()),
        "positive_windows": int(df["label"].sum()),
        "negative_windows": int((df["label"] == 0).sum()),
        "positive_rate": float(df["label"].mean()),
    }


def build_manifest(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    dataset_summary: Dict[str, float],
    feature_sets: Dict[str, List[str]],
    models: List[ModelConfig],
) -> Dict:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "out_dir": str(run_dir.parent.parent.resolve()),
            "run_dir": str(run_dir.resolve()),
            "data_root": str(Path(args.data_root).resolve()),
        },
        "dataset": dataset_summary,
        "window_config": {
            "window_sec": args.window_sec,
            "overlap": args.overlap,
            "exclude_margin_sec": args.exclude_margin_sec,
            "max_interictal_per_file": args.max_interictal_per_file,
        },
        "preprocessing": {
            "l_freq": args.l_freq,
            "h_freq": args.h_freq,
            "notch": args.notch,
            "resample_hz": args.resample_hz,
        },
        "cv": {
            "strategy": "GroupKFold",
            "n_splits": args.n_splits,
            "group_column": "subject_id",
            "threshold": 0.5,
            "tune_hyperparams": args.tune_hyperparams,
            "inner_cv_splits": args.inner_cv_splits if args.tune_hyperparams else None,
        },
        "subjects": args.subjects,
        "models": [{"name": m.name, "params": m.params} for m in models],
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "compare_to": args.compare_to,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compare_results_frames(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["feature_set", "model", "n_features"]
    current_sel = current_df[join_cols + ["auc_roc_mean", "pr_auc_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean"]].copy()
    baseline_sel = baseline_df[join_cols + ["auc_roc_mean", "pr_auc_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean"]].copy()

    merged = current_sel.merge(baseline_sel, on=join_cols, suffixes=("_current", "_baseline"))
    if merged.empty:
        return merged

    for metric in ["auc_roc_mean", "pr_auc_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_current"] - merged[f"{metric}_baseline"]

    return merged.sort_values(by=["delta_auc_roc_mean", "delta_pr_auc_mean"], ascending=False)


# ----------------------------
# Parser del summary CHB-MIT
# ----------------------------
def parse_chb_summary(summary_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    text = summary_path.read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    file_to_seizures: Dict[str, List[Tuple[float, float]]] = {}
    current_file: Optional[str] = None

    file_re = re.compile(r"File Name:\s*(.+\.edf)", re.IGNORECASE)
    start_re = re.compile(r"Seizure\s*\d+\s*Start Time:\s*(\d+)\s*seconds", re.IGNORECASE)
    end_re = re.compile(r"Seizure\s*\d+\s*End Time:\s*(\d+)\s*seconds", re.IGNORECASE)

    pending_start: Optional[float] = None

    for ln in lines:
        mfile = file_re.search(ln)
        if mfile:
            current_file = mfile.group(1).strip()
            file_to_seizures.setdefault(current_file, [])
            pending_start = None
            continue

        if current_file is None:
            continue

        ms = start_re.search(ln)
        if ms:
            pending_start = float(ms.group(1))
            continue

        me = end_re.search(ln)
        if me and pending_start is not None:
            end_t = float(me.group(1))
            if end_t > pending_start:
                file_to_seizures[current_file].append((pending_start, end_t))
            pending_start = None

    return file_to_seizures


# ----------------------------
# Utilidades: ventanas y labels
# ----------------------------
def make_windows(n_samples: int, fs: float, win_sec: float, overlap: float) -> List[Tuple[int, int, float, float]]:
    win = int(round(win_sec * fs))
    step = int(round(win * (1.0 - overlap)))
    step = max(step, 1)

    windows = []
    for start in range(0, n_samples - win + 1, step):
        end = start + win
        t0 = start / fs
        t1 = end / fs
        windows.append((start, end, t0, t1))
    return windows


def interval_overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) < min(a1, b1)


def label_window(t0: float, t1: float, seizure_intervals: List[Tuple[float, float]], margin: float = 0.0) -> Optional[int]:
    if margin > 0:
        for s0, s1 in seizure_intervals:
            if interval_overlaps(t0, t1, s0 - margin, s0) or interval_overlaps(t0, t1, s1, s1 + margin):
                return None

    for s0, s1 in seizure_intervals:
        if interval_overlaps(t0, t1, s0, s1):
            return 1
    return 0


# ----------------------------
# Features
# ----------------------------
def compute_bandpower_relative(epoch: np.ndarray, fs: float, bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    # MNE exige n_fft int; además no debe exceder la ventana
    n_fft = min(2048, epoch.shape[1])

    psd, freqs = psd_array_welch(
        epoch,
        sfreq=fs,
        fmin=0.5,
        fmax=45.0,
        n_fft=n_fft,
        verbose=False
    )

    total_power = integrate_trapezoid(psd, freqs, axis=1) + 1e-12

    out: Dict[str, np.ndarray] = {}
    for name, (f0, f1) in bands.items():
        mask = (freqs >= f0) & (freqs < f1)
        if not np.any(mask):
            out[name] = np.zeros(epoch.shape[0], dtype=float)
            continue
        band_power = integrate_trapezoid(psd[:, mask], freqs[mask], axis=1)
        out[name] = band_power / total_power
    return out


def agg_channels(values: np.ndarray, aggs: Tuple[str, ...]) -> Dict[str, float]:
    # IMPORTANTÍSIMO: limpia NaN/inf a 0 para no contaminar el pipeline
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    res: Dict[str, float] = {}
    if "mean" in aggs:
        res["mean"] = float(np.mean(values))
    if "std" in aggs:
        res["std"] = float(np.std(values))
    if "min" in aggs:
        res["min"] = float(np.min(values))
    if "max" in aggs:
        res["max"] = float(np.max(values))
    return res


def extract_features(epoch: np.ndarray, fs: float, feat_cfg: FeatureConfig) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    if feat_cfg.use_bandpower_rel:
        bp = compute_bandpower_relative(epoch, fs, feat_cfg.bands)
        for band_name, per_ch in bp.items():
            ag = agg_channels(per_ch, feat_cfg.channel_agg)
            for agg_name, v in ag.items():
                feats[f"bp_rel_{band_name}_{agg_name}"] = v

    if feat_cfg.use_rms:
        per_ch = np.sqrt(np.mean(epoch ** 2, axis=1))
        ag = agg_channels(per_ch, feat_cfg.channel_agg)
        for agg_name, v in ag.items():
            feats[f"rms_{agg_name}"] = v

    if feat_cfg.use_kurtosis:
        per_ch = kurtosis(epoch, axis=1, fisher=True, bias=False)
        # kurtosis puede dar NaN en señales constantes
        per_ch = np.nan_to_num(per_ch, nan=0.0, posinf=0.0, neginf=0.0)
        ag = agg_channels(per_ch, feat_cfg.channel_agg)
        for agg_name, v in ag.items():
            feats[f"kurtosis_{agg_name}"] = v

    if feat_cfg.use_skewness:
        per_ch = skew(epoch, axis=1, bias=False)
        # skew puede dar NaN en señales constantes
        per_ch = np.nan_to_num(per_ch, nan=0.0, posinf=0.0, neginf=0.0)
        ag = agg_channels(per_ch, feat_cfg.channel_agg)
        for agg_name, v in ag.items():
            feats[f"skewness_{agg_name}"] = v

    return feats


# ----------------------------
# Carga EDF + preproc
# ----------------------------
def load_raw_edf(edf_path: Path, pp: PreprocessConfig) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path.as_posix(), preload=True, verbose="ERROR" if not pp.verbose else True)
    raw.pick(pp.picks)

    if pp.notch is not None:
        raw.notch_filter(pp.notch, verbose="ERROR" if not pp.verbose else True)

    raw.filter(pp.l_freq, pp.h_freq, verbose="ERROR" if not pp.verbose else True)

    if pp.resample_hz is not None:
        raw.resample(pp.resample_hz, verbose="ERROR" if not pp.verbose else True)

    return raw


# ----------------------------
# Dataset builder (ventanas) + progreso
# ----------------------------
def build_windows_dataset(
    data_root: Path,
    subjects: List[str],
    win_cfg: WindowConfig,
    pp_cfg: PreprocessConfig,
    feat_cfg: FeatureConfig,
    max_interictal_per_file: Optional[int] = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows: List[Dict] = []

    total_files = 0
    total_windows_used = 0
    total_ictal = 0
    total_inter = 0

    t_global = time.time()

    for si, subj in enumerate(subjects, start=1):
        subj_t0 = time.time()
        subj_dir = data_root / subj
        summary_path = subj_dir / f"{subj}-summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"No existe summary: {summary_path}")

        seizures_by_file = parse_chb_summary(summary_path)
        file_items = list(seizures_by_file.items())

        print(f"\n[SUBJECT {si}/{len(subjects)}] {subj} | files_in_summary={len(file_items)}")

        for fi, (edf_name, seizure_intervals) in enumerate(file_items, start=1):
            edf_path = subj_dir / edf_name
            if not edf_path.exists():
                print(f"  [SKIP {fi}/{len(file_items)}] {edf_name} (no existe en disco)")
                continue

            t_file0 = time.time()

            raw = load_raw_edf(edf_path, pp_cfg)
            data = raw.get_data()
            fs = float(raw.info["sfreq"])
            n_samples = data.shape[1]

            windows = make_windows(n_samples, fs, win_cfg.window_sec, win_cfg.overlap)

            labeled = []
            for s, e, t0, t1 in windows:
                lab = label_window(t0, t1, seizure_intervals, margin=win_cfg.exclude_margin_sec)
                if lab is None:
                    continue
                labeled.append((s, e, t0, t1, lab))

            if not labeled:
                print(f"  [SKIP {fi}/{len(file_items)}] {edf_name} (0 ventanas etiquetadas)")
                continue

            ictal = [x for x in labeled if x[4] == 1]
            inter = [x for x in labeled if x[4] == 0]

            inter_before = len(inter)
            if max_interictal_per_file is not None and len(inter) > max_interictal_per_file:
                idx = rng.choice(len(inter), size=max_interictal_per_file, replace=False)
                inter = [inter[i] for i in idx]

            used = ictal + inter

            print(
                f"  [FILE {fi}/{len(file_items)}] {edf_name} | "
                f"win_total={len(windows)} labeled={len(labeled)} "
                f"ictal={len(ictal)} inter={len(inter)} (from {inter_before}) | extracting..."
            )

            for s, e, t0, t1, lab in used:
                epoch = data[:, s:e]
                feats = extract_features(epoch, fs, feat_cfg)
                row = {
                    "subject_id": subj,
                    "edf_file": edf_name,
                    "t0_sec": t0,
                    "t1_sec": t1,
                    "label": int(lab),
                }
                row.update(feats)
                rows.append(row)

            total_files += 1
            total_windows_used += len(used)
            total_ictal += len(ictal)
            total_inter += len(inter)

            print(f"    [OK] extracted={len(used)} windows | file_time={(time.time() - t_file0):.1f}s")

        print(f"[SUBJECT DONE] {subj} | subject_time={(time.time() - subj_t0) / 60:.1f} min")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No se generaron muestras. Revisa rutas/EDF/summary.")

    # Limpieza final extra por si quedó algo raro
    feat_cols = [c for c in df.columns if c not in ["subject_id", "edf_file", "t0_sec", "t1_sec", "label"]]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    nan_count = int(df[feat_cols].isna().sum().sum())
    if nan_count > 0:
        print(f"[WARN] Dataset tiene {nan_count} NaNs en features; se imputarán en el pipeline.")

    print("\n[DATASET SUMMARY]")
    print(f"  files_used={total_files}")
    print(f"  windows_used={total_windows_used}")
    print(f"  ictal_windows (pre-sample sum)={total_ictal}")
    print(f"  interictal_windows (post-sample sum)={total_inter}")
    print(f"  build_time={(time.time() - t_global) / 60:.1f} min")
    print(f"  df_rows={len(df)} | pos_rate={df['label'].mean():.4f}")

    return df


# ----------------------------
# Evaluación (GroupKFold por sujeto) - robusta
# ----------------------------
def compute_metrics_safe(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Maneja el caso donde en y_true hay una sola clase (no hay positivos o no hay negativos).
    En ese caso AUC-ROC y PR-AUC no están definidos -> devolvemos NaN.
    """
    y_true = y_true.astype(int)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)

    if pos == 0 or neg == 0:
        # Métricas tipo ranking no definidas
        auc = np.nan
        pr_auc = np.nan

        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp + 1e-12) if (tn + fp) > 0 else np.nan
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

        return {
            "auc_roc": float(auc),
            "pr_auc": float(pr_auc),
            "sensitivity": float(sens) if not np.isnan(sens) else np.nan,
            "specificity": float(spec) if not np.isnan(spec) else np.nan,
            "accuracy": float(acc),
            "test_pos": pos,
            "test_neg": neg,
        }

    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "auc_roc": float(auc),
        "pr_auc": float(pr_auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
        "test_pos": pos,
        "test_neg": neg,
    }


def run_groupkfold_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_cfg: ModelConfig,
    n_splits: int = 7,
    tune: bool = False,
    inner_cv_splits: int = 3,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    groups = df["subject_id"].values

    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        train_unique = np.unique(y[tr])
        if train_unique.size < 2:
            m = {
                "auc_roc": np.nan,
                "pr_auc": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "accuracy": np.nan,
                "test_pos": int(y[te].sum()),
                "test_neg": int(len(te) - y[te].sum()),
                "fold": fold,
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "train_pos": int(y[tr].sum()),
                "train_neg": int(len(tr) - y[tr].sum()),
                "train_subjects": ";".join(sorted(np.unique(groups[tr]).tolist())),
                "test_subjects": ";".join(sorted(np.unique(groups[te]).tolist())),
                "train_single_class": int(train_unique[0]),
                "best_params": "",
                "best_inner_score": np.nan,
            }
            fold_metrics.append(m)
            print(
                f"  [CV] fold={fold}/{n_splits} skipped (train one class={int(train_unique[0])}) "
                f"| test_pos={m['test_pos']} test_neg={m['test_neg']}"
            )
            continue

        estimator = build_estimator(model_cfg)
        pipe = Pipeline([
            # ✅ Esto evita el crash por NaN:
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])

        if tune and model_cfg.param_grid is not None:
            # CV anidado: inner GroupKFold busca mejores hiperparámetros,
            # outer fold (este loop) estima el rendimiento real (sin sesgo).
            # Las claves del param_grid se prefiján con "clf__" para el Pipeline.
            if isinstance(model_cfg.param_grid, list):
                prefixed_grid = [{f"clf__{k}": v for k, v in g.items()} for g in model_cfg.param_grid]
            else:
                prefixed_grid = {f"clf__{k}": v for k, v in model_cfg.param_grid.items()}

            inner_cv = GroupKFold(n_splits=inner_cv_splits)
            search = RandomizedSearchCV(
                pipe,
                param_distributions=prefixed_grid,
                n_iter=model_cfg.n_iter_search,
                scoring="roc_auc",
                cv=inner_cv,
                refit=True,
                n_jobs=-1,
                random_state=42,
                error_score=np.nan,
            )
            search.fit(X[tr], y[tr], groups=groups[tr])
            fitted_estimator = search.best_estimator_
            best_params_str = json.dumps(
                {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                ensure_ascii=False,
                default=str,
            )
            best_inner_score = float(search.best_score_) if not np.isnan(search.best_score_) else np.nan
            print(f"    [TUNE] fold={fold} best_inner_auc={best_inner_score:.4f} | {best_params_str}")
        else:
            pipe.fit(X[tr], y[tr])
            fitted_estimator = pipe
            best_params_str = ""
            best_inner_score = np.nan

        y_score = fitted_estimator.predict_proba(X[te])[:, 1]

        m = compute_metrics_safe(y[te], y_score, threshold=0.5)
        m["fold"] = fold
        m["train_rows"] = int(len(tr))
        m["test_rows"] = int(len(te))
        m["train_pos"] = int(y[tr].sum())
        m["train_neg"] = int(len(tr) - y[tr].sum())
        m["train_subjects"] = ";".join(sorted(np.unique(groups[tr]).tolist()))
        m["test_subjects"] = ";".join(sorted(np.unique(groups[te]).tolist()))
        m["train_single_class"] = ""
        m["best_params"] = best_params_str
        m["best_inner_score"] = best_inner_score
        fold_metrics.append(m)

        auc_str = "nan" if (m["auc_roc"] is np.nan or np.isnan(m["auc_roc"])) else f"{m['auc_roc']:.3f}"
        pr_str = "nan" if (m["pr_auc"] is np.nan or np.isnan(m["pr_auc"])) else f"{m['pr_auc']:.3f}"

        print(
            f"  [CV] fold={fold}/{n_splits} pos={m['test_pos']} neg={m['test_neg']} "
            f"| auc={auc_str} pr_auc={pr_str}"
        )

    fm = pd.DataFrame(fold_metrics)

    # Promedios ignorando NaN (cuando un fold no tiene positivos/negativos)
    out = {}
    for col in ["auc_roc", "pr_auc", "sensitivity", "specificity", "accuracy"]:
        vals = fm[col].astype(float).to_numpy()
        out[f"{col}_mean"] = float(np.nanmean(vals))
        out[f"{col}_std"] = float(np.nanstd(vals, ddof=1))

    # Extras de diagnóstico
    out["folds_with_nan_auc"] = int(np.isnan(fm["auc_roc"]).sum())
    out["folds_with_nan_prauc"] = int(np.isnan(fm["pr_auc"]).sum())
    out["avg_test_pos"] = float(fm["test_pos"].mean())
    return out, fm


# ----------------------------
# Experimentos: combinaciones de features
# ----------------------------
def select_feature_sets(all_cols: List[str]) -> Dict[str, List[str]]:
    def cols_like(prefix: str) -> List[str]:
        return [c for c in all_cols if c.startswith(prefix)]

    bp = [c for c in all_cols if c.startswith("bp_rel_")]
    rms = cols_like("rms_")
    kurt = cols_like("kurtosis_")
    skw = cols_like("skewness_")

    return {
        "bp_only": bp,
        "bp_plus_rms": bp + rms,
        "bp_plus_rms_kurt": bp + rms + kurt,
        "bp_plus_rms_kurt_skew": bp + rms + kurt + skw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--compare_to", type=str, default=None)

    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--exclude_margin_sec", type=float, default=0.0)

    parser.add_argument("--l_freq", type=float, default=0.5)
    parser.add_argument("--h_freq", type=float, default=45.0)
    parser.add_argument("--notch", type=float, default=None)
    parser.add_argument("--resample_hz", type=float, default=256.0)

    parser.add_argument("--max_interictal_per_file", type=int, default=300)
    parser.add_argument("--n_splits", type=int, default=7)
    parser.add_argument(
        "--tune_hyperparams",
        action="store_true",
        help="Activa CV anidado con RandomizedSearchCV para encontrar los mejores hiperparámetros.",
    )
    parser.add_argument(
        "--inner_cv_splits",
        type=int,
        default=3,
        help="Número de folds en el CV interno (búsqueda de hiperparámetros). Por defecto 3.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.run_name)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    win_cfg = WindowConfig(args.window_sec, args.overlap, args.exclude_margin_sec)
    pp_cfg = PreprocessConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch=args.notch,
        resample_hz=args.resample_hz,
        verbose=False,
    )
    feat_cfg = FeatureConfig(
        bands=DEFAULT_BANDS,
        use_bandpower_rel=True,
        use_rms=True,
        use_kurtosis=True,
        use_skewness=True,
        channel_agg=("mean", "std"),
    )

    print("[STEP 1/3] Building windows dataset (feature extraction)...")
    t0 = time.time()
    df = build_windows_dataset(
        data_root=Path(args.data_root),
        subjects=args.subjects,
        win_cfg=win_cfg,
        pp_cfg=pp_cfg,
        feat_cfg=feat_cfg,
        max_interictal_per_file=args.max_interictal_per_file,
        random_state=42,
    )
    print(f"[TIME] build_windows_dataset={(time.time() - t0) / 60:.1f} min")

    df_path = out_dir / "windows_dataset.csv"
    run_df_path = run_dir / "windows_dataset.csv"
    df.to_csv(df_path, index=False)
    df.to_csv(run_df_path, index=False)
    print(f"[OK] Dataset guardado en: {df_path}  (rows={len(df)})")

    print("\n[STEP 2/3] Defining models + feature sets...")

    # NOTA: ConvergenceWarning es común con clases extremadamente desbalanceadas.
    # Subimos max_iter y relajamos tolerancia un poco para evitar warnings (sin “maquillar” resultados).
    models = [
        ModelConfig(
            name="logreg",
            params=dict(
                solver="saga",
                penalty="l2",
                C=1.0,
                max_iter=20000,   # ↑ para reducir ConvergenceWarning
                tol=1e-3,         # ↑ un poco más tolerante
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            ),
            # Espacio de búsqueda para nested CV.
            # Lista de dicts: evita combinar l1_ratio con penalty l1/l2.
            param_grid=[
                {
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "penalty": ["l1", "l2"],
                },
                {
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "penalty": ["elasticnet"],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
            ],
            n_iter_search=30,
        ),
        ModelConfig(
            name="random_forest",
            params=dict(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
            # Espacio de búsqueda para nested CV.
            param_grid={
                "n_estimators": [100, 200, 400, 600],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            n_iter_search=30,
        ),
        ModelConfig(
            name="svm",
            params=dict(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
                cache_size=2000,
            ),
            param_grid={
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto"],
            },
            n_iter_search=20,
        ),
        ModelConfig(
            name="gradient_boosting",
            params=dict(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            ),
            param_grid={
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 6],
            },
            n_iter_search=20,
        ),
    ]

    feature_cols = [c for c in df.columns if c not in ["subject_id", "edf_file", "t0_sec", "t1_sec", "label"]]
    feature_sets = select_feature_sets(feature_cols)

    print("\n[STEP 3/3] Running CV experiments...")
    t1 = time.time()

    results = []
    fold_frames = []
    for fs_name, fs_cols in feature_sets.items():
        if not fs_cols:
            print(f"[WARN] Feature set {fs_name} vacío. Saltando.")
            continue

        print(f"\n[EXPERIMENT] feature_set={fs_name} | n_features={len(fs_cols)}")
        for mcfg in models:
            metrics, fold_df = run_groupkfold_cv(
                df, fs_cols, mcfg,
                n_splits=args.n_splits,
                tune=args.tune_hyperparams,
                inner_cv_splits=args.inner_cv_splits,
            )
            row = {
                "run_id": run_id,
                "cohort": "6_10",
                "subjects": ",".join(args.subjects),
                "window_sec": win_cfg.window_sec,
                "overlap": win_cfg.overlap,
                "exclude_margin_sec": win_cfg.exclude_margin_sec,
                "l_freq": pp_cfg.l_freq,
                "h_freq": pp_cfg.h_freq,
                "notch": pp_cfg.notch,
                "resample_hz": pp_cfg.resample_hz,
                "max_interictal_per_file": args.max_interictal_per_file,
                "cv": f"GroupKFold(n_splits={args.n_splits})",
                "model": mcfg.name,
                "model_params": json.dumps(mcfg.params, ensure_ascii=False),
                "feature_set": fs_name,
                "n_features": len(fs_cols),
                "n_rows": len(df),
                "pos_rate": float(df["label"].mean()),
            }
            row.update(metrics)
            results.append(row)

            fold_df = fold_df.copy()
            fold_df.insert(0, "run_id", run_id)
            fold_df.insert(1, "feature_set", fs_name)
            fold_df.insert(2, "model", mcfg.name)
            fold_df.insert(3, "n_features", len(fs_cols))
            fold_frames.append(fold_df)

            print(
                f"[DONE] {fs_name} + {mcfg.name} => "
                f"AUC={row['auc_roc_mean']:.3f}±{row['auc_roc_std']:.3f} | "
                f"PR-AUC={row['pr_auc_mean']:.3f}±{row['pr_auc_std']:.3f} | "
                f"nan_auc_folds={row['folds_with_nan_auc']}"
            )

    print(f"\n[TIME] experiments={(time.time() - t1) / 60:.1f} min")

    res_df = pd.DataFrame(results).sort_values(by="auc_roc_mean", ascending=False)
    res_path = out_dir / "results.csv"
    run_res_path = run_dir / "results.csv"
    res_df.to_csv(res_path, index=False)
    res_df.to_csv(run_res_path, index=False)
    print(f"[OK] Resultados guardados en: {res_path}")

    folds_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    if not folds_df.empty:
        fold_path = out_dir / "cv_fold_metrics.csv"
        run_fold_path = run_dir / "cv_fold_metrics.csv"
        folds_df.to_csv(fold_path, index=False)
        folds_df.to_csv(run_fold_path, index=False)
        print(f"[OK] Métricas por fold guardadas en: {fold_path}")

    manifest = build_manifest(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        dataset_summary=summarize_dataset(df),
        feature_sets=feature_sets,
        models=models,
    )
    manifest["artifacts"] = {
        "windows_dataset": str(run_df_path.resolve()),
        "results": str(run_res_path.resolve()),
        "cv_fold_metrics": str((run_dir / "cv_fold_metrics.csv").resolve()) if not folds_df.empty else None,
    }
    save_json(out_dir / "run_manifest.json", manifest)
    save_json(run_dir / "run_manifest.json", manifest)

    if args.compare_to:
        baseline_path = Path(args.compare_to)
        if not baseline_path.exists():
            raise FileNotFoundError(f"No existe compare_to: {baseline_path}")
        baseline_df = pd.read_csv(baseline_path)
        comparison_df = compare_results_frames(res_df, baseline_df)
        comparison_path = run_dir / "comparison_vs_baseline.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"[OK] Comparación contra baseline guardada en: {comparison_path}")

    print("\nTOP 5 por AUC-ROC:")
    print(
        res_df[
            ["feature_set", "model", "n_features", "auc_roc_mean", "auc_roc_std", "pr_auc_mean", "pr_auc_std", "folds_with_nan_auc"]
        ].head(5).to_string(index=False)
    )
    print(f"\n[RUN] run_id={run_id} | run_dir={run_dir}")


if __name__ == "__main__":
    main()
