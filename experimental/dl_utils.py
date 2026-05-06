"""
dl_utils.py — Infraestructura compartida para experimentos Deep Learning (CNN + LSTM).

Funciones principales:
  - load_raw_edf_dl: Carga un EDF con MNE y retorna numpy array (n_ch, n_samples).
  - build_raw_arrays: Lee windows_dataset.csv, agrupa por EDF y extrae señal cruda.
    Soporta caché .npy para evitar re-leer EDFs en runs sucesivos.
  - compute_metrics_safe: Calcula AUC-ROC, PR-AUC, sensitivity, specificity.
  - save_dl_results: Escribe results.csv y cv_fold_metrics.csv compatibles con
    el esquema de chbmit_experiments.py.
  - make_run_id: Normaliza nombre de run (igual que chbmit_experiments.py).
"""

from __future__ import annotations

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Utilidades de naming
# ---------------------------------------------------------------------------

def make_run_id(user_run_name: Optional[str] = None) -> str:
    if user_run_name:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", user_run_name.strip())
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Carga EDF
# ---------------------------------------------------------------------------

def load_raw_edf_dl(
    edf_path: Path,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
    resample_hz: float = 256.0,
) -> Tuple[np.ndarray, float]:
    """
    Carga y preprocesa un EDF. Retorna (data, fs) donde data es (n_ch, n_samples) float32.
    Usa el mismo pipeline que chbmit_experiments.load_raw_edf():
    pick EEG → filtro pasa-banda → resample.
    """
    raw = mne.io.read_raw_edf(
        edf_path.as_posix(), preload=True, verbose="ERROR"
    )
    raw.pick("eeg")
    raw.filter(l_freq, h_freq, verbose="ERROR")
    if resample_hz is not None:
        raw.resample(resample_hz, verbose="ERROR")
    data = raw.get_data().astype(np.float32)
    fs = float(raw.info["sfreq"])
    return data, fs


# ---------------------------------------------------------------------------
# Construcción del array crudo (con caché)
# ---------------------------------------------------------------------------

def build_raw_arrays(
    df: pd.DataFrame,
    data_root: Path,
    window_sec: float = 5.0,
    fs: float = 256.0,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
    subjects: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae señal cruda para cada ventana en df, usando edf_file + t0_sec del CSV.
    Carga cada EDF una sola vez (agrupando por edf_file).

    Args:
        df         : DataFrame con columnas subject_id, edf_file, t0_sec, t1_sec, label
        data_root  : directorio raíz con subdirectorios chb##/
        window_sec : duración esperada de cada ventana (segundos)
        fs         : frecuencia de muestreo tras resample
        l_freq     : filtro pasa-banda inferior
        h_freq     : filtro pasa-banda superior
        subjects   : lista de sujetos a incluir (None = todos)
        cache_dir  : directorio donde guardar/leer X_raw.npy, y.npy, groups.npy

    Returns:
        X_raw  : (N, n_channels, n_samples)  float32
        y      : (N,)  int8
        groups : (N,)  str — subject_id para GroupKFold
    """
    if subjects is not None:
        df = df[df["subject_id"].isin(subjects)].reset_index(drop=True)

    subjects_key = sorted(df["subject_id"].unique().tolist())

    if cache_dir is not None:
        cache_X = cache_dir / "X_raw.npy"
        cache_y = cache_dir / "y.npy"
        cache_g = cache_dir / "groups.npy"
        cache_meta = cache_dir / "cache_meta.json"
        cache_valid = False
        if cache_X.exists() and cache_y.exists() and cache_g.exists() and cache_meta.exists():
            with open(cache_meta) as f:
                meta = json.load(f)
            if meta.get("subjects") == subjects_key:
                cache_valid = True
        if cache_valid:
            print(f"[DL] Cargando arrays desde caché: {cache_dir}")
            X_raw = np.load(cache_X)
            y = np.load(cache_y)
            groups = np.load(cache_g, allow_pickle=True)
            print(f"[DL] Cache cargado: X={X_raw.shape} y={y.shape}")
            return X_raw, y, groups
        elif cache_X.exists():
            print(f"[DL] Caché inválido (sujetos distintos) — re-extrayendo")

    n_samples = int(round(window_sec * fs))
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups_list: List[str] = []
    skipped = 0
    n_ch_ref: Optional[int] = None

    t0_global = time.time()
    grouped = df.groupby("edf_file")
    n_edfs = len(grouped)

    for ei, (edf_name, group) in enumerate(grouped, start=1):
        subject_id = group["subject_id"].iloc[0]
        edf_path = data_root / subject_id / edf_name

        if not edf_path.exists():
            print(f"  [SKIP] EDF no encontrado: {edf_path}")
            skipped += len(group)
            continue

        try:
            raw_data, actual_fs = load_raw_edf_dl(edf_path, l_freq, h_freq, fs)
        except Exception as exc:
            print(f"  [WARN] Error leyendo {edf_path.name}: {exc}")
            skipped += len(group)
            continue

        n_ch = raw_data.shape[0]
        if n_ch_ref is None:
            n_ch_ref = n_ch
        elif n_ch != n_ch_ref:
            # Recortar al mínimo de canales para consistencia
            n_ch_use = min(n_ch, n_ch_ref)
            raw_data = raw_data[:n_ch_use]
            n_ch_ref = n_ch_use

        total_edf_samples = raw_data.shape[1]

        for _, row in group.iterrows():
            s = int(round(row["t0_sec"] * fs))
            e = s + n_samples

            if s >= total_edf_samples:
                skipped += 1
                continue

            epoch = raw_data[:n_ch_ref, s:e]

            if epoch.shape[1] < n_samples:
                pad = n_samples - epoch.shape[1]
                epoch = np.pad(epoch, ((0, 0), (0, pad)))

            X_list.append(epoch)
            y_list.append(int(row["label"]))
            groups_list.append(subject_id)

        if ei % 10 == 0 or ei == n_edfs:
            elapsed = time.time() - t0_global
            print(
                f"  [DL build] EDF {ei}/{n_edfs} | "
                f"windows_so_far={len(X_list)} | elapsed={elapsed:.0f}s"
            )

    if not X_list:
        raise RuntimeError(
            "build_raw_arrays: no se extrajeron ventanas. "
            "Verificar paths del CSV y data_root."
        )

    print(f"[DL] Total ventanas: {len(X_list)} | skipped: {skipped}")

    X_raw = np.stack(X_list, axis=0)  # (N, n_ch, n_samples)
    y = np.array(y_list, dtype=np.int8)
    groups = np.array(groups_list, dtype=object)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "X_raw.npy", X_raw)
        np.save(cache_dir / "y.npy", y)
        np.save(cache_dir / "groups.npy", groups)
        with open(cache_dir / "cache_meta.json", "w") as f:
            json.dump({"subjects": subjects_key, "n_windows": len(X_list)}, f)
        print(f"[DL] Arrays guardados en caché: {cache_dir}")

    return X_raw, y, groups


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics_safe(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación. Retorna NaN si no hay clase positiva.
    Idéntico en semántica a chbmit_experiments.py.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return dict(
            auc_roc=float("nan"),
            pr_auc=float("nan"),
            sensitivity=float("nan"),
            specificity=float("nan"),
            accuracy=float("nan"),
            test_pos=int(y_true.sum()),
            test_neg=int(len(y_true) - y_true.sum()),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_roc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / len(y_true)

    return dict(
        auc_roc=float(auc_roc),
        pr_auc=float(pr_auc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        accuracy=float(accuracy),
        test_pos=int(y_true.sum()),
        test_neg=int(len(y_true) - y_true.sum()),
    )


# ---------------------------------------------------------------------------
# Guardar resultados (esquema compatible con chbmit_experiments.py)
# ---------------------------------------------------------------------------

def save_dl_results(
    result_rows: List[Dict],
    fold_frames: List[pd.DataFrame],
    out_dir: Path,
    run_dir: Path,
    run_id: str,
    manifest_extras: Optional[Dict] = None,
) -> None:
    """
    Escribe results.csv y cv_fold_metrics.csv con el mismo esquema que
    chbmit_experiments.py para poder hacer pd.concat() limpio.
    Append-on-exist para results.csv global; run_dir tiene copias propias.
    """
    res_df = pd.DataFrame(result_rows)

    # results.csv — append si ya existe
    global_res_path = out_dir / "results.csv"
    if global_res_path.exists():
        existing = pd.read_csv(global_res_path)
        res_df = pd.concat([existing, res_df], ignore_index=True)
    res_df.to_csv(global_res_path, index=False)
    res_df.tail(len(result_rows)).to_csv(run_dir / "results.csv", index=False)
    print(f"[OK] Resultados → {run_dir / 'results.csv'}")

    if fold_frames:
        folds_df = pd.concat(fold_frames, ignore_index=True)
        folds_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
        folds_df.to_csv(run_dir / "cv_fold_metrics.csv", index=False)
        print(f"[OK] Fold metrics → {run_dir / 'cv_fold_metrics.csv'}")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        **(manifest_extras or {}),
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
