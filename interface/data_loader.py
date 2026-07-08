# interface/data_loader.py
"""
Carga de artefactos ya generados por el pipeline (results.csv, comparison_table,
run_manifest, figuras). La interfaz los muestra e interpreta; no los recalcula.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Raíz del repo (este archivo vive en interface/).
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out_thesis_final"
CLASSICAL_DIR = OUT_DIR / "classical_all_models"


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def load_results() -> Optional[pd.DataFrame]:
    """Tabla de 16 combinaciones (modelo × feature_set) con media ± std."""
    path = CLASSICAL_DIR / "results.csv"
    if not _exists(path):
        return None
    return pd.read_csv(path)


def load_fold_metrics() -> Optional[pd.DataFrame]:
    """Métricas por fold (112 filas) — insumo de las pruebas de Wilcoxon."""
    path = CLASSICAL_DIR / "cv_fold_metrics.csv"
    if not _exists(path):
        return None
    return pd.read_csv(path)


def load_comparison_table() -> Optional[pd.DataFrame]:
    """Tabla comparativa filtrada por feature set de referencia."""
    path = OUT_DIR / "comparison_table.csv"
    if not _exists(path):
        return None
    return pd.read_csv(path)


def load_manifest() -> Optional[dict]:
    """Manifiesto de reproducibilidad del run principal."""
    path = CLASSICAL_DIR / "run_manifest.json"
    if not _exists(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_report_figures() -> List[Path]:
    """Figuras PDF generadas para el reporte."""
    figdir = OUT_DIR / "report_images"
    if not _exists(figdir):
        return []
    return sorted(figdir.glob("*.pdf"))


def results_available() -> bool:
    return load_results() is not None


def best_combination(df: pd.DataFrame, metric: str = "pr_auc_mean") -> Optional[pd.Series]:
    """Fila con el mejor valor de la métrica dada."""
    if df is None or df.empty or metric not in df.columns:
        return None
    return df.loc[df[metric].idxmax()]


# Etiquetas legibles para modelos y feature sets.
MODEL_LABELS = {
    "logreg": "Regresión Logística",
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "gradient_boosting": "Gradient Boosting",
}

FEATURESET_LABELS = {
    "bp_only": "Solo potencia por banda",
    "bp_plus_rms": "Potencia + RMS",
    "bp_plus_rms_kurt": "Potencia + RMS + curtosis",
    "bp_plus_rms_kurt_skew": "Potencia + RMS + curtosis + asimetría",
}
