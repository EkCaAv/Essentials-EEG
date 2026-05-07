"""
consolidate_results.py — Tabla comparativa entre algoritmos de ML clásico.

Línea de investigación: Machine Learning. Lee los resultados producidos por
chbmit_experiments.py (LogReg, RF, SVM, GradientBoosting) y produce:
  - comparison_table.csv: tabla resumen para incluir en la tesis.
  - Wilcoxon signed-rank test pareado por fold entre algoritmos.

Uso:
    python3 consolidate_results.py \\
      --classical_dir ./out_thesis_final/runs/classical_all_models \\
      --out           ./out_thesis_final/comparison_table.csv \\
      --feature_set   bp_plus_rms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


INPUT_MAP = {
    "logreg": "Features tabulares (BP+RMS)",
    "random_forest": "Features tabulares (BP+RMS)",
    "svm": "Features tabulares (BP+RMS)",
    "gradient_boosting": "Features tabulares (BP+RMS)",
}


def load_classical(classical_dir: Path, feature_set: str) -> pd.DataFrame:
    """Carga results.csv clásico y filtra al feature_set de referencia."""
    p = classical_dir / "results.csv"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    df = pd.read_csv(p)
    df_fs = df[df["feature_set"] == feature_set].copy()
    if df_fs.empty:
        available = df["feature_set"].unique().tolist()
        raise ValueError(
            f"Feature set '{feature_set}' no encontrado. "
            f"Disponibles: {available}"
        )
    return df_fs


def load_fold_metrics(results_dir: Path, model_name: str) -> pd.Series:
    """Extrae PR-AUC por fold para un modelo dado."""
    for candidate in [
        results_dir / "cv_fold_metrics.csv",
        results_dir / "runs",
    ]:
        if candidate.is_file():
            df = pd.read_csv(candidate)
            subset = df[df["model"] == model_name]["pr_auc"].dropna()
            return subset.reset_index(drop=True)
        elif candidate.is_dir():
            # buscar el último run en la carpeta runs/
            runs = sorted(candidate.iterdir())
            for run in reversed(runs):
                p = run / "cv_fold_metrics.csv"
                if p.exists():
                    df = pd.read_csv(p)
                    subset = df[df["model"] == model_name]["pr_auc"].dropna()
                    if len(subset):
                        return subset.reset_index(drop=True)
    return pd.Series(dtype=float)


def build_comparison_row(df_row: pd.Series) -> dict:
    model = df_row["model"]
    pr_mean = df_row.get("pr_auc_mean", float("nan"))
    pr_std = df_row.get("pr_auc_std", float("nan"))
    auc_mean = df_row.get("auc_roc_mean", float("nan"))
    auc_std = df_row.get("auc_roc_std", float("nan"))
    return {
        "model": model,
        "input_type": INPUT_MAP.get(model, "—"),
        "pr_auc_mean": pr_mean,
        "pr_auc_std": pr_std,
        "pr_auc_str": f"{pr_mean:.4f} ± {pr_std:.4f}",
        "auc_roc_mean": auc_mean,
        "auc_roc_std": auc_std,
        "auc_roc_str": f"{auc_mean:.4f} ± {auc_std:.4f}",
        "sensitivity_mean": df_row.get("sensitivity_mean", float("nan")),
        "specificity_mean": df_row.get("specificity_mean", float("nan")),
        "folds_with_nan": df_row.get("folds_with_nan_auc", float("nan")),
    }


def wilcoxon_test(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> None:
    """Wilcoxon signed-rank test entre dos vectores de métricas por fold."""
    common_len = min(len(a), len(b))
    if common_len < 3:
        print(
            f"  {label_a} vs {label_b}: insuficientes folds ({common_len}) para Wilcoxon"
        )
        return
    a_vals = a.values[:common_len]
    b_vals = b.values[:common_len]
    if np.allclose(a_vals, b_vals):
        print(f"  {label_a} vs {label_b}: métricas idénticas — test no aplicable")
        return
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        stat, p = stats.wilcoxon(a_vals, b_vals)
    sig = "*" if p < 0.05 else "(n.s.)"
    print(f"  {label_a} vs {label_b} | W={stat:.1f} p={p:.4f} {sig}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidar resultados entre algoritmos de ML clásico para tesis"
    )
    parser.add_argument("--classical_dir", type=str, required=True,
                        help="Directorio del run clásico con results.csv")
    parser.add_argument("--out", type=str,
                        default="./out_thesis_final/comparison_table.csv",
                        help="Ruta del CSV de salida")
    parser.add_argument("--feature_set", type=str, default="bp_plus_rms",
                        help="Feature set de referencia para los modelos clásicos")
    args = parser.parse_args()

    rows = []

    # ── Clásicos ───────────────────────────────────────────────────────────
    print(f"[Consolidar] Cargando clásicos desde: {args.classical_dir}")
    df_cl = load_classical(Path(args.classical_dir), args.feature_set)
    for _, r in df_cl.iterrows():
        rows.append(build_comparison_row(r))
    print(f"  {len(df_cl)} filas cargadas ({', '.join(df_cl['model'].unique())})")

    # ── Tabla final ────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_final = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    df_final.to_csv(out_path, index=False)
    print(f"\n[OK] Tabla guardada: {out_path}")

    # ── Imprimir tabla resumen ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA — algoritmos de ML clásico (ordenada por PR-AUC ↓)")
    print("=" * 80)
    cols = ["model", "pr_auc_str", "auc_roc_str", "sensitivity_mean", "specificity_mean"]
    print(df_final[cols].to_string(index=False))

    # ── Tests estadísticos (Wilcoxon) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST sobre PR-AUC por fold (7 folds)")
    print("(H0: distribuciones iguales; * → p < 0.05)")
    print("Nota: con 7 folds la potencia es baja — interpretar con cautela")
    print("-" * 60)

    classical_dir = Path(args.classical_dir)
    fold_metrics = {
        model: load_fold_metrics(classical_dir, model)
        for model in df_cl["model"].unique()
    }

    classic_models = list(fold_metrics.keys())
    for i, model_a in enumerate(classic_models):
        for model_b in classic_models[i + 1:]:
            a, b = fold_metrics[model_a], fold_metrics[model_b]
            if len(a) and len(b):
                wilcoxon_test(a, b, model_a, model_b)

    print("\n[Consolidar] Listo.")


if __name__ == "__main__":
    main()
