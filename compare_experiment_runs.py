from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["feature_set", "model", "n_features"]
METRIC_COLUMNS = [
    "auc_roc_mean",
    "pr_auc_mean",
    "sensitivity_mean",
    "specificity_mean",
    "accuracy_mean",
]


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de resultados: {path}")
    return pd.read_csv(path)


def compare_runs(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    current_sel = current_df[KEY_COLUMNS + METRIC_COLUMNS].copy()
    baseline_sel = baseline_df[KEY_COLUMNS + METRIC_COLUMNS].copy()

    merged = current_sel.merge(baseline_sel, on=KEY_COLUMNS, suffixes=("_current", "_baseline"))
    for metric in METRIC_COLUMNS:
        merged[f"delta_{metric}"] = merged[f"{metric}_current"] - merged[f"{metric}_baseline"]
    return merged.sort_values(by=["delta_auc_roc_mean", "delta_pr_auc_mean"], ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Compara dos corridas del benchmark CHB-MIT")
    parser.add_argument("--current", type=str, required=True, help="Ruta al results.csv actual")
    parser.add_argument("--baseline", type=str, required=True, help="Ruta al results.csv base")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida CSV para la comparación")
    args = parser.parse_args()

    current_path = Path(args.current)
    baseline_path = Path(args.baseline)
    output_path = Path(args.output) if args.output else current_path.parent / "comparison_vs_baseline.csv"

    current_df = load_results(current_path)
    baseline_df = load_results(baseline_path)
    comparison_df = compare_runs(current_df, baseline_df)
    comparison_df.to_csv(output_path, index=False)

    print(f"[OK] Comparación guardada en: {output_path}")
    if not comparison_df.empty:
        print("\nTOP 5 mejoras por AUC-ROC:")
        print(
            comparison_df[
                KEY_COLUMNS + ["delta_auc_roc_mean", "delta_pr_auc_mean"]
            ].head(5).to_string(index=False)
        )


if __name__ == "__main__":
    main()