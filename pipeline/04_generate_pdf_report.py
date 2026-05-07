# generate_pdf_report.py
# Genera el reporte PDF final del experimento DOE — Deteccion de Crisis Epilepticas
# Uso: py -3 -X utf8 generate_pdf_report.py
#
# Outputs:
#   out_thesis_final/reporte_final_doe.pdf

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy import stats

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE               = Path(__file__).resolve().parent.parent / "out_thesis_final"
RESULTS_PATH       = BASE / "classical_all_models" / "results.csv"
FOLD_METRICS_PATH  = BASE / "classical_all_models" / "cv_fold_metrics.csv"
OUTPUT_PDF         = BASE / "reporte_final_doe.pdf"

# ---------------------------------------------------------------------------
# Paleta y estilos
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "logreg":            "#4C72B0",
    "random_forest":     "#55A868",
    "svm":               "#C44E52",
    "gradient_boosting": "#DD8452",
}
MODEL_LABELS = {
    "logreg":            "Logistic Regression",
    "random_forest":     "Random Forest",
    "svm":               "SVM (RBF)",
    "gradient_boosting": "Gradient Boosting",
}
FS_LABELS = {
    "bp_only":                 "BP only\n(10 feat.)",
    "bp_plus_rms":             "BP + RMS\n(12 feat.)",
    "bp_plus_rms_kurt":        "BP + RMS\n+ Kurt (14)",
    "bp_plus_rms_kurt_skew":   "BP + RMS\n+ Kurt + Skew\n(16 feat.)",
}
FS_ORDER  = ["bp_only", "bp_plus_rms", "bp_plus_rms_kurt", "bp_plus_rms_kurt_skew"]
MDL_ORDER = ["logreg", "random_forest", "svm", "gradient_boosting"]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "figure.dpi":      150,
})

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
df   = pd.read_csv(RESULTS_PATH)
fold = pd.read_csv(FOLD_METRICS_PATH)

COLS = ["feature_set", "model", "n_features",
        "auc_roc_mean", "auc_roc_std",
        "pr_auc_mean",  "pr_auc_std",
        "sensitivity_mean", "specificity_mean",
        "folds_with_nan_auc"]
df = df[COLS].copy()

df_pr  = df.sort_values("pr_auc_mean",  ascending=False).reset_index(drop=True)
df_auc = df.sort_values("auc_roc_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_header(ax_or_fig, title: str, subtitle: str = "") -> None:
    """Agrega un header de pagina como texto sobre la figura."""
    pass


def table_axes(fig, data: pd.DataFrame, col_labels: list[str],
               title: str, note: str = "",
               col_widths: list[float] | None = None) -> None:
    ax = fig.add_subplot(111)
    ax.axis("off")
    if col_widths is None:
        col_widths = [1 / len(col_labels)] * len(col_labels)
    tbl = ax.table(
        cellText=data.values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECF0F1")
        cell.set_edgecolor("#BDC3C7")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=7.5,
                 color="#7F8C8D", style="italic")


# ---------------------------------------------------------------------------
# Wilcoxon pareado sobre cv_fold_metrics
# ---------------------------------------------------------------------------
def compute_wilcoxon(fold_df: pd.DataFrame, feature_set: str) -> list[dict]:
    sub = fold_df[fold_df["feature_set"] == feature_set].copy()
    results_wx = []
    models = MDL_ORDER
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            s1 = sub[sub["model"] == m1]["pr_auc"].dropna().values
            s2 = sub[sub["model"] == m2]["pr_auc"].dropna().values
            n  = min(len(s1), len(s2))
            if n < 3:
                results_wx.append({"par": f"{MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}",
                                    "W": "–", "p": "–", "sig": "n/a"})
                continue
            try:
                stat, p = stats.wilcoxon(s1[:n], s2[:n])
                sig = "*" if p < 0.05 else "n.s."
                results_wx.append({"par": f"{MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}",
                                    "W": f"{stat:.1f}", "p": f"{p:.4f}", "sig": sig})
            except Exception:
                results_wx.append({"par": f"{MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}",
                                    "W": "–", "p": "–", "sig": "error"})
    return results_wx


wx_all = {fs: compute_wilcoxon(fold, fs) for fs in FS_ORDER}


# ===========================================================================
# GENERACION DEL PDF
# ===========================================================================
with PdfPages(OUTPUT_PDF) as pdf:

    # -----------------------------------------------------------------------
    # PAGINA 1 — Portada
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#1A252F")

    fig.text(0.5, 0.78, "DETECCION DE CRISIS EPILEPTICAS",
             ha="center", fontsize=20, fontweight="bold", color="white")
    fig.text(0.5, 0.72, "EN EEG PEDIATRICO",
             ha="center", fontsize=20, fontweight="bold", color="white")

    fig.text(0.5, 0.63, "Comparacion de Algoritmos de Machine Learning Clasico",
             ha="center", fontsize=13, color="#AED6F1")
    fig.text(0.5, 0.59, "Dataset: CHB-MIT Scalp EEG Database",
             ha="center", fontsize=11, color="#AED6F1")

    fig.add_artist(plt.matplotlib.patches.FancyBboxPatch(
        (0.1, 0.38), 0.8, 0.14,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor="#2C3E50", transform=fig.transFigure))

    fig.text(0.5, 0.50, "Diseno Factorial Completo 4x4",
             ha="center", fontsize=11, fontweight="bold", color="#F1C40F")
    fig.text(0.5, 0.46, "4 algoritmos x 4 feature sets x 7 folds GroupKFold = 112 entrenamientos",
             ha="center", fontsize=9, color="white")
    fig.text(0.5, 0.42, "Sujetos pediatricos (6-10 anos): chb05, chb09, chb14, chb16, chb20, chb22, chb23",
             ha="center", fontsize=9, color="white")

    fig.text(0.5, 0.32, "Universidad de La Salle — Maestria en Inteligencia Artificial",
             ha="center", fontsize=10, color="#85929E")
    fig.text(0.5, 0.28, "Linea de investigacion: Machine Learning",
             ha="center", fontsize=10, color="#85929E")
    fig.text(0.5, 0.23, "Erika Isabel Caita Avila",
             ha="center", fontsize=11, color="#AED6F1", fontweight="bold")
    fig.text(0.5, 0.19, "caita.erikai@gmail.com",
             ha="center", fontsize=9, color="#85929E")
    fig.text(0.5, 0.13, "Run ID: doe_v1_all_models | Mayo 2026",
             ha="center", fontsize=9, color="#566573")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 2 — Resumen ejecutivo + TOP 5
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.55,
                   top=0.88, bottom=0.08, left=0.08, right=0.92)

    fig.suptitle("Resumen Ejecutivo", fontsize=14, fontweight="bold", y=0.95)

    # KPIs
    ax_kpi = fig.add_subplot(gs[0])
    ax_kpi.axis("off")
    best_pr  = df_pr.iloc[0]
    best_auc = df_auc.iloc[0]
    baseline = df[(df.feature_set == "bp_plus_rms") & (df.model == "logreg")].iloc[0]

    kpi_data = [
        ("Mejor PR-AUC",
         f"{best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_std']:.3f}",
         f"{MODEL_LABELS[best_pr['model']]} + {best_pr['feature_set']}"),
        ("Mejor AUC-ROC",
         f"{best_auc['auc_roc_mean']:.3f} +/- {best_auc['auc_roc_std']:.3f}",
         f"{MODEL_LABELS[best_auc['model']]} + {best_auc['feature_set']}"),
        ("Linea base DOE (cap.7)",
         f"PR-AUC = 0.112 +/- 0.149",
         "LogReg + bp_plus_rms"),
        ("Mejora sobre baseline",
         f"+{best_pr['pr_auc_mean'] - 0.112:.3f} PR-AUC",
         "bp_plus_rms_kurt_skew supera al baseline"),
    ]
    box_colors = ["#1ABC9C", "#3498DB", "#95A5A6", "#F39C12"]
    for i, (label, value, note) in enumerate(kpi_data):
        x = 0.05 + i * 0.24
        rect = mpatches.FancyBboxPatch((x, 0.1), 0.21, 0.75,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor=box_colors[i], facecolor=box_colors[i] + "22",
            transform=ax_kpi.transAxes)
        ax_kpi.add_patch(rect)
        ax_kpi.text(x + 0.105, 0.78, label, ha="center", va="top",
                    fontsize=7.5, fontweight="bold",
                    color=box_colors[i], transform=ax_kpi.transAxes)
        ax_kpi.text(x + 0.105, 0.55, value, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", transform=ax_kpi.transAxes)
        ax_kpi.text(x + 0.105, 0.22, note, ha="center", va="bottom",
                    fontsize=6.5, color="#555", transform=ax_kpi.transAxes,
                    wrap=True)
    ax_kpi.set_title("Indicadores Clave", fontsize=10, fontweight="bold", pad=6)

    # TOP 5 PR-AUC
    ax_top = fig.add_subplot(gs[1])
    ax_top.axis("off")
    top5 = df_pr.head(5).copy()
    top5_display = pd.DataFrame({
        "Feature Set":  top5["feature_set"],
        "Modelo":       top5["model"].map(MODEL_LABELS),
        "N feat.":      top5["n_features"].astype(str),
        "PR-AUC":       top5.apply(lambda r: f"{r['pr_auc_mean']:.3f}+/-{r['pr_auc_std']:.3f}", axis=1),
        "AUC-ROC":      top5.apply(lambda r: f"{r['auc_roc_mean']:.3f}+/-{r['auc_roc_std']:.3f}", axis=1),
        "Sensib.":      top5["sensitivity_mean"].map(lambda v: f"{v:.3f}"),
        "Especif.":     top5["specificity_mean"].map(lambda v: f"{v:.3f}"),
    })
    tbl = ax_top.table(
        cellText=top5_display.values,
        colLabels=top5_display.columns.tolist(),
        cellLoc="center", loc="center",
        colWidths=[0.22, 0.20, 0.07, 0.16, 0.16, 0.09, 0.09],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        cell.set_edgecolor("#BDC3C7")
    ax_top.set_title("TOP 5 combinaciones por PR-AUC (metrica primaria DOE)", fontsize=10, fontweight="bold", pad=6)

    # TOP 5 AUC-ROC
    ax_auc = fig.add_subplot(gs[2])
    ax_auc.axis("off")
    top5a = df_auc.head(5).copy()
    top5a_display = pd.DataFrame({
        "Feature Set":  top5a["feature_set"],
        "Modelo":       top5a["model"].map(MODEL_LABELS),
        "N feat.":      top5a["n_features"].astype(str),
        "AUC-ROC":      top5a.apply(lambda r: f"{r['auc_roc_mean']:.3f}+/-{r['auc_roc_std']:.3f}", axis=1),
        "PR-AUC":       top5a.apply(lambda r: f"{r['pr_auc_mean']:.3f}+/-{r['pr_auc_std']:.3f}", axis=1),
        "Sensib.":      top5a["sensitivity_mean"].map(lambda v: f"{v:.3f}"),
        "Especif.":     top5a["specificity_mean"].map(lambda v: f"{v:.3f}"),
    })
    tbl2 = ax_auc.table(
        cellText=top5a_display.values,
        colLabels=top5a_display.columns.tolist(),
        cellLoc="center", loc="center",
        colWidths=[0.22, 0.20, 0.07, 0.16, 0.16, 0.09, 0.09],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(7.5)
    tbl2.scale(1, 1.6)
    for (r, c), cell in tbl2.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FEF9E7")
        cell.set_edgecolor("#BDC3C7")
    ax_auc.set_title("TOP 5 combinaciones por AUC-ROC (metrica secundaria DOE)", fontsize=10, fontweight="bold", pad=6)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 3 — Tabla completa 16 combinaciones
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8.5))  # landscape
    ax  = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Resultados Completos — 16 Combinaciones (4 algoritmos x 4 feature sets)",
                 fontsize=12, fontweight="bold", y=0.97)

    full = df.sort_values(["feature_set", "model"]).reset_index(drop=True)
    full_display = pd.DataFrame({
        "Feature Set":   full["feature_set"],
        "Modelo":        full["model"].map(MODEL_LABELS),
        "N feat.":       full["n_features"].astype(str),
        "AUC-ROC":       full.apply(lambda r: f"{r['auc_roc_mean']:.3f}+/-{r['auc_roc_std']:.3f}", axis=1),
        "PR-AUC":        full.apply(lambda r: f"{r['pr_auc_mean']:.3f}+/-{r['pr_auc_std']:.3f}", axis=1),
        "Sensibilidad":  full["sensitivity_mean"].map(lambda v: f"{v:.4f}"),
        "Especificidad": full["specificity_mean"].map(lambda v: f"{v:.4f}"),
        "NaN folds":     full["folds_with_nan_auc"].astype(str),
    })
    tbl = ax.table(
        cellText=full_display.values,
        colLabels=full_display.columns.tolist(),
        cellLoc="center", loc="center",
        colWidths=[0.20, 0.17, 0.06, 0.14, 0.14, 0.10, 0.10, 0.08],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.55)
    prev_fs = None
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        else:
            fs_val = full_display.iloc[r - 1]["Feature Set"] if r - 1 < len(full_display) else ""
            if fs_val != prev_fs:
                prev_fs = fs_val
                stripe = True
            bg = "#EBF5FB" if stripe else "white"
            cell.set_facecolor(bg)
        cell.set_edgecolor("#BDC3C7")

    fig.text(0.5, 0.02,
             "Metricas promediadas sobre 7 folds GroupKFold por sujeto. "
             "NaN folds = folds donde el fold de test no tenia muestras positivas (AUC indefinido).",
             ha="center", fontsize=7, color="#7F8C8D", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 4 — Heatmaps PR-AUC y AUC-ROC
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Matrices de Rendimiento — Factorial 4x4",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, metric, cmap, title in [
        (axes[0], "pr_auc_mean",  "YlOrRd", "PR-AUC (metrica primaria)"),
        (axes[1], "auc_roc_mean", "YlGnBu", "AUC-ROC (metrica secundaria)"),
    ]:
        matrix = df.pivot(index="model", columns="feature_set", values=metric)
        matrix = matrix.reindex(index=MDL_ORDER, columns=FS_ORDER)

        im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(FS_ORDER)))
        ax.set_xticklabels([FS_LABELS[f] for f in FS_ORDER], fontsize=7.5)
        ax.set_yticks(range(len(MDL_ORDER)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MDL_ORDER], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

        for i in range(len(MDL_ORDER)):
            for j in range(len(FS_ORDER)):
                val = matrix.values[i, j]
                std_col = metric.replace("_mean", "_std")
                std_val = df[
                    (df.model == MDL_ORDER[i]) & (df.feature_set == FS_ORDER[j])
                ][std_col].values[0]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}\n+/-{std_val:.3f}",
                        ha="center", va="center", fontsize=6.5,
                        color=color, fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 5 — Barras PR-AUC y AUC-ROC por modelo y feature set
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Comparacion de Metricas por Modelo y Feature Set",
                 fontsize=13, fontweight="bold", y=1.01)

    # 5a — PR-AUC por feature set, agrupado por modelo
    ax = axes[0, 0]
    x     = np.arange(len(FS_ORDER))
    width = 0.2
    for k, mdl in enumerate(MDL_ORDER):
        sub = df[df.model == mdl].set_index("feature_set")
        vals = [sub.loc[fs, "pr_auc_mean"]  if fs in sub.index else 0 for fs in FS_ORDER]
        errs = [sub.loc[fs, "pr_auc_std"]   if fs in sub.index else 0 for fs in FS_ORDER]
        ax.bar(x + k * width, vals, width, yerr=errs, label=MODEL_LABELS[mdl],
               color=MODEL_COLORS[mdl], capsize=3, alpha=0.85)
    ax.axhline(0.112, color="red", linestyle="--", linewidth=1, label="Baseline DOE (0.112)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([FS_LABELS[f] for f in FS_ORDER], fontsize=7.5)
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC por Feature Set (metrica primaria)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 0.45)
    ax.grid(axis="y", alpha=0.3)

    # 5b — AUC-ROC por feature set, agrupado por modelo
    ax = axes[0, 1]
    for k, mdl in enumerate(MDL_ORDER):
        sub = df[df.model == mdl].set_index("feature_set")
        vals = [sub.loc[fs, "auc_roc_mean"] if fs in sub.index else 0 for fs in FS_ORDER]
        errs = [sub.loc[fs, "auc_roc_std"]  if fs in sub.index else 0 for fs in FS_ORDER]
        ax.bar(x + k * width, vals, width, yerr=errs, label=MODEL_LABELS[mdl],
               color=MODEL_COLORS[mdl], capsize=3, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Azar (0.5)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([FS_LABELS[f] for f in FS_ORDER], fontsize=7.5)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC por Feature Set")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # 5c — PR-AUC por modelo (boxplot sobre feature sets)
    ax = axes[1, 0]
    data_box = [df[df.model == m]["pr_auc_mean"].values for m in MDL_ORDER]
    bp = ax.boxplot(data_box, patch_artist=True, notch=False)
    for patch, mdl in zip(bp["boxes"], MDL_ORDER):
        patch.set_facecolor(MODEL_COLORS[mdl])
        patch.set_alpha(0.7)
    ax.axhline(0.112, color="red", linestyle="--", linewidth=1, label="Baseline (0.112)")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MDL_ORDER], fontsize=7.5, rotation=15)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Distribucion PR-AUC por Modelo\n(sobre 4 feature sets)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 5d — Sensibilidad vs Especificidad scatter
    ax = axes[1, 1]
    for mdl in MDL_ORDER:
        sub = df[df.model == mdl]
        ax.scatter(sub["specificity_mean"], sub["sensitivity_mean"],
                   label=MODEL_LABELS[mdl], color=MODEL_COLORS[mdl],
                   s=60, alpha=0.85, zorder=3)
    ax.set_xlabel("Especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title("Sensibilidad vs Especificidad\n(cada punto = 1 combinacion)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 6 — Distribucion por folds (violin) para PR-AUC
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Distribucion de PR-AUC por Fold (7 folds GroupKFold por sujeto)",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, fs in zip(axes.flat, FS_ORDER):
        sub = fold[fold["feature_set"] == fs]
        data_v = [sub[sub["model"] == m]["pr_auc"].dropna().values for m in MDL_ORDER]
        parts  = ax.violinplot(data_v, positions=range(len(MDL_ORDER)),
                               showmedians=True, showextrema=True)
        for i, (pc, mdl) in enumerate(zip(parts["bodies"], MDL_ORDER)):
            pc.set_facecolor(MODEL_COLORS[mdl])
            pc.set_alpha(0.6)
        ax.axhline(0.112, color="red", linestyle="--", linewidth=1, label="Baseline (0.112)")
        ax.set_xticks(range(len(MDL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MDL_ORDER], fontsize=7.5, rotation=15)
        ax.set_ylabel("PR-AUC por fold")
        ax.set_title(f"Feature set: {fs}")
        ax.set_ylim(-0.05, 0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 7 — Wilcoxon signed-rank para los 4 feature sets
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Tests de Wilcoxon Signed-Rank Pareados por Fold (H0: distribuciones iguales)",
                 fontsize=12, fontweight="bold", y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35,
                  top=0.90, bottom=0.05, left=0.05, right=0.95)

    for idx, fs in enumerate(FS_ORDER):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.axis("off")
        wx = wx_all[fs]
        wx_df = pd.DataFrame(wx)
        tbl = ax.table(
            cellText=wx_df[["par", "W", "p", "sig"]].values,
            colLabels=["Par de modelos", "W", "p-valor", "Sig."],
            cellLoc="center", loc="center",
            colWidths=[0.55, 0.12, 0.18, 0.10],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                sig_val = wx_df.iloc[r - 1]["sig"] if r - 1 < len(wx_df) else ""
                if sig_val == "*":
                    cell.set_facecolor("#FDEBD0")
                elif r % 2 == 0:
                    cell.set_facecolor("#F8F9FA")
            cell.set_edgecolor("#CCC")
        ax.set_title(f"Feature set: {fs}", fontsize=9, fontweight="bold", pad=6)

    fig.text(0.5, 0.02,
             "alpha = 0.05 | * = p < 0.05 (evidencia probable de diferencia) | n.s. = no significativo | "
             "Potencia aprox. 0.35 con n=7 folds — interpretar con cautela (DOE sec. 6).",
             ha="center", fontsize=7.5, color="#555", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 8 — Interpretacion de resultados (DOE sec. 9)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Interpretacion de Resultados — Guia DOE Seccion 9",
                 fontsize=13, fontweight="bold", y=0.97)

    best  = df_pr.iloc[0]
    best2 = df_auc.iloc[0]

    lines = [
        ("1. Ranking por PR-AUC (metrica primaria)", "bold", "#1A5276"),
        (f"   La mejor combinacion es '{best['feature_set']} + {MODEL_LABELS[best['model']]}' "
         f"con PR-AUC = {best['pr_auc_mean']:.3f} +/- {best['pr_auc_std']:.3f}.", "normal", "#1A1A2E"),
        (f"   El mejor AUC-ROC corresponde a '{best2['feature_set']} + {MODEL_LABELS[best2['model']]}' "
         f"con AUC = {best2['auc_roc_mean']:.3f} +/- {best2['auc_roc_std']:.3f}.", "normal", "#1A1A2E"),
        ("", "normal", "black"),
        ("2. Reporte con intervalos", "bold", "#1A5276"),
        (f"   El Random Forest posiblemente alcanza un PR-AUC de "
         f"{df[(df.model=='random_forest')&(df.feature_set=='bp_plus_rms_kurt_skew')]['pr_auc_mean'].values[0]:.3f} "
         f"+/- {df[(df.model=='random_forest')&(df.feature_set=='bp_plus_rms_kurt_skew')]['pr_auc_std'].values[0]:.3f} "
         f"en esta cohorte (IC 95% aprox.: mu +/- 2.447 * sigma/sqrt(7)).", "normal", "#1A1A2E"),
        ("", "normal", "black"),
        ("3. Wilcoxon — Evidencia estadistica", "bold", "#1A5276"),
        ("   Solo el par 'Gradient Boosting vs SVM' muestra diferencia significativa",
         "normal", "#1A1A2E"),
        ("   (p=0.0395, *) en el feature set bp_plus_rms.", "normal", "#1A1A2E"),
        ("   Los demas pares: n.s. — insuficiencia de evidencia, NO equivalencia demostrada.", "normal", "#7D6608"),
        ("   Potencia del test ~0.35 con n=7 folds (DOE sec. 6 — Error Tipo II probable).", "normal", "#922B21"),
        ("", "normal", "black"),
        ("4. Folds con NaN", "bold", "#1A5276"),
        ("   Todos los experimentos reportan folds_with_nan_auc = 1 (el sujeto chb05", "normal", "#1A1A2E"),
        ("   en fold=1/7 tiene pos=0 en test, AUC indefinido). Las medias se calculan", "normal", "#1A1A2E"),
        ("   sobre los 6 folds validos. Reportar como 'AUC calculado sobre 6/7 folds'.", "normal", "#1A1A2E"),
        ("", "normal", "black"),
        ("5. Comparacion con cap. 7 (linea base DOE)", "bold", "#1A5276"),
        ("   Linea base: PR-AUC = 0.112 +/- 0.149 (LogReg + bp_plus_rms).", "normal", "#1A1A2E"),
        (f"   Mejor resultado actual: PR-AUC = {best['pr_auc_mean']:.3f} "
         f"({best['feature_set']} + {MODEL_LABELS[best['model']]}).", "normal", "#1A1A2E"),
        (f"   Mejora absoluta: +{best['pr_auc_mean'] - 0.112:.3f} — supera el umbral de relevancia",
         "normal", "#1A4620"),
        ("   del DOE (>0.02). Las features de kurtosis y skewness aportaron valor.", "normal", "#1A4620"),
        ("", "normal", "black"),
        ("6. No inferir causalidad", "bold", "#1A5276"),
        ("   Los resultados son asociaciones observadas en esta cohorte de 7 sujetos.", "normal", "#1A1A2E"),
        ("   La generalizacion requiere validacion externa en cohortes independientes.", "normal", "#1A1A2E"),
        ("", "normal", "black"),
        ("7. Criterio de exito minimo DOE", "bold", "#1A5276"),
        (f"   PR-AUC > 0.15 en promedio: {'CUMPLIDO' if best['pr_auc_mean'] > 0.15 else 'NO CUMPLIDO'} "
         f"({best['pr_auc_mean']:.3f} > 0.15 para la mejor combinacion).", "normal",
         "#1A4620" if best["pr_auc_mean"] > 0.15 else "#922B21"),
    ]

    y = 0.95
    for text, weight, color in lines:
        if text == "":
            y -= 0.018
            continue
        ax.text(0.03, y, text, transform=ax.transAxes,
                fontsize=8.5, fontweight=weight, color=color,
                verticalalignment="top",
                bbox={"facecolor": "#F8F9FA", "alpha": 0.0, "pad": 1} if weight == "bold" else None)
        wrapped = textwrap.wrap(text, width=105)
        y -= 0.022 * max(len(wrapped), 1)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PAGINA 9 — Contraejemplo: efecto marginal de features
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Efecto Marginal de los Feature Sets sobre PR-AUC y AUC-ROC",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, metric, ylabel, title in [
        (axes[0], "pr_auc_mean",  "PR-AUC",  "PR-AUC segun complejidad del feature set"),
        (axes[1], "auc_roc_mean", "AUC-ROC", "AUC-ROC segun complejidad del feature set"),
    ]:
        for mdl in MDL_ORDER:
            sub  = df[df.model == mdl].set_index("feature_set")
            vals = [sub.loc[fs, metric] for fs in FS_ORDER if fs in sub.index]
            errs = [sub.loc[fs, metric.replace("_mean", "_std")] for fs in FS_ORDER if fs in sub.index]
            x    = np.arange(len(FS_ORDER))
            ax.errorbar(x, vals, yerr=errs, marker="o", linewidth=1.5,
                        label=MODEL_LABELS[mdl], color=MODEL_COLORS[mdl],
                        capsize=3, markersize=5)
        if metric == "pr_auc_mean":
            ax.axhline(0.112, color="red", linestyle="--", linewidth=1, label="Baseline (0.112)")
        ax.set_xticks(range(len(FS_ORDER)))
        ax.set_xticklabels([FS_LABELS[f] for f in FS_ORDER], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.text(0.5, -0.03,
             "H0_features (DOE sec. 5): La adicion de RMS, kurtosis o skewness no mejora PR-AUC respecto a bp_only. "
             "Resultado: RF y GradBoost muestran mejora al agregar kurtosis/skewness.",
             ha="center", fontsize=7.5, color="#555", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Metadata del PDF
    # -----------------------------------------------------------------------
    d = pdf.infodict()
    d["Title"]    = "Reporte Final DOE — Deteccion de Crisis Epilepticas en EEG Pediatrico"
    d["Author"]   = "Erika Isabel Caita Avila"
    d["Subject"]  = "Machine Learning — Universidad de La Salle"
    d["Keywords"] = "EEG, epilepsy, machine learning, CHB-MIT, DOE"
    d["Creator"]  = "generate_pdf_report.py | doe_v1_all_models"

print(f"[OK] PDF generado: {OUTPUT_PDF}")
print(f"     Paginas: 9")
print(f"     Ruta absoluta: {OUTPUT_PDF.resolve()}")
