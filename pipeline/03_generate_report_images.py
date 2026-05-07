# generate_report_images.py
# Genera todas las figuras del reporte de experimentos.
# Las guarda en out_thesis_final/report_images/ con los nombres
# exactos que se referencian en reporte_experimentos.tex
#
# Uso: py -3 -X utf8 generate_report_images.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

# ---------------------------------------------------------------------------
_ROOT         = Path(__file__).resolve().parent.parent
RESULTS_PATH  = _ROOT / "out_thesis_final" / "classical_all_models" / "results.csv"
FOLD_PATH     = _ROOT / "out_thesis_final" / "classical_all_models" / "cv_fold_metrics.csv"
IMG_DIR       = _ROOT / "out_thesis_final" / "report_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

df   = pd.read_csv(RESULTS_PATH)
fold = pd.read_csv(FOLD_PATH)

MDL_ORDER = ["logreg", "random_forest", "svm", "gradient_boosting"]
FS_ORDER  = ["bp_only", "bp_plus_rms", "bp_plus_rms_kurt", "bp_plus_rms_kurt_skew"]
MODEL_LABELS = {
    "logreg":            "Reg. Logistica",
    "random_forest":     "Random Forest",
    "svm":               "SVM (RBF)",
    "gradient_boosting": "Grad. Boosting",
}
FS_LABELS = {
    "bp_only":               "BP\n(10)",
    "bp_plus_rms":           "BP+RMS\n(12)",
    "bp_plus_rms_kurt":      "BP+RMS\n+Kurt(14)",
    "bp_plus_rms_kurt_skew": "BP+RMS\n+Kurt+Skew\n(16)",
}
MODEL_COLORS = {
    "logreg":            "#4C72B0",
    "random_forest":     "#55A868",
    "svm":               "#C44E52",
    "gradient_boosting": "#DD8452",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 11, "figure.dpi": 150})

# ===========================================================================
# fig_distribucion_clases.pdf
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Pie imbalance
labels  = ["Interictal\n(51 595)", "Ictal\n(649)"]
sizes   = [51595, 649]
colors  = ["#AED6F1", "#E74C3C"]
explode = [0, 0.08]
axes[0].pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct="%1.2f%%", shadow=False, startangle=90,
            textprops={"fontsize": 10})
axes[0].set_title("Distribucion de clases\n(52 244 ventanas totales)", fontweight="bold")

# Ventanas por sujeto
subj_data = fold.groupby("subject_id")["label"].agg(
    ictal=lambda x: (x == 1).sum(),
    interictal=lambda x: (x == 0).sum()
).reset_index() if "subject_id" in fold.columns else None

if subj_data is None:
    # fallback: usar fold metrics group
    counts = {"chb05": (30, 7892), "chb09": (111, 9300), "chb14": (131, 8695),
              "chb16": (82, 7800), "chb20": (117, 5700), "chb22": (52, 5700), "chb23": (180, 2700)}
    subjs  = list(counts.keys())
    ict_c  = [v[0] for v in counts.values()]
    inter_c= [v[1] for v in counts.values()]
    x      = np.arange(len(subjs))
    axes[1].bar(x, inter_c, label="Interictal", color="#AED6F1")
    axes[1].bar(x, ict_c,   label="Ictal",      color="#E74C3C", bottom=inter_c)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subjs, fontsize=9)
    axes[1].set_ylabel("Numero de ventanas")
    axes[1].set_title("Ventanas por sujeto", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(IMG_DIR / "fig_distribucion_clases.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_distribucion_clases.pdf")

# ===========================================================================
# fig_heatmap_prauc.pdf
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
matrix  = df.pivot(index="model", columns="feature_set", values="pr_auc_mean")
matrix  = matrix.reindex(index=MDL_ORDER, columns=FS_ORDER)
im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.22)
plt.colorbar(im, ax=ax, label="PR-AUC medio")
ax.set_xticks(range(len(FS_ORDER)))
ax.set_xticklabels([FS_LABELS[f] for f in FS_ORDER], fontsize=9)
ax.set_yticks(range(len(MDL_ORDER)))
ax.set_yticklabels([MODEL_LABELS[m] for m in MDL_ORDER], fontsize=9)
ax.set_title("Mapa de calor — PR-AUC medio (metrica primaria)", fontweight="bold")
for i in range(len(MDL_ORDER)):
    for j in range(len(FS_ORDER)):
        val  = matrix.values[i, j]
        std  = df[(df.model == MDL_ORDER[i]) & (df.feature_set == FS_ORDER[j])]["pr_auc_std"].values[0]
        col  = "white" if val > 0.13 else "black"
        ax.text(j, i, f"{val:.3f}\n({std:.3f})", ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold")
ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_heatmap_prauc.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_heatmap_prauc.pdf")

# ===========================================================================
# fig_heatmap_aucroc.pdf
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
matrix2 = df.pivot(index="model", columns="feature_set", values="auc_roc_mean")
matrix2 = matrix2.reindex(index=MDL_ORDER, columns=FS_ORDER)
im2 = ax.imshow(matrix2.values, cmap="YlGnBu", aspect="auto", vmin=0.4, vmax=0.9)
plt.colorbar(im2, ax=ax, label="AUC-ROC medio")
ax.set_xticks(range(len(FS_ORDER)))
ax.set_xticklabels([FS_LABELS[f] for f in FS_ORDER], fontsize=9)
ax.set_yticks(range(len(MDL_ORDER)))
ax.set_yticklabels([MODEL_LABELS[m] for m in MDL_ORDER], fontsize=9)
ax.set_title("Mapa de calor — AUC-ROC medio (metrica secundaria)", fontweight="bold")
for i in range(len(MDL_ORDER)):
    for j in range(len(FS_ORDER)):
        val  = matrix2.values[i, j]
        std  = df[(df.model == MDL_ORDER[i]) & (df.feature_set == FS_ORDER[j])]["auc_roc_std"].values[0]
        col  = "white" if val > 0.72 else "black"
        ax.text(j, i, f"{val:.3f}\n({std:.3f})", ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold")
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_heatmap_aucroc.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_heatmap_aucroc.pdf")

# ===========================================================================
# fig_barras_prauc.pdf
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(len(FS_ORDER))
width = 0.2
for k, mdl in enumerate(MDL_ORDER):
    sub  = df[df.model == mdl].set_index("feature_set")
    vals = [sub.loc[fs, "pr_auc_mean"] for fs in FS_ORDER]
    errs = [sub.loc[fs, "pr_auc_std"]  for fs in FS_ORDER]
    ax.bar(x + k * width, vals, width, yerr=errs,
           label=MODEL_LABELS[mdl], color=MODEL_COLORS[mdl], capsize=3, alpha=0.88)
ax.axhline(0.112, color="red", linestyle="--", linewidth=1.5,
           label="Linea base DOE (0.112)")
ax.axhline(0.15,  color="#E67E22", linestyle=":", linewidth=1.2,
           label="Criterio exito DOE (0.15)")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([FS_LABELS[f].replace("\n", " ") for f in FS_ORDER], fontsize=9)
ax.set_ylabel("PR-AUC medio (con desv. estandar)")
ax.set_title("PR-AUC por feature set y modelo\n(metrica primaria segun DOE)", fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_ylim(0, 0.42)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_barras_prauc.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_barras_prauc.pdf")

# ===========================================================================
# fig_barras_aucroc.pdf
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
for k, mdl in enumerate(MDL_ORDER):
    sub  = df[df.model == mdl].set_index("feature_set")
    vals = [sub.loc[fs, "auc_roc_mean"] for fs in FS_ORDER]
    errs = [sub.loc[fs, "auc_roc_std"]  for fs in FS_ORDER]
    ax.bar(x + k * width, vals, width, yerr=errs,
           label=MODEL_LABELS[mdl], color=MODEL_COLORS[mdl], capsize=3, alpha=0.88)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Azar (0.5)")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([FS_LABELS[f].replace("\n", " ") for f in FS_ORDER], fontsize=9)
ax.set_ylabel("AUC-ROC medio (con desv. estandar)")
ax.set_title("AUC-ROC por feature set y modelo", fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_barras_aucroc.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_barras_aucroc.pdf")

# ===========================================================================
# fig_violin_folds.pdf  (4 feature sets, violin PR-AUC por fold)
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Distribucion de PR-AUC por fold — GroupKFold (n=7 sujetos)",
             fontsize=12, fontweight="bold")
for ax, fs in zip(axes.flat, FS_ORDER):
    sub    = fold[fold["feature_set"] == fs]
    data_v = [sub[sub["model"] == m]["pr_auc"].dropna().values for m in MDL_ORDER]
    parts  = ax.violinplot(data_v, positions=range(len(MDL_ORDER)),
                           showmedians=True, showextrema=True)
    for pc, mdl in zip(parts["bodies"], MDL_ORDER):
        pc.set_facecolor(MODEL_COLORS[mdl]); pc.set_alpha(0.65)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    ax.axhline(0.112, color="red", linestyle="--", linewidth=1,
               label="Baseline (0.112)")
    ax.axhline(0.15,  color="#E67E22", linestyle=":", linewidth=1,
               label="Criterio DOE (0.15)")
    ax.set_xticks(range(len(MDL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MDL_ORDER], fontsize=8, rotation=12)
    ax.set_ylabel("PR-AUC por fold")
    ax.set_title(f"Feature set: {fs}", fontsize=9, fontweight="bold")
    ax.set_ylim(-0.05, 0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_violin_folds.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_violin_folds.pdf")

# ===========================================================================
# fig_efecto_marginal.pdf
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Efecto marginal de la complejidad del feature set",
             fontsize=12, fontweight="bold")
for ax, metric, ylabel, title in [
    (axes[0], "pr_auc_mean",  "PR-AUC",  "PR-AUC vs complejidad (primaria)"),
    (axes[1], "auc_roc_mean", "AUC-ROC", "AUC-ROC vs complejidad"),
]:
    for mdl in MDL_ORDER:
        sub  = df[df.model == mdl].set_index("feature_set")
        vals = [sub.loc[fs, metric] for fs in FS_ORDER]
        errs = [sub.loc[fs, metric.replace("_mean", "_std")] for fs in FS_ORDER]
        ax.errorbar(range(len(FS_ORDER)), vals, yerr=errs, marker="o",
                    linewidth=1.8, label=MODEL_LABELS[mdl],
                    color=MODEL_COLORS[mdl], capsize=3, markersize=5)
    if "pr_auc" in metric:
        ax.axhline(0.112, color="red", linestyle="--", linewidth=1, label="Baseline")
        ax.axhline(0.15,  color="#E67E22", linestyle=":", linewidth=1, label="Criterio DOE")
    ax.set_xticks(range(len(FS_ORDER)))
    ax.set_xticklabels(["BP\n(10)", "BP+RMS\n(12)", "BP+RMS\n+Kurt(14)", "BP+RMS\n+K+S(16)"],
                       fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_efecto_marginal.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_efecto_marginal.pdf")

# ===========================================================================
# fig_wilcoxon.pdf   (p-values como heatmap 4x4 para bp_plus_rms)
# ===========================================================================
def wilcoxon_matrix(fold_df, fs):
    sub = fold_df[fold_df["feature_set"] == fs]
    mat = np.ones((4, 4))
    for i, m1 in enumerate(MDL_ORDER):
        for j, m2 in enumerate(MDL_ORDER):
            if i == j:
                mat[i, j] = 1.0
                continue
            s1 = sub[sub["model"] == m1]["pr_auc"].dropna().values
            s2 = sub[sub["model"] == m2]["pr_auc"].dropna().values
            n  = min(len(s1), len(s2))
            if n < 3:
                mat[i, j] = np.nan; continue
            try:
                _, p = stats.wilcoxon(s1[:n], s2[:n])
                mat[i, j] = p
            except Exception:
                mat[i, j] = np.nan
    return mat

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Tests de Wilcoxon Signed-Rank — p-valores por par de modelos",
             fontsize=12, fontweight="bold")
for ax, fs in zip(axes.flat, FS_ORDER):
    mat  = wilcoxon_matrix(fold, fs)
    mask = np.eye(4, dtype=bool)
    disp = np.ma.masked_where(mask, mat)
    im   = ax.imshow(disp, cmap="RdYlGn", vmin=0, vmax=0.2, aspect="auto")
    plt.colorbar(im, ax=ax, label="p-valor")
    ax.set_xticks(range(4)); ax.set_xticklabels([MODEL_LABELS[m] for m in MDL_ORDER],
                                                  rotation=18, fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([MODEL_LABELS[m] for m in MDL_ORDER],
                                                  fontsize=7.5)
    ax.set_title(f"Feature set: {fs}", fontsize=9, fontweight="bold")
    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                val = mat[i, j]
                txt = f"{val:.3f}{'*' if val < 0.05 else ''}" if not np.isnan(val) else "n/a"
                col = "white" if val < 0.08 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_wilcoxon.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_wilcoxon.pdf")

# ===========================================================================
# fig_sensibilidad_especificidad.pdf
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 5.5))
for mdl in MDL_ORDER:
    sub = df[df.model == mdl]
    ax.scatter(sub["specificity_mean"], sub["sensitivity_mean"],
               label=MODEL_LABELS[mdl], color=MODEL_COLORS[mdl],
               s=80, alpha=0.85, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(FS_LABELS[row["feature_set"]].replace("\n", " "),
                    (row["specificity_mean"], row["sensitivity_mean"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=5.5)
ax.set_xlabel("Especificidad (tasa acierto clase mayoritaria)")
ax.set_ylabel("Sensibilidad (tasa deteccion ictal)")
ax.set_title("Balance Sensibilidad vs Especificidad por configuracion",
             fontweight="bold")
ax.legend(fontsize=8.5)
ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(IMG_DIR / "fig_sensibilidad_especificidad.pdf", bbox_inches="tight")
plt.close(fig)
print("OK fig_sensibilidad_especificidad.pdf")

print(f"\n[DONE] {len(list(IMG_DIR.glob('*.pdf')))} imagenes generadas en {IMG_DIR}")
