"""
thesis_dashboard.py — Dashboard interactivo para la tesis de detección de
crisis epilépticas focales pediátricas (CHB-MIT).

Lanzamiento:
    streamlit run thesis_dashboard.py --server.port 8501

Secciones:
    1. Diseño Experimental (DoE)
    2. Datos
    3. Resultados ML Clásico
    4. Resultados Deep Learning
    5. Comparación Final
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Configuración de la app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tesis EEG — Detección de Crisis (CHB-MIT)",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "out_thesis_final"

CLASSICAL_DIR_DEFAULT = DEFAULT_OUT / "runs" / "classical_all_models"
CLASSICAL_FALLBACK = DEFAULT_OUT / "runs" / "baseline_classical"
CNN_DIR_DEFAULT = DEFAULT_OUT / "cnn_stft"
CNN_FALLBACK = ROOT / "out_smoke_cnn"
LSTM_DIR_DEFAULT = DEFAULT_OUT / "lstm_raw"
LSTM_FALLBACK = ROOT / "out_smoke_lstm"
DATASET_CSV_DEFAULT = CLASSICAL_FALLBACK / "windows_dataset.csv"
COMPARISON_CSV_DEFAULT = DEFAULT_OUT / "comparison_table.csv"


# ---------------------------------------------------------------------------
# Helpers de carga (con caché de Streamlit)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        st.error(f"Error leyendo {p.name}: {exc}")
        return None


def first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def status_badge(label: str, ok: bool, info: str = "") -> str:
    icon = "[OK]" if ok else "[--]"
    extra = f" — {info}" if info else ""
    return f"{icon} **{label}**{extra}"


def banner_pendiente(msg: str) -> None:
    st.warning(f"{msg}\n\nEjecuta el script correspondiente para generar resultados.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Tesis EEG")
st.sidebar.caption("Detección de crisis epilépticas focales pediátricas (CHB-MIT)")

st.sidebar.markdown("---")
seccion = st.sidebar.radio(
    "Secciones",
    [
        "1. Diseño Experimental (DoE)",
        "2. Datos",
        "3. Resultados ML Clásico",
        "4. Resultados Deep Learning",
        "5. Comparación Final",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado de runs**")

classical_path = first_existing(
    [CLASSICAL_DIR_DEFAULT / "results.csv", CLASSICAL_FALLBACK / "results.csv"]
)
cnn_path = first_existing(
    [CNN_DIR_DEFAULT / "results.csv", CNN_FALLBACK / "results.csv"]
)
lstm_path = first_existing(
    [LSTM_DIR_DEFAULT / "results.csv", LSTM_FALLBACK / "results.csv"]
)
dataset_path = first_existing([DATASET_CSV_DEFAULT])

st.sidebar.markdown(
    status_badge("Classical", classical_path is not None,
                 classical_path.parent.name if classical_path else "pendiente")
)
st.sidebar.markdown(
    status_badge("CNN STFT", cnn_path is not None,
                 cnn_path.parent.name if cnn_path else "pendiente")
)
st.sidebar.markdown(
    status_badge("LSTM raw", lstm_path is not None,
                 lstm_path.parent.name if lstm_path else "pendiente")
)
st.sidebar.markdown(
    status_badge("Dataset",  dataset_path is not None,
                 f"{dataset_path.parent.name}" if dataset_path else "pendiente")
)


# ---------------------------------------------------------------------------
# Sección 1: Diseño Experimental (DoE)
# ---------------------------------------------------------------------------

if seccion.startswith("1."):
    st.title("1. Diseño Experimental")

    st.markdown("""
    ### Delimitación del problema
    **Predicción/detección de crisis epilépticas focales pediátricas** a partir
    del EEG superficial multicanal del dataset *CHB-MIT Scalp EEG Database*
    (PhysioNet). El objetivo cuantitativo es discriminar ventanas de 5 segundos
    de EEG entre clase **ictal** (`label=1`) y **interictal** (`label=0`) bajo
    un esquema de validación inter-sujeto que mide la capacidad de generalización
    del modelo a pacientes nuevos.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### Variables de respuesta
        | Métrica            | Rol         | Justificación                                  |
        |--------------------|-------------|-----------------------------------------------|
        | **PR-AUC**         | Primaria    | Robusta a desbalance extremo (~1% positivos)  |
        | **AUC-ROC**        | Secundaria  | Comparable con literatura clínica              |
        | **Sensitivity**    | Secundaria  | Crítica clínicamente (no perder crisis)       |
        | **Specificity**    | Secundaria  | Falsos positivos por hora                     |
        | **Accuracy**       | Auxiliar    | Engañosa con desbalance pero exigida en tesis |
        """)

    with col_b:
        st.markdown("""
        ### Factores y niveles
        | Factor              | Tipo         | Niveles                                              |
        |---------------------|--------------|------------------------------------------------------|
        | **Tipo de modelo**  | Fijo         | LogReg · RF · SVM · GradientBoosting · CNN · BiLSTM  |
        | **Feature set**     | Fijo         | bp_only · bp+rms · bp+rms+kurt · bp+rms+kurt+skew    |
        | **Sujeto (held-out)** | Aleatorio  | 7 sujetos (chb05/09/14/16/20/22/23)                  |
        | **Ventana**         | Constante    | 5.0 s con 50 % overlap                               |
        """)

    st.markdown("""
    ### Clasificación del diseño
    - **Comparación de tratamientos** entre familias de modelos (clásico vs DL).
    - **Diseño factorial parcial 6 × 4** sobre modelo × feature set para los
      clásicos; los DL no usan features tabulares.
    - **Diseño robusto inter-sujeto**: el sujeto entra como factor aleatorio
      mediante `GroupKFold(n_splits=7)` — cada fold deja a *un* paciente
      completamente fuera de entrenamiento.
    - **Bloqueo**: las ventanas de un mismo EDF nunca se reparten entre train
      y test; el grupo es el `subject_id`.

    ### Plan experimental y protocolo
    1. Extracción de ventanas de 5 s con 50 % de overlap, descartando segmentos
       a `<exclude_margin_sec>` de los marcadores de crisis para evitar fugas.
    2. Para cada combinación `(modelo, feature_set)` se entrena bajo
       `GroupKFold(n_splits=7)`.
    3. En cada fold se calculan PR-AUC, AUC-ROC, sensibilidad, especificidad,
       exactitud y se reporta media ± desviación entre folds.
    4. Las pruebas estadísticas inter-modelos son **Wilcoxon signed-rank** sobre
       PR-AUC pareada por fold (potencia limitada a 7 muestras).
    """)

    st.info(
        "Hipótesis nula H0: la PR-AUC media de los modelos comparados es igual. "
        "Se rechaza si p < 0.05 en Wilcoxon."
    )


# ---------------------------------------------------------------------------
# Sección 2: Datos
# ---------------------------------------------------------------------------

elif seccion.startswith("2."):
    st.title("2. Datos")

    if dataset_path is None:
        banner_pendiente("Dataset de ventanas no encontrado.")
    else:
        df = load_csv(str(dataset_path))
        if df is None or df.empty:
            banner_pendiente("Dataset vacío o ilegible.")
        else:
            n_total = len(df)
            n_pos = int(df["label"].sum())
            n_neg = n_total - n_pos
            pos_rate = df["label"].mean()
            n_subjects = df["subject_id"].nunique()
            n_edfs = df["edf_file"].nunique()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ventanas totales", f"{n_total:,}")
            c2.metric("Ictales (positivas)", f"{n_pos:,}", f"{pos_rate*100:.2f} %")
            c3.metric("Sujetos", f"{n_subjects}")
            c4.metric("Archivos EDF", f"{n_edfs}")

            st.markdown("### Distribución por sujeto")
            grp = (
                df.groupby("subject_id")["label"]
                .agg(pos="sum", total="count", rate="mean")
                .reset_index()
                .sort_values("subject_id")
            )
            grp["pos"] = grp["pos"].astype(int)
            grp["pct_ictal"] = grp["rate"] * 100

            fig = px.bar(
                grp,
                x="subject_id",
                y="total",
                color="pct_ictal",
                color_continuous_scale="Reds",
                hover_data={"pos": True, "rate": ":.4f", "pct_ictal": ":.2f"},
                labels={
                    "subject_id": "Sujeto",
                    "total": "Ventanas totales",
                    "pct_ictal": "% ictales",
                },
                title="Ventanas por sujeto (color = % ictales)",
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Tabla por sujeto")
            st.dataframe(
                grp.rename(
                    columns={
                        "subject_id": "Sujeto",
                        "pos": "Ventanas ictales",
                        "total": "Ventanas totales",
                        "rate": "Tasa positiva",
                        "pct_ictal": "% ictales",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Vista previa del dataset (primeras 20 filas)"):
                st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sección 3: Resultados ML Clásico
# ---------------------------------------------------------------------------

elif seccion.startswith("3."):
    st.title("3. Resultados — Modelos Clásicos")

    if classical_path is None:
        banner_pendiente("Aún no hay results.csv del run clásico.")
        st.stop()

    df = load_csv(str(classical_path))
    if df is None or df.empty:
        banner_pendiente("results.csv clásico vacío.")
        st.stop()

    st.caption(f"Fuente: `{classical_path}`")

    # Métricas clave
    n_models = df["model"].nunique()
    n_fs = df["feature_set"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Combinaciones (modelo × feature)", f"{len(df)}")
    c2.metric("Modelos", f"{n_models}")
    c3.metric("Feature sets", f"{n_fs}")

    st.markdown("### Tabla de resultados")
    cols_show = [
        "model", "feature_set", "n_features",
        "auc_roc_mean", "auc_roc_std",
        "pr_auc_mean", "pr_auc_std",
        "sensitivity_mean", "specificity_mean",
        "folds_with_nan_auc",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    df_view = df[cols_show].sort_values("pr_auc_mean", ascending=False)
    st.dataframe(
        df_view.style.format(
            {
                "auc_roc_mean": "{:.4f}",
                "auc_roc_std": "{:.4f}",
                "pr_auc_mean": "{:.4f}",
                "pr_auc_std": "{:.4f}",
                "sensitivity_mean": "{:.4f}",
                "specificity_mean": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Heatmap modelo × feature_set por PR-AUC
    st.markdown("### Heatmap PR-AUC (modelo × feature set)")
    pivot = df.pivot_table(
        index="model", columns="feature_set", values="pr_auc_mean", aggfunc="mean"
    )
    fig = px.imshow(
        pivot,
        text_auto=".3f",
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(color="PR-AUC"),
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot por fold (si hay cv_fold_metrics.csv)
    fold_path = classical_path.parent / "cv_fold_metrics.csv"
    if fold_path.exists():
        fold_df = load_csv(str(fold_path))
        if fold_df is not None and not fold_df.empty:
            st.markdown("### Distribución de PR-AUC por fold")
            feature_sets = sorted(fold_df["feature_set"].unique())
            chosen_fs = st.selectbox(
                "Feature set", feature_sets,
                index=feature_sets.index("bp_plus_rms")
                if "bp_plus_rms" in feature_sets else 0,
            )
            sub = fold_df[fold_df["feature_set"] == chosen_fs].dropna(subset=["pr_auc"])
            fig = px.box(
                sub, x="model", y="pr_auc", points="all",
                color="model",
                title=f"PR-AUC por fold — feature set: {chosen_fs}",
            )
            fig.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("`cv_fold_metrics.csv` no disponible — no se puede mostrar boxplot por fold.")


# ---------------------------------------------------------------------------
# Sección 4: Resultados Deep Learning
# ---------------------------------------------------------------------------

elif seccion.startswith("4."):
    st.title("4. Resultados — Deep Learning")

    tab_cnn, tab_lstm = st.tabs(["CNN (espectrograma STFT)", "BiLSTM (EEG crudo)"])

    def render_dl(results_path: Optional[Path], modelo: str) -> None:
        if results_path is None:
            banner_pendiente(f"Sin resultados aún para {modelo}.")
            return
        df = load_csv(str(results_path))
        if df is None or df.empty:
            banner_pendiente(f"results.csv vacío para {modelo}.")
            return

        st.caption(f"Fuente: `{results_path}`")

        cols_show = [
            "model", "feature_set", "n_features",
            "auc_roc_mean", "auc_roc_std",
            "pr_auc_mean", "pr_auc_std",
            "sensitivity_mean", "specificity_mean",
            "folds_with_nan_auc",
        ]
        cols_show = [c for c in cols_show if c in df.columns]

        st.dataframe(
            df[cols_show].style.format(
                {
                    "auc_roc_mean": "{:.4f}",
                    "auc_roc_std": "{:.4f}",
                    "pr_auc_mean": "{:.4f}",
                    "pr_auc_std": "{:.4f}",
                    "sensitivity_mean": "{:.4f}",
                    "specificity_mean": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        manifest_path = results_path.parent / "run_manifest.json"
        runs_dir = results_path.parent / "runs"
        if not manifest_path.exists() and runs_dir.exists():
            for r in sorted(runs_dir.iterdir(), reverse=True):
                cand = r / "run_manifest.json"
                if cand.exists():
                    manifest_path = cand
                    break

        if manifest_path.exists():
            with st.expander("Manifiesto del run"):
                with open(manifest_path) as f:
                    st.json(json.load(f))

        fold_path = results_path.parent / "cv_fold_metrics.csv"
        if not fold_path.exists() and runs_dir.exists():
            for r in sorted(runs_dir.iterdir(), reverse=True):
                cand = r / "cv_fold_metrics.csv"
                if cand.exists():
                    fold_path = cand
                    break

        if fold_path.exists():
            fold_df = load_csv(str(fold_path))
            if fold_df is not None and not fold_df.empty:
                st.markdown("#### Métricas por fold")
                st.dataframe(
                    fold_df.style.format(
                        {c: "{:.4f}" for c in ["auc_roc", "pr_auc", "sensitivity",
                                               "specificity", "accuracy"]
                         if c in fold_df.columns}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                fig = px.bar(
                    fold_df, x="fold", y="pr_auc",
                    color="test_subjects" if "test_subjects" in fold_df.columns else None,
                    title=f"{modelo} — PR-AUC por fold",
                    labels={"pr_auc": "PR-AUC", "fold": "Fold"},
                )
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin métricas por fold disponibles.")

    with tab_cnn:
        render_dl(cnn_path, "CNN STFT")
    with tab_lstm:
        render_dl(lstm_path, "BiLSTM raw")


# ---------------------------------------------------------------------------
# Sección 5: Comparación Final
# ---------------------------------------------------------------------------

elif seccion.startswith("5."):
    st.title("5. Comparación Final — Todos los Modelos")

    cmp_path = COMPARISON_CSV_DEFAULT if COMPARISON_CSV_DEFAULT.exists() else None
    df_cmp: Optional[pd.DataFrame] = None
    if cmp_path is not None:
        df_cmp = load_csv(str(cmp_path))

    if df_cmp is None or df_cmp.empty:
        st.info("`comparison_table.csv` aún no existe — construyendo tabla en vivo.")
        rows = []
        for path, role in [(classical_path, "classical"),
                           (cnn_path, "cnn"),
                           (lstm_path, "lstm")]:
            if path is None:
                continue
            d = load_csv(str(path))
            if d is None or d.empty:
                continue
            for _, r in d.iterrows():
                rows.append({
                    "model": r.get("model"),
                    "feature_set": r.get("feature_set"),
                    "category": "Deep Learning" if role in ("cnn", "lstm")
                                else "Classical ML",
                    "pr_auc_mean": r.get("pr_auc_mean", float("nan")),
                    "pr_auc_std": r.get("pr_auc_std", float("nan")),
                    "auc_roc_mean": r.get("auc_roc_mean", float("nan")),
                    "auc_roc_std": r.get("auc_roc_std", float("nan")),
                    "sensitivity_mean": r.get("sensitivity_mean", float("nan")),
                    "specificity_mean": r.get("specificity_mean", float("nan")),
                })
        if not rows:
            banner_pendiente("Sin resultados para consolidar todavía.")
            st.stop()
        df_cmp = pd.DataFrame(rows)

    df_cmp = df_cmp.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)

    st.markdown("### Tabla comparativa global")
    cols = [c for c in [
        "model", "feature_set", "category", "pr_auc_mean", "pr_auc_std",
        "auc_roc_mean", "auc_roc_std", "sensitivity_mean", "specificity_mean",
    ] if c in df_cmp.columns]
    st.dataframe(
        df_cmp[cols].style.format(
            {c: "{:.4f}" for c in cols if c not in ("model", "feature_set", "category")}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Ranking PR-AUC (mean ± std)")
    df_plot = df_cmp.copy()
    df_plot["label_full"] = df_plot.apply(
        lambda r: f"{r['model']} · {r.get('feature_set','—')}", axis=1
    )
    fig = go.Figure()
    fig.add_bar(
        x=df_plot["label_full"],
        y=df_plot["pr_auc_mean"],
        error_y=dict(type="data", array=df_plot["pr_auc_std"]),
        marker_color=[
            "#1f77b4" if c == "Classical ML" else "#d62728"
            for c in df_plot.get("category", ["Classical ML"] * len(df_plot))
        ],
        text=[f"{m:.3f}" for m in df_plot["pr_auc_mean"]],
        textposition="outside",
    )
    fig.update_layout(
        height=460,
        xaxis_tickangle=-40,
        yaxis_title="PR-AUC",
        margin=dict(l=0, r=0, t=10, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Wilcoxon en vivo ──────────────────────────────────────────────────
    st.markdown("### Test estadístico — Wilcoxon signed-rank (PR-AUC por fold)")
    st.caption(
        "H0: las distribuciones de PR-AUC pareadas por fold son iguales. "
        "Con 7 folds la potencia es baja — interpretar con cautela."
    )

    def fold_pr_auc(path: Optional[Path], model_name: str,
                    feature_set: Optional[str] = None) -> Optional[pd.Series]:
        if path is None:
            return None
        candidates = [path.parent / "cv_fold_metrics.csv"]
        runs_dir = path.parent / "runs"
        if runs_dir.exists():
            for r in sorted(runs_dir.iterdir(), reverse=True):
                candidates.append(r / "cv_fold_metrics.csv")
        for c in candidates:
            if c.exists():
                d = load_csv(str(c))
                if d is None or d.empty:
                    continue
                sub = d[d["model"] == model_name]
                if feature_set and "feature_set" in sub.columns:
                    sub_fs = sub[sub["feature_set"] == feature_set]
                    if not sub_fs.empty:
                        sub = sub_fs
                series = sub["pr_auc"].dropna().reset_index(drop=True)
                if len(series):
                    return series
        return None

    rf_fold = fold_pr_auc(classical_path, "random_forest", "bp_plus_rms_kurt")
    cnn_fold = fold_pr_auc(cnn_path, "cnn_spectrogram")
    lstm_fold = fold_pr_auc(lstm_path, "lstm_bilstm")

    rows = []
    pairs = [
        ("RandomForest", rf_fold, "CNN_STFT", cnn_fold),
        ("RandomForest", rf_fold, "BiLSTM", lstm_fold),
        ("CNN_STFT", cnn_fold, "BiLSTM", lstm_fold),
    ]
    for la, a, lb, b in pairs:
        if a is None or b is None:
            rows.append({"A": la, "B": lb, "n_folds": 0, "stat": np.nan,
                         "p_value": np.nan, "verdict": "datos insuficientes"})
            continue
        n = min(len(a), len(b))
        if n < 3:
            rows.append({"A": la, "B": lb, "n_folds": n, "stat": np.nan,
                         "p_value": np.nan, "verdict": "<3 folds"})
            continue
        a_v, b_v = a.values[:n], b.values[:n]
        if np.allclose(a_v, b_v):
            rows.append({"A": la, "B": lb, "n_folds": n, "stat": 0.0,
                         "p_value": 1.0, "verdict": "idénticos"})
            continue
        try:
            stat, p = stats.wilcoxon(a_v, b_v)
            verdict = "rechaza H0 (*)" if p < 0.05 else "no rechaza H0"
            rows.append({"A": la, "B": lb, "n_folds": n,
                         "stat": float(stat), "p_value": float(p),
                         "verdict": verdict})
        except Exception as exc:
            rows.append({"A": la, "B": lb, "n_folds": n, "stat": np.nan,
                         "p_value": np.nan, "verdict": f"error: {exc}"})

    st.dataframe(
        pd.DataFrame(rows).style.format({"stat": "{:.2f}", "p_value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # ── Export LaTeX ───────────────────────────────────────────────────────
    st.markdown("### Exportar tabla a LaTeX (para tesis)")
    latex_df = df_cmp[cols].copy()
    for c in cols:
        if c not in ("model", "feature_set", "category"):
            latex_df[c] = latex_df[c].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—"
            )
    latex_str = latex_df.to_latex(index=False, escape=True)
    st.code(latex_str, language="latex")

    st.download_button(
        "Descargar comparison_table.csv",
        df_cmp.to_csv(index=False).encode(),
        file_name="comparison_table.csv",
        mime="text/csv",
    )
