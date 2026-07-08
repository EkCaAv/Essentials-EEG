# interface/app.py
"""
Interfaz explicativa del pipeline EEG pediátrico (Fase 1).

Ejecutar:
    streamlit run interface/app.py
o bien:
    .\run_interface.ps1
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite importar `config` e `interface` al correr con `streamlit run`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config.profiles import list_profiles, get_profile, suggest_profile, get_default_profile
from config.subject_metadata import SUBJECTS_DB, get_subjects
from interface import content as C
from interface import data_loader as DL
from interface import runner as R
from interface import theme

# --------------------------------------------------------------------------
# Configuración de página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=C.APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUBJECT_DF = pd.DataFrame(
    [{"sujeto": sid, "edad": m["age"], "sexo": m["gender"], "grupo": m["group"]}
     for sid, m in SUBJECTS_DB.items()]
)

# Paleta clínica sobria (centralizada en theme.py)
BAND_COLORS = theme.BAND_COLORS
COHORT_GRAY = "#C9D2D7"
ACCENT = theme.PETROL
SHADE = "#3E7C8C"


# --------------------------------------------------------------------------
# Helpers de visualización
# --------------------------------------------------------------------------
def plot_band_profile(profile, reference=None):
    """Dibuja las bandas de un perfil sobre el eje de frecuencia (0–45 Hz)."""
    fig, ax = plt.subplots(figsize=(9, 1.9))
    bands = profile.bands_as_dict()
    for name, (lo, hi) in bands.items():
        ax.axvspan(lo, hi, color=BAND_COLORS.get(name, "gray"), alpha=0.55)
        ax.text((lo + hi) / 2, 0.5, name, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    # Línea de referencia (perfil estándar) para comparar desplazamientos.
    if reference is not None:
        for _, (lo, hi) in reference.bands_as_dict().items():
            ax.axvline(lo, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_title(profile.label, fontsize=10)
    fig.tight_layout()
    return fig


def metric_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------
# PÁGINA 1 — Inicio / Valor
# --------------------------------------------------------------------------
def page_inicio():
    st.title("Visión general del proyecto")
    st.caption(C.APP_SUBTITLE)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(C.VALUE_SOCIAL)
    with col2:
        st.markdown(C.VALUE_MEDICAL)
    with col3:
        st.markdown(C.VALUE_SCIENTIFIC)

    st.divider()
    res = DL.load_results()
    if res is not None:
        best = DL.best_combination(res, "pr_auc_mean")
        st.subheader("Resultado principal del estudio")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mejor modelo",
                  DL.MODEL_LABELS.get(best["model"], best["model"]))
        m2.metric("PR-AUC (métrica clave)", f"{best['pr_auc_mean']:.3f}")
        m3.metric("AUC-ROC", f"{best['auc_roc_mean']:.3f}")
        m4.metric("Ventanas analizadas", f"{int(best['n_rows']):,}")
        st.caption("Detalle e interpretación en la sección **Resultados**.")
    else:
        st.info("Aún no hay resultados generados. Ve a **Configurar experimento** "
                "para lanzar una corrida, o consulta la sección **Resultados** "
                "cuando termine.")

    st.markdown(C.CLINICAL_DISCLAIMER)


# --------------------------------------------------------------------------
# PÁGINA — Guía de uso
# --------------------------------------------------------------------------
def page_guia():
    st.title("Guía de uso")
    st.markdown(C.USAGE_OVERVIEW)

    cols = st.columns(len(C.USAGE_STEPS))
    for col, (num, titulo, desc) in zip(cols, C.USAGE_STEPS):
        with col:
            st.markdown(
                f"<div style='display:inline-block;width:30px;height:30px;"
                f"line-height:30px;text-align:center;border-radius:50%;"
                f"background:{theme.PETROL};color:#fff;font-weight:600;'>{num}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{titulo}**")
            st.caption(desc)

    st.divider()
    st.markdown(C.USAGE_DATA_LOADING)
    st.divider()
    st.markdown(C.USAGE_ADD_POPULATION)
    st.divider()
    st.subheader("Mapa de las secciones")
    st.markdown(
        "- **Población** — explora la cohorte (no ejecuta nada).\n"
        "- **Bandas de frecuencia** — entiende y elige el perfil de frecuencias.\n"
        "- **Configurar experimento** — arma y lanza una corrida.\n"
        "- **Resultados** — lee e interpreta lo ya ejecutado."
    )


# --------------------------------------------------------------------------
# PÁGINA 2 — Población de estudio
# --------------------------------------------------------------------------
def page_poblacion():
    st.title("Población de estudio")
    with st.expander("Cómo usar esta sección"):
        st.markdown(C.HELP_POBLACION)
    st.markdown(C.POPULATION_INTRO)

    st.subheader("Explorador de la cohorte")
    c1, c2, c3 = st.columns(3)
    with c1:
        sexes = st.multiselect("Sexo", ["F", "M"], default=["F", "M"])
    with c2:
        edad = st.slider("Rango de edad (años)", 0, 24, (6, 10))
    with c3:
        grupos = st.multiselect(
            "Grupo", sorted(SUBJECT_DF["grupo"].unique()),
            default=sorted(SUBJECT_DF["grupo"].unique()),
        )

    mask = (
        SUBJECT_DF["sexo"].isin(sexes)
        & SUBJECT_DF["edad"].between(edad[0], edad[1])
        & SUBJECT_DF["grupo"].isin(grupos)
    )
    filtered = SUBJECT_DF[mask].sort_values("edad")

    k1, k2, k3 = st.columns(3)
    k1.metric("Sujetos seleccionados", len(filtered))
    k2.metric("Edad media", f"{filtered['edad'].mean():.1f}" if len(filtered) else "—")
    k3.metric("% mujeres",
              metric_pct((filtered["sexo"] == "F").mean()) if len(filtered) else "—")

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.dataframe(filtered, width="stretch", hide_index=True)
    with cc2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        all_ages = SUBJECT_DF["edad"]
        ax.hist(all_ages, bins=range(0, 26, 1), color=COHORT_GRAY,
                edgecolor="white", label="Cohorte completa")
        if len(filtered):
            ax.hist(filtered["edad"], bins=range(0, 26, 1), color=ACCENT,
                    edgecolor="white", label="Selección")
        ax.axvspan(6, 10, color=SHADE, alpha=0.12)
        ax.set_xlabel("Edad (años)")
        ax.set_ylabel("Nº de sujetos")
        ax.set_title("Distribución por edad (franja 6–10 sombreada)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(C.POPULATION_SCALABILITY)


# --------------------------------------------------------------------------
# PÁGINA 3 — Bandas de frecuencia (Fase 2)
# --------------------------------------------------------------------------
def page_bandas():
    st.title("Bandas de frecuencia adaptables")
    with st.expander("Cómo usar esta sección"):
        st.markdown(C.HELP_BANDAS)
    st.markdown(C.BANDS_INTRO)

    cols = st.columns(len(C.BANDS_GLOSSARY))
    for col, (key, (nombre, rango, desc)) in zip(cols, C.BANDS_GLOSSARY.items()):
        with col:
            st.markdown(f"**{nombre}**")
            st.caption(rango)
            st.write(desc)

    st.divider()
    st.markdown(C.BANDS_WHY_DEMOGRAPHIC)

    st.subheader("Perfiles disponibles")
    profiles = list_profiles()
    default = get_default_profile()
    names = [p.name for p in profiles]
    labels = {p.name: p.label for p in profiles}

    selected = st.selectbox(
        "Selecciona un perfil para inspeccionarlo",
        names, format_func=lambda n: labels[n],
    )
    prof = get_profile(selected)

    st.pyplot(plot_band_profile(prof, reference=default))

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**Justificación clínica.** {prof.rationale}")
        st.caption(f"Referencia: {prof.reference}")
    with c2:
        st.markdown("**Aplica a**")
        st.write(f"Edad: {prof.age_min:.0f}–{prof.age_max:.0f} años")
        st.write(f"Sexo: {'cualquiera' if prof.sex == 'any' else prof.sex}")
        if prof.is_thesis_baseline:
            st.success("Línea base reproducible de la tesis")

    st.divider()
    st.subheader("Sugerencia automática por edad")
    edad_demo = st.slider("Edad del paciente (años)", 0, 90, 7)
    sug = suggest_profile(age=edad_demo)
    st.info(f"Para **{edad_demo} años** se sugiere el perfil "
            f"**{sug.label}** (`{sug.name}`).")

    st.markdown(C.BANDS_REPRODUCIBILITY_NOTE)


# --------------------------------------------------------------------------
# PÁGINA 4 — Configurar y ejecutar
# --------------------------------------------------------------------------
def page_configurar():
    st.title("Configurar experimento")
    with st.expander("Cómo usar esta sección"):
        st.markdown(C.HELP_CONFIGURAR)
    st.write("Arma una corrida combinando **población** + **perfil de bandas** + "
             "**parámetros**. La separación es lo que hace al pipeline escalable.")

    st.subheader("1. Población")
    c1, c2, c3 = st.columns(3)
    with c1:
        sexes = st.multiselect("Sexo", ["F", "M"], default=["F", "M"])
    with c2:
        edad = st.slider("Edad (años)", 0, 24, (6, 10))
    with c3:
        grupos = st.multiselect(
            "Grupo", sorted(SUBJECT_DF["grupo"].unique()),
            default=["pediatric"],
        )

    subjects = sorted(SUBJECT_DF[
        SUBJECT_DF["sexo"].isin(sexes)
        & SUBJECT_DF["edad"].between(edad[0], edad[1])
        & SUBJECT_DF["grupo"].isin(grupos)
    ]["sujeto"].tolist())
    st.write(f"**{len(subjects)} sujetos:** {', '.join(subjects) if subjects else '—'}")

    st.subheader("2. Perfil de bandas")
    profiles = list_profiles()
    labels = {p.name: p.label for p in profiles}
    edad_media = SUBJECT_DF[SUBJECT_DF["sujeto"].isin(subjects)]["edad"].mean() if subjects else None
    sugerido = suggest_profile(age=edad_media).name if edad_media is not None else get_default_profile().name
    band_profile = st.selectbox(
        "Perfil", [p.name for p in profiles],
        index=[p.name for p in profiles].index(sugerido),
        format_func=lambda n: labels[n],
    )
    st.caption(f"Sugerido para la población seleccionada: **{labels[sugerido]}**")

    st.subheader("3. Parámetros del experimento")
    c1, c2, c3 = st.columns(3)
    window_sec = c1.number_input("Ventana (s)", 1.0, 30.0, 5.0, 0.5)
    overlap = c2.slider("Solape", 0.0, 0.9, 0.5, 0.1)
    n_splits = c3.number_input("Folds (GroupKFold)", 2, 10, 7, 1)

    quick = st.checkbox(
        "Modo rápido de prueba (menos sujetos y folds — para validar el flujo, "
        "no para resultados de tesis)", value=False,
    )

    st.divider()
    n_sub = min(3, len(subjects)) if quick else len(subjects)
    eff_subjects = subjects[:n_sub]
    eff_splits = 2 if quick else int(n_splits)

    est_min = max(1, len(eff_subjects)) * (4 if quick else 20)
    st.info(
        f"Una corrida completa son **16 combinaciones × {eff_splits} folds** "
        f"sobre {len(eff_subjects)} sujetos. Tiempo estimado aproximado: "
        f"**~{est_min}–{est_min*3} min**. Se ejecuta en segundo plano; puede "
        f"seguir su avance más abajo."
    )

    if st.button("Lanzar experimento en segundo plano", type="primary",
                 disabled=len(eff_subjects) < 2):
        if len(eff_subjects) < 2:
            st.error("Se necesitan al menos 2 sujetos para la validación por grupos.")
        else:
            run_name = f"ui_{band_profile}_{'quick' if quick else 'full'}"
            req = R.RunRequest(
                subjects=eff_subjects,
                band_profile=band_profile,
                out_dir=str(R.REPO_ROOT / "out_interface" / run_name),
                run_name=run_name,
                window_sec=float(window_sec),
                overlap=float(overlap),
                n_splits=eff_splits,
            )
            launched = R.launch(req)
            st.session_state["last_run"] = {
                "run_name": launched.run_name,
                "pid": launched.pid,
                "log_path": str(launched.log_path),
                "started_at": launched.started_at,
            }
            st.success(f"Lanzado (PID {launched.pid}). Log: {launched.log_path.name}")

    if "last_run" in st.session_state:
        st.subheader("Última corrida lanzada")
        lr = st.session_state["last_run"]
        running = R.is_running(lr["pid"])
        estado = "en ejecución" if running else "finalizada / detenida"
        st.write(f"**{lr['run_name']}** · PID {lr['pid']} · {estado}")
        if st.button("Refrescar log"):
            st.rerun()
        st.code(R.read_log_tail(Path(lr["log_path"])), language="text")


# --------------------------------------------------------------------------
# PÁGINA 5 — Resultados
# --------------------------------------------------------------------------
def page_resultados():
    st.title("Resultados e interpretación")
    with st.expander("Cómo usar esta sección"):
        st.markdown(C.HELP_RESULTADOS)
    res = DL.load_results()
    if res is None:
        st.info("Todavía no hay `results.csv`. Lanza una corrida en "
                "**Configurar experimento**.")
        return

    res = res.copy()
    res["modelo"] = res["model"].map(DL.MODEL_LABELS).fillna(res["model"])
    res["características"] = res["feature_set"].map(DL.FEATURESET_LABELS).fillna(res["feature_set"])

    best = DL.best_combination(res, "pr_auc_mean")
    st.subheader("Mejor combinación")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modelo", DL.MODEL_LABELS.get(best["model"], best["model"]))
    m2.metric("PR-AUC", f"{best['pr_auc_mean']:.3f}", help="Métrica principal")
    m3.metric("Sensibilidad", metric_pct(best["sensitivity_mean"]))
    m4.metric("Especificidad", metric_pct(best["specificity_mean"]))
    st.caption(f"Conjunto de características: "
               f"{DL.FEATURESET_LABELS.get(best['feature_set'], best['feature_set'])}")

    st.divider()
    st.subheader("Comparación de modelos (PR-AUC por conjunto de características)")
    pivot = res.pivot_table(index="características", columns="modelo",
                            values="pr_auc_mean", aggfunc="mean")
    st.bar_chart(pivot)

    st.subheader("Tabla completa (16 combinaciones)")
    show_cols = ["modelo", "características", "pr_auc_mean", "pr_auc_std",
                 "auc_roc_mean", "sensitivity_mean", "specificity_mean"]
    nice = res[show_cols].rename(columns={
        "pr_auc_mean": "PR-AUC", "pr_auc_std": "PR-AUC σ",
        "auc_roc_mean": "AUC-ROC", "sensitivity_mean": "Sensibilidad",
        "specificity_mean": "Especificidad",
    }).sort_values("PR-AUC", ascending=False)
    st.dataframe(nice, width="stretch", hide_index=True,
                 column_config={
                     "Sensibilidad": st.column_config.NumberColumn(format="%.2f"),
                     "Especificidad": st.column_config.NumberColumn(format="%.2f"),
                 })

    st.divider()
    st.subheader("Cómo leer estas métricas")
    for nombre, (titulo, desc) in C.METRICS_GUIDE.items():
        with st.expander(f"{nombre} — {titulo}"):
            st.write(desc)

    # Reproducibilidad: qué perfil de bandas se usó.
    manifest = DL.load_manifest()
    if manifest:
        st.divider()
        st.subheader("Reproducibilidad de este resultado")
        bp = manifest.get("band_profile")
        cc1, cc2 = st.columns(2)
        with cc1:
            if bp:
                st.write(f"**Perfil de bandas:** {bp.get('label', bp.get('name'))}")
                if bp.get("reference"):
                    st.caption(f"Referencia: {bp['reference']}")
            else:
                st.write("**Perfil de bandas:** estándar (línea base)")
            ds = manifest.get("dataset", {})
            if ds:
                st.write(f"**Ventanas:** {ds.get('n_rows', '—')} · "
                         f"**Sujetos:** {ds.get('n_subjects', '—')}")
        with cc2:
            cv = manifest.get("cv", {})
            st.write(f"**Validación:** {cv.get('strategy', '—')} "
                     f"({cv.get('n_splits', '—')} folds)")
            st.write(f"**Creado:** {manifest.get('created_at', '—')}")

    figs = DL.list_report_figures()
    if figs:
        st.divider()
        st.subheader("Figuras del reporte (PDF)")
        for f in figs:
            with open(f, "rb") as fh:
                st.download_button(f.stem, fh.read(),
                                   file_name=f.name, mime="application/pdf",
                                   key=f.name)


# --------------------------------------------------------------------------
# Navegación
# --------------------------------------------------------------------------
PAGES = {
    "Inicio": page_inicio,
    "Guía de uso": page_guia,
    "Población de estudio": page_poblacion,
    "Bandas de frecuencia": page_bandas,
    "Configurar experimento": page_configurar,
    "Resultados": page_resultados,
}


def main():
    theme.inject_style()

    st.sidebar.markdown("### Apoyo Diagnóstico EEG")
    st.sidebar.caption("Pediatría · Detección asistida de crisis")
    st.sidebar.divider()
    choice = st.sidebar.radio("Secciones", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        "Universidad de La Salle\n\n"
        "Maestría en Inteligencia Artificial\n\n"
        "Dataset CHB-MIT · Población 6–10 años"
    )

    theme.render_header()
    PAGES[choice]()
    theme.render_footer()


if __name__ == "__main__":
    main()
