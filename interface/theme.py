# interface/theme.py
"""
Estilo visual de la interfaz: sobrio, corporativo, con aire clínico.

Centraliza la paleta, el CSS y los componentes de marca (cabecera, pie) para que
toda la app tenga un aspecto consistente de herramienta de apoyo médico.
"""

from __future__ import annotations

import streamlit as st

# Paleta clínica (coherente con .streamlit/config.toml)
PETROL = "#15657A"       # azul-petróleo (acento principal)
PETROL_DARK = "#0E4A5A"  # variante oscura (cabecera)
SLATE = "#1B2A33"        # texto
MUTED = "#5B6B73"        # texto secundario
LINE = "#D7DEE2"         # bordes sutiles
SURFACE = "#EEF2F4"      # superficies

# Paleta de bandas en tono clínico (sobria, no saturada)
BAND_COLORS = {
    "delta": "#2C5F73",
    "theta": "#3E7C8C",
    "alpha": "#5A8F7B",
    "beta": "#8A7E6B",
    "gamma": "#9C6B6B",
}

_CSS = f"""
<style>
/* --- Lienzo general --- */
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
html, body, [class*="css"] {{ color: {SLATE}; }}

/* --- Jerarquía tipográfica sobria --- */
h1 {{ font-weight: 600; letter-spacing: -0.01em; color: {SLATE}; }}
h2, h3 {{ font-weight: 600; color: {SLATE}; }}
h2 {{ border-bottom: 1px solid {LINE}; padding-bottom: .3rem; }}

/* --- Cabecera clínica --- */
.clinical-header {{
    background: linear-gradient(90deg, {PETROL_DARK} 0%, {PETROL} 100%);
    color: #FFFFFF; border-radius: 8px; padding: 18px 24px;
    margin-bottom: 22px; display: flex; align-items: center; gap: 16px;
}}
.clinical-header .ch-mark {{
    font-size: 26px; line-height: 1; opacity: .9;
    border-right: 1px solid rgba(255,255,255,.35); padding-right: 16px;
}}
.clinical-header .ch-title {{ font-size: 1.15rem; font-weight: 600; }}
.clinical-header .ch-sub {{ font-size: .82rem; opacity: .85; font-weight: 400; }}

/* --- Tarjetas de métrica --- */
[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid {LINE};
    border-radius: 8px; padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(16,40,52,.04);
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}

/* --- Barra lateral --- */
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

/* --- Expanders y avisos (sobrios) --- */
[data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 8px; }}

/* --- Pie institucional --- */
.clinical-footer {{
    margin-top: 28px; padding-top: 12px; border-top: 1px solid {LINE};
    color: {MUTED}; font-size: .78rem; text-align: center;
}}

/* --- Limpieza de cromo de Streamlit --- */
[data-testid="stDecoration"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_style() -> None:
    """Inyecta el CSS global. Llamar una vez por ejecución de página."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(subtitle: str = "Detección asistida de crisis epilépticas") -> None:
    """Cabecera clínica corporativa (marca + título)."""
    st.markdown(
        f"""
        <div class="clinical-header">
          <div class="ch-mark">＋</div>
          <div>
            <div class="ch-title">Plataforma de Apoyo Diagnóstico · EEG Pediátrico</div>
            <div class="ch-sub">{subtitle} — Universidad de La Salle</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Pie institucional sobrio."""
    st.markdown(
        """
        <div class="clinical-footer">
          Herramienta de investigación y apoyo a la decisión clínica · No es un
          dispositivo médico certificado.<br>
          Maestría en Inteligencia Artificial · Universidad de La Salle · Dataset CHB-MIT
        </div>
        """,
        unsafe_allow_html=True,
    )
