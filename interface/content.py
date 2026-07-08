# interface/content.py
"""
Textos explicativos de la interfaz, separados de la lógica.

Objetivo de la interfaz (Fase 1): ser EXPLICATIVA y comunicar el valor
científico, médico y social del trabajo a una audiencia mixta (jurado de tesis,
clínicos, público general), no solo ejecutar el pipeline.
"""

APP_TITLE = "EEG Pediátrico · Detección de Crisis Epilépticas"
APP_SUBTITLE = (
    "Plataforma reproducible de Machine Learning para apoyar el diagnóstico "
    "de epilepsia en población infantil"
)

# --------------------------------------------------------------------------
# VALOR: por qué importa este trabajo
# --------------------------------------------------------------------------
VALUE_SOCIAL = """
### Aporte al bien social

La epilepsia afecta a cerca de **50 millones de personas** en el mundo, y es una
de las enfermedades neurológicas más frecuentes en la **infancia**. Cerca del
**80 % de los casos** ocurre en países de ingresos bajos y medios, donde el
acceso a un neurólogo o a un electroencefalografista experto es limitado.

En esos contextos, un mismo registro de EEG puede tardar **días o semanas** en
ser interpretado. Una herramienta que **preseleccione automáticamente** los
segmentos sospechosos de crisis permite que el especialista —escaso— concentre
su tiempo donde realmente importa, **acortando la brecha diagnóstica** y llevando
tecnología de tamizaje a lugares donde hoy no existe.
"""

VALUE_MEDICAL = """
### Valor médico

El diagnóstico de epilepsia se apoya en la **lectura visual** del EEG por un
especialista: un proceso lento, dependiente de la experiencia y sujeto a
variabilidad entre observadores.

Este sistema **no reemplaza al médico**: actúa como un **segundo lector
automático** que marca los tramos de señal con mayor probabilidad de contener
una crisis. Está diseñado priorizando la **especificidad** (no inundar al
clínico de falsas alarmas) sin perder de vista la **sensibilidad** (no dejar
pasar eventos reales) — el equilibrio clínico que hace utilizable a una
herramienta de apoyo a la decisión.
"""

VALUE_SCIENTIFIC = """
### Rigor científico

El proyecto no busca el "mejor número" sino un resultado **reproducible y
trazable** (Objetivo 3 de la tesis). Por eso:

- **Validación honesta:** se separa por sujeto (*GroupKFold*), de modo que el
  modelo nunca se evalúa con datos del mismo niño con que se entrenó. Esto evita
  el inflado de métricas más común en la literatura de EEG.
- **Diseño de experimentos formal (DOE):** 4 algoritmos × 4 conjuntos de
  características × 7 particiones = 112 entrenamientos, con pruebas estadísticas
  (Wilcoxon) para decidir si las diferencias son reales o azar.
- **Procedencia documentada:** cada corrida registra exactamente qué datos,
  bandas de frecuencia e hiperparámetros usó (`run_manifest.json`).
"""

# --------------------------------------------------------------------------
# POBLACIÓN
# --------------------------------------------------------------------------
POPULATION_INTRO = """
La población de estudio proviene del dataset público **CHB-MIT Scalp EEG**
(Children's Hospital Boston – MIT), registros de EEG de cuero cabelludo de
pacientes pediátricos con crisis epilépticas refractarias.

La tesis se centra en el subgrupo de **niños de 6 a 10 años**, una franja donde
el cerebro aún está madurando y donde —como verás en la sección de bandas— la
actividad eléctrica normal **difiere de la del adulto**. Trabajar con una
población homogénea evita mezclar fisiologías distintas y hace los resultados
más interpretables.
"""

POPULATION_SCALABILITY = """
> **Escalabilidad.** El diseño separa *quiénes* (la población, filtrable por edad
> y sexo) de *cómo* (el perfil de bandas y los parámetros). Esa separación es lo
> que permitirá, a futuro, aplicar el mismo pipeline a **otras poblaciones**
> simplemente cambiando el filtro demográfico y el perfil de bandas, sin tocar
> el código del experimento.
"""

# --------------------------------------------------------------------------
# BANDAS DE FRECUENCIA — explicación clínica
# --------------------------------------------------------------------------
BANDS_INTRO = """
El EEG mide la actividad eléctrica del cerebro como una mezcla de **ritmos** a
distintas frecuencias. Por convención clínica esa actividad se divide en
**bandas**, y cada una se asocia a estados y procesos cerebrales:
"""

BANDS_GLOSSARY = {
    "delta": ("Delta (δ)", "0.5–4 Hz",
              "Sueño profundo. En vigilia, su exceso focal puede indicar disfunción."),
    "theta": ("Theta (θ)", "4–8 Hz",
              "Somnolencia y, en niños, actividad normal abundante. Clave en pediatría."),
    "alpha": ("Alpha (α)", "8–13 Hz",
              "Reposo con ojos cerrados. Su frecuencia exacta **madura con la edad**."),
    "beta":  ("Beta (β)", "13–30 Hz",
              "Estado de alerta y concentración activa."),
    "gamma": ("Gamma (γ)", "30–45 Hz",
              "Procesamiento cognitivo de alto nivel; sensible a artefactos."),
}

BANDS_WHY_DEMOGRAPHIC = """
### ¿Por qué adaptar las bandas por edad?

El **ritmo posterior dominante** (la frecuencia de base del cerebro en reposo)
**no es fija**: en un niño de 6 años ronda los 8–9 Hz y va subiendo hasta
alcanzar ~10 Hz hacia los 10 años. Si usamos los límites del adulto en un niño,
parte de su actividad *alpha* normal se "cuela" en la banda *theta* y
distorsiona las características que alimentan al modelo.

Por eso esta plataforma ofrece **perfiles de banda documentados por demografía**.
Cada perfil declara sus límites, su justificación clínica y su referencia
bibliográfica — y la corrida registra cuál se usó, de modo que **ajustar las
bandas nunca rompe la reproducibilidad**: queda explícito en el manifiesto.
"""

BANDS_REPRODUCIBILITY_NOTE = """
**Nota de rigor.** Cambiar las bandas cambia las características extraídas, así
que resultados obtenidos con perfiles distintos **no son directamente
comparables**. El perfil `standard_adult` reproduce exactamente las bandas de
los experimentos formales de la tesis y sirve como línea base de comparación.
"""

# --------------------------------------------------------------------------
# MÉTRICAS — cómo leerlas
# --------------------------------------------------------------------------
METRICS_GUIDE = {
    "PR-AUC": (
        "Área bajo la curva Precisión-Recall",
        "La métrica **principal** aquí. Con clases muy desbalanceadas (las crisis "
        "son <2 % de las ventanas), resume mejor que el accuracy la capacidad real "
        "de detectar el evento raro. El criterio de éxito del DOE es PR-AUC > 0.15.",
    ),
    "AUC-ROC": (
        "Área bajo la curva ROC",
        "Capacidad global de separar crisis de no-crisis. 0.5 = azar, 1.0 = "
        "perfecto. Útil pero optimista con clases desbalanceadas.",
    ),
    "Sensibilidad": (
        "Recall / Tasa de verdaderos positivos",
        "De todas las crisis reales, ¿cuántas detecta? Una sensibilidad baja "
        "significa crisis que pasan desapercibidas — clínicamente costoso.",
    ),
    "Especificidad": (
        "Tasa de verdaderos negativos",
        "De todo lo que NO es crisis, ¿cuánto clasifica bien? Una especificidad "
        "alta evita saturar al clínico con falsas alarmas.",
    ),
}

CLINICAL_DISCLAIMER = """
---
**Aviso.** Esta es una herramienta de **investigación y apoyo a la decisión**.
No es un dispositivo médico certificado y no debe usarse como única base para un
diagnóstico clínico.
"""

# ==========================================================================
# GUÍA DE USO — cómo operar cada parte de la plataforma
# ==========================================================================

USAGE_OVERVIEW = """
Esta plataforma convierte registros de EEG en bouquet de **experimentos de
detección de crisis**, explicando cada paso. El flujo completo es:
"""

# Pasos del flujo (número, título, descripción).
USAGE_STEPS = [
    ("1", "Cargar datos",
     "Se colocan los registros EEG en la carpeta `data/`. La app los detecta "
     "automáticamente."),
    ("2", "Elegir población",
     "Filtras los sujetos por edad, sexo y grupo en la sección *Población* o "
     "directamente al *Configurar experimento*."),
    ("3", "Configurar bandas",
     "Eliges un perfil de bandas adecuado a la edad (o usas el sugerido "
     "automáticamente)."),
    ("4", "Ejecutar",
     "Lanzas la corrida; se ejecuta en segundo plano (no bloquea la app)."),
    ("5", "Interpretar",
     "Cuando termina, la sección *Resultados* muestra y explica las métricas."),
]

# --------------------------------------------------------------------------
# CÓMO SE CARGAN LOS DATOS
# --------------------------------------------------------------------------
USAGE_DATA_LOADING = """
### Cómo se cargan los datos

La plataforma **lee los registros desde la carpeta `data/`** del proyecto. No se
suben por el navegador: los archivos de EEG son grandes y, al ser datos clínicos,
conviene mantenerlos en el servidor y no transferirlos por la red.

**Estructura esperada** (formato CHB-MIT), una carpeta por sujeto:

```
data/
├── chb05/
│   ├── chb05_01.edf          ← registro EEG (señal)
│   ├── chb05_02.edf
│   ├── ...
│   └── chb05-summary.txt      ← anotaciones de crisis (inicio/fin en segundos)
├── chb09/
│   ├── chb09_01.edf
│   └── chb09-summary.txt
└── ...
```

| Archivo | Qué contiene | Por qué importa |
|---------|--------------|-----------------|
| `*.edf` | La señal EEG multicanal (European Data Format) | Es la materia prima de la que se extraen las características |
| `*-summary.txt` | Qué archivos tienen crisis y en qué segundos empiezan/terminan | Son las **etiquetas**: sin ellas no se puede *entrenar* (aprendizaje supervisado) |

> El `.edf` se lee con la librería **MNE**; el `-summary.txt` se interpreta para
> marcar qué ventanas de 5 s son "crisis" y cuáles "no-crisis".
"""

USAGE_ADD_POPULATION = """
### Cómo añadir una población nueva (escalar)

1. **Coloca los datos.** Crea `data/<id>/` con sus `.edf` y un `<id>-summary.txt`
   en formato CHB-MIT.
2. **Registra el sujeto.** Añade una entrada en `config/subject_metadata.py`
   (`SUBJECTS_DB`) con `age`, `gender` y `group`. Así aparece en los filtros.
3. **(Opcional) Crea un perfil de banda.** Si la nueva población tiene una
   fisiología distinta, añade un perfil en `config/band_profiles.yaml` con su
   justificación y referencia.
4. **Listo.** El nuevo sujeto aparece automáticamente en *Población* y
   *Configurar experimento*.

> **Nota.** Si los datos vienen en **otro formato** (montaje distinto, otros
> nombres de canal, anotaciones en otro estándar), se requiere un *adaptador de
> ingesta* (Fase 3, trabajo futuro). El formato CHB-MIT funciona sin cambios.
"""

# --------------------------------------------------------------------------
# AYUDA CONTEXTUAL POR SECCIÓN (se muestra en un desplegable en cada página)
# --------------------------------------------------------------------------
HELP_POBLACION = """
**Para qué sirve esta sección:** explorar quiénes componen la cohorte antes de
experimentar.

**Cómo usar los controles:**
- **Sexo / Edad / Grupo:** filtran la lista de sujetos. La tabla y el histograma
  se actualizan al instante.
- La franja **6–10 años** aparece sombreada porque es la población de la tesis.
- Los sujetos seleccionados aquí son los candidatos que luego puedes llevar a
  *Configurar experimento*.

*Nota:* esta sección no ejecuta nada; solo te ayuda a entender la cohorte.
"""

HELP_BANDAS = """
**Para qué sirve esta sección:** entender y elegir cómo se divide el espectro de
frecuencias antes de extraer características.

**Cómo usar los controles:**
- **Selector de perfil:** muestra los límites de cada banda, su justificación
  clínica y su referencia. Las líneas punteadas marcan el perfil estándar para
  que veas cuánto se desplazan las bandas.
- **Sugerencia por edad:** mueve el deslizador y la app recomienda el perfil más
  apropiado para esa edad.

**Qué cambia al cambiar el perfil:** se redefinen los límites de δ/θ/α/β/γ, lo
que cambia las características de *potencia por banda* que recibe el modelo. Por
eso resultados con perfiles distintos no son directamente comparables.
"""

HELP_CONFIGURAR = """
**Para qué sirve esta sección:** armar y lanzar un experimento concreto.

**Cómo usar los controles (en orden):**
1. **Población:** filtra los sujetos (sexo, edad, grupo). Abajo ves cuántos y
   cuáles quedaron seleccionados.
2. **Perfil de bandas:** elige uno; por defecto se preselecciona el sugerido
   para la edad media de tu selección.
3. **Parámetros:** ventana (segundos por fragmento), solape entre ventanas y
   número de *folds* de validación.
4. **Modo rápido (opcional):** usa pocos sujetos y 2 folds para validar que el
   flujo funciona en minutos — **no** para resultados de tesis.
5. **Lanzar:** la corrida arranca en segundo plano. Sigue su avance en el log de
   abajo (botón *Refrescar log*); puedes cambiar de sección sin detenerla.

Los resultados aparecerán en `out_interface/` y, al terminar, en *Resultados*.
"""

HELP_RESULTADOS = """
**Para qué sirve esta sección:** leer e interpretar los experimentos ya
ejecutados.

**Cómo leerla:**
- **Mejor combinación:** el modelo + conjunto de características con mayor PR-AUC.
- **Gráfico de barras:** compara PR-AUC entre modelos y conjuntos de
  características.
- **Tabla completa:** las 16 combinaciones ordenadas; pasa el cursor por los
  encabezados para ver ayudas.
- **Cómo leer las métricas:** despliega cada métrica para su explicación clínica.
- **Reproducibilidad:** muestra qué perfil de bandas y validación se usaron (del
  `run_manifest.json`).
"""
