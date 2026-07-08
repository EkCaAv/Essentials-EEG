# Interfaz explicativa EEG Pediátrico

Interfaz web local que permite **configurar la población y el perfil de bandas**,
**lanzar experimentos** y **explicar los resultados** con su valor científico,
médico y social. Construida sobre el pipeline reproducible existente, sin
duplicar lógica.

## Cómo ejecutar

```powershell
# desde la raíz del repo
.\run_interface.ps1
```

o directamente:

```powershell
py -3 -m streamlit run interface/app.py
```

Se abre en `http://localhost:8501`.

## Qué incluye (Fases 0, 1 y 2)

| Fase | Componente | Archivos |
|------|------------|----------|
| **0** | Config declarativo único de bandas (fuente de verdad) | [config/band_profiles.yaml](../config/band_profiles.yaml), [config/profiles.py](../config/profiles.py) |
| **2** | Perfiles de banda por demografía, documentados con procedencia | mismo `band_profiles.yaml` (perfiles `standard_adult`, `pediatric_6_10`, …) |
| **1** | Interfaz explicativa de 5 secciones | [interface/](.) |

## Secciones de la app

1. **Inicio** — valor social, médico y científico + resultado principal.
2. **Guía de uso** — flujo paso a paso, **cómo se cargan los datos** (formato CHB-MIT) y cómo añadir una población nueva.
3. **Población de estudio** — explorador de la cohorte CHB-MIT filtrable por edad/sexo/grupo.
4. **Bandas adaptables** — glosario clínico de bandas, perfiles por edad y sugerencia automática.
5. **Configurar experimento** — arma población + perfil + parámetros y lanza la corrida en segundo plano.
6. **Resultados** — tabla de 16 combinaciones, mejor modelo, guía de métricas y trazabilidad (qué perfil de bandas se usó).

Además, cada sección operativa incluye un desplegable **"ℹ️ Cómo usar esta sección"**
con instrucciones de sus controles.

## Cómo se cargan los datos

La app lee los registros desde `data/`, una carpeta por sujeto en formato CHB-MIT:

```
data/chb05/
├── chb05_01.edf          # señal EEG
├── ...
└── chb05-summary.txt     # anotaciones de crisis (etiquetas)
```

No se suben por el navegador (los EEG son grandes y son datos clínicos). Para
añadir una población: coloca sus `.edf` + `-summary.txt` en `data/<id>/`, registra
el sujeto en `config/subject_metadata.py` y, si aplica, añade un perfil en
`config/band_profiles.yaml`. La sección **Guía de uso** lo explica dentro de la app.

## Reproducibilidad

El perfil `standard_adult` reproduce **exactamente** las bandas de los experimentos
formales de la tesis. Cambiar de perfil queda registrado en `run_manifest.json`,
de modo que ajustar las bandas por demografía **nunca rompe la trazabilidad**.

## Arquitectura interna

```
interface/
├── app.py          Entrada Streamlit (navegación + 5 páginas)
├── content.py      Textos explicativos (separados de la lógica)
├── data_loader.py  Lectura de artefactos existentes (results.csv, manifest, figuras)
├── runner.py       Lanzamiento del pipeline como proceso en segundo plano
└── _runs/          Logs de corridas lanzadas desde la UI (gitignored)
```

## Escalabilidad a otras poblaciones

El diseño separa *quiénes* (población filtrable) de *cómo* (perfil de bandas +
parámetros). Para una población nueva basta añadir un perfil en `band_profiles.yaml`
y ampliar `SUBJECTS_DB`. La ingesta de datasets con otro montaje/anotaciones
(Fase 3) queda como trabajo futuro.
