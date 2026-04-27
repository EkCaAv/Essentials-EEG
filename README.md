# EEG Pediatric Seizure Detection Pipeline

Este proyecto es un pipeline de Machine Learning para el análisis de registros EEG pediátricos y la detección de crisis epilépticas focales. Está diseñado para reproducir un flujo completo de datos: adquisición, preprocesamiento, extracción de características, experimentación comparativa y generación de reportes.

## Objetivo

El objetivo es desarrollar y validar un sistema automatizado que permita detectar crisis epilépticas en EEG pediátrico usando datos públicos, con énfasis en:

- preprocesamiento de señales EEG (filtrado, segmentación, normalización);
- comparación de al menos dos técnicas de Machine Learning para clasificar ventanas ictales e interictales;
- consolidación de un pipeline reproducible que integre preprocesamiento, entrenamiento y evaluación.

## Estructura del proyecto

- `run_pipeline.py` — entrada principal del pipeline modular.
- `run_experiments.py` — ejecuta múltiples configuraciones experimentales en lote.
- `chbmit_experiments.py` — experimento enfocado en clasificación y validación con CHB-MIT.
- `config/base_config.py` — configuración base del pipeline y parámetros de procesamiento.
- `config/experiments/` — variantes experimentales específicas (`exp_pediatric`, `exp_no_gamma`, `exp_high_spike_threshold`, `exp_pli_only`, `exp_smoke_test`).
- `orchestration/pipeline_runner.py` — orquestador que procesa sujetos, carga EDF, detecta spikes, extrae features y crea reportes.
- `features/` — módulos por vertical slice: preprocesamiento, análisis de potencia, detección de espigas, conectividad y topografía.
- `analysis/` — análisis estadístico y visualizaciones para tesis y estudio pediátrico.
- `reporting/` — generación de reportes HTML y ensamblado de resultados visuales.
- `data/` — datos del dataset CHB-MIT con registros EEG y anotaciones de crisis.
- `requirements.txt` — dependencias principales del proyecto.
- `copilot-instructions.md` — guía interna para revisión del asistente de código.

## Requisitos

Instalar dependencias principales:

```bash
pip install -r requirements.txt
```

Además, algunos scripts del proyecto usan librerías adicionales que pueden necesitar instalación:

```bash
pip install scikit-learn seaborn
```

## Uso básico

### Ejecutar el pipeline por defecto

```bash
python run_pipeline.py
```

### Ejecutar una configuración experimental

```bash
python run_pipeline.py --config experiments.exp_pediatric
```

### Ejecutar el conjunto de experimentos en batch

```bash
python run_experiments.py
```

### Ejecutar un experimento de clasificación CHB-MIT

```bash
python chbmit_experiments.py --data_root "./data" --out_dir "./out_6_10" --subjects chb05 chb09 chb14 chb16 chb20 chb22 chb23 --window_sec 5 --overlap 0.5 --n_splits 7
```

Cada corrida del benchmark CHB-MIT ahora guarda:

- `results.csv` con métricas agregadas por modelo y conjunto de features;
- `cv_fold_metrics.csv` con métricas por fold, sujetos de entrenamiento/prueba y tamaños de partición;
- `run_manifest.json` con configuración, sujetos, parámetros, dataset y artefactos generados;
- una copia versionada en `out_dir/runs/<run_id>/...` para comparación entre corridas.

### Comparar dos corridas del benchmark

```bash
python compare_experiment_runs.py --current "./out_6_10/runs/run_YYYYMMDD_HHMMSS/results.csv" --baseline "./out_6_10/runs/run_YYYYMMDD_HHMMSS/results.csv"
```

### Ejecucion one-shot en TensorDock

El proyecto incluye un script para automatizar setup + baseline + corrida comparativa + comparacion final:

```bash
chmod +x run_tensordock_training.sh
./run_tensordock_training.sh
```

Variables utiles (opcionales):

- `PYTHON_BIN` (default: `python3.11`)
- `VENV_DIR` (default: `.venv311`)
- `OUT_DIR` (default: `./out_tensordock`)
- `SUBJECTS` (default: `chb05 chb09 chb14 chb16 chb20 chb22 chb23`)
- `N_SPLITS` (default: `5`)
- `BASE_OVERLAP` (default: `0.5`)
- `CURR_OVERLAP` (default: `0.25`)

### Generar reportes HTML a partir de resultados existentes

```bash
python build_html_reports.py
```

## Datos esperados

El proyecto espera la estructura de datos del dataset CHB-MIT en `data/`, por ejemplo:

- `data/chb01/chb01_03.edf`
- `data/chb01/chb01_03.edf.seizures`
- `data/chb01/chb01_03.edf_ictal0_pli_alpha.csv`

También usa archivos de summary y metadatos para asignar etiquetas de crisis.

## Enfoque del pipeline

El pipeline está organizado para:

- cargar registros EDF y anotaciones de crisis;
- preprocesar EEG con filtrado, resampling y limpieza;
- extraer features de bandas de frecuencia, spikes y conectividad;
- comparar variantes experimentales y evaluar en condiciones realistas;
- generar resultados numéricos y visuales por sujeto;
- documentar la reproducibilidad mediante configuraciones modulares y resultados exportados.

## Scripts de análisis

- `analysis/thesis_analysis.py` — análisis estadístico y figuras para tesis.
- `analysis/plot_pediatric_results.py` — visualizaciones específicas de la población pediátrica.
- `analysis/compare_combinations.py` — comparación de combinaciones de características y modelos.
- `analysis/iterative_modeling.py` — modelado iterativo sobre conjuntos de datos.

## Notas importantes

- El proyecto no es solo una red neuronal aislada: busca evidenciar calidad de datos, segmentación, construcción de features, comparación experimental y validación rigurosa.
- La separación por sujeto y la evaluación con métricas adecuadas para clases desbalanceadas son aspectos críticos.
- `copilot-instructions.md` contiene instrucciones específicas para revisión de código bajo los objetivos de tesis.

## Licencia

El proyecto no incluye una licencia explícita en el repositorio.
