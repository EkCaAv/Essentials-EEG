# copilot-instructions.md

## Contexto del proyecto

Este proyecto de maestría pertenece a la línea de investigación de Machine Learning y su objetivo es desarrollar y validar un sistema automatizado para la detección oportuna de crisis epilépticas focales en registros EEG pediátricos usando datos públicos. El enfoque principal es un pipeline completo de ML aplicado a EEG, no una implementación aislada de una red neuronal.

## Objetivos explícitos del proyecto

El asistente debe revisar el código en función de estos tres objetivos:

1. Preprocesar señales EEG mediante filtrado, segmentación y normalización.
2. Analizar comparativamente el desempeño de al menos dos técnicas de inteligencia artificial o machine learning para clasificar segmentos ictales e interictales.
3. Consolidar un pipeline reproducible que integre preprocesamiento, entrenamiento y evaluación.

## Qué debe entender el asistente de código

El proyecto es un pipeline ML aplicado a EEG pediátrico. La revisión debe centrarse en:

- Calidad del preprocesamiento de EEG.
- Lógica de segmentación en ventanas y etiquetado.
- Construcción de features o representaciones útiles.
- Comparación experimental entre al menos dos modelos.
- Validación correcta por sujeto para evitar leakage.
- Uso de métricas adecuadas para clases desbalanceadas: PR-AUC, sensibilidad, especificidad y AUC-ROC.
- Trazabilidad y reproducibilidad del pipeline completo.

## Criterios de alineación con la tesis

El código estará bien alineado si demuestra:

- un bloque robusto de adquisición y preparación de datos;
- un bloque de modelado comparativo entre técnicas;
- un bloque de entrenamiento y evaluación consistente;
- una estructura reproducible que permita repetir experimentos y comparar configuraciones.

La prioridad no es solamente maximizar una métrica, sino evidenciar progreso metodológico, comparabilidad y validez técnica bajo condiciones realistas de variabilidad inter-sujeto y desbalance de clases.

## Orientación para la revisión

Priorizar lo siguiente:

- datos bien preparados y preprocesados;
- segmentación y etiquetado correctos de ventanas EEG;
- comparación entre técnicas y configuraciones;
- validación rigurosa, preferiblemente con separación por sujeto u otra técnica que evite fuga;
- métricas adecuadas para desbalance de clases y detección de crisis;
- pipeline reproducible y documentado.

Si el código incluye modelos clásicos y/o deep learning, la revisión debe priorizar que el proyecto siga siendo defendible como trabajo de Machine Learning: enfoque en datos, validación y comparabilidad más que en la complejidad del modelo.

## Archivos relevantes

Revisar especialmente los siguientes componentes:

- `run_pipeline.py`
- `config/base_config.py`
- `config/experiments/*`
- `orchestration/pipeline_runner.py`
- `chbmit_experiments.py`
- `analysis/thesis_analysis.py`
- `features/`
- `build_html_reports.py`
