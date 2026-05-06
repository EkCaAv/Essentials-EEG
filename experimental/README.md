# Exploración Complementaria — Deep Learning

> **AVISO**: Esta carpeta contiene **exploraciones técnicas fuera del alcance principal de la tesis**.
>
> La línea de investigación aprobada es **Machine Learning**, y la propuesta de tesis se sustenta en la comparación experimental entre algoritmos clásicos (Logistic Regression, Random Forest, SVM, Gradient Boosting) sobre características tabulares extraídas del EEG (bandpower + RMS + momentos estadísticos).
>
> Los archivos en esta carpeta **no forman parte de los entregables principales** de la tesis y **no se reportan en el documento final**. Se conservan únicamente como referencia de exploración técnica complementaria que podría retomarse en trabajo futuro.

## Contenido

| Archivo | Descripción |
|---|---|
| `cnn_experiments.py` | Red convolucional 2D sobre espectrogramas STFT del EEG. |
| `lstm_experiments.py` | BiLSTM sobre señal EEG cruda multi-canal. |
| `dl_utils.py` | Infraestructura compartida (carga de EDF a tensores, métricas, manifestos). |

## Por qué están fuera del alcance

1. **Línea de investigación**: la propuesta declara explícitamente "Línea de Investigación: Machine Learning". Incorporar Deep Learning como eje principal mezclaría líneas y ampliaría el alcance más allá de lo aprobado por el comité.
2. **Aporte central de la tesis**: el valor diferencial está en (a) el pipeline reproducible, (b) la validación rigurosa por sujeto con `GroupKFold`, y (c) la comparación sistemática de combinaciones de features dentro de ML clásico.
3. **Defensibilidad académica**: los resultados de ML clásico ya son consistentes y defendibles; añadir DL implicaría justificar entrenamientos que no son centrales al objetivo.

## Si se ejecutan estos scripts

Estos scripts requieren dependencias adicionales **no incluidas en `requirements.txt`**:

```bash
pip install torch
```

Y usan como entrada el archivo `windows_dataset.csv` generado por `chbmit_experiments.py` (núcleo ML del proyecto).

## Trabajo futuro

Si en una fase posterior (post-tesis o tesis doctoral) se decide ampliar la comparación a modelos profundos, estos scripts proporcionan un punto de partida con:

- Validación `GroupKFold` por sujeto (mismo protocolo que el clásico).
- Métricas comparables (PR-AUC, AUC-ROC, sensibilidad, especificidad).
- Esquema de manifiestos (`run_manifest.json`) compatible con `consolidate_results.py`.
