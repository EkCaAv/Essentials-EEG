# Diseño de Experimentos (DoE)
## Detección de Crisis Epilépticas en EEG Pediátrico — Comparación de Algoritmos ML

**Tesis**: Automatización de la detección de crisis en epilepsia focal pediátrica mediante IA  
**Institución**: Universidad de La Salle — Maestría en Inteligencia Artificial  
**Línea de investigación**: Machine Learning  
**Dataset**: CHB-MIT Scalp EEG Database (Pediatric Epilepsy)

---

## 1. Delimitación del Problema y Objeto de Estudio

**Problema**: La detección manual de crisis epilépticas en registros de EEG pediátrico es costosa en tiempo y requiere especialización clínica. Existe la necesidad de sistemas automáticos que puedan identificar segmentos ictales con alta sensibilidad y baja tasa de falsos positivos.

**Objeto de estudio**: Comparación sistemática de algoritmos de Machine Learning clásico para clasificación binaria (ictal vs. interictal) sobre características espectrales y estadísticas extraídas de señales EEG.

**Restricciones del dominio**:
- Desbalance severo: ~1.24% de ventanas positivas (ictales)
- Heterogeneidad inter-sujeto: cada paciente tiene un patrón ictal distinto
- Validación por sujeto obligatoria: impedir fuga de información entre pacientes

**Alcance**: Subcohorte pediátrica de 7 sujetos (6–10 años): chb05, chb09, chb14, chb16, chb20, chb22, chb23.

---

## 2. Variables de Respuesta (Métricas de Evaluación)

### Variable primaria
| Métrica | Justificación |
|---|---|
| **PR-AUC** (Average Precision) | Robusta ante desbalance severo; captura la relación precisión-sensibilidad en la clase minoritaria (ictal). Es la métrica de decisión. |

### Variables secundarias
| Métrica | Rol |
|---|---|
| AUC-ROC | Discriminación global; comparación con literatura existente |
| Sensibilidad (Recall ictal) | Fracción de crisis detectadas; crítica en contexto clínico |
| Especificidad | Fracción de segmentos no-ictal clasificados correctamente |
| Accuracy | Referencia; **no usarse como criterio principal** por el desbalance |

**Criterio de éxito mínimo**: PR-AUC > 0.15 en promedio sobre los 7 folds (baseline cap. 7: 0.112 ± 0.149).

---

## 3. Factores y Niveles Experimentales

### Factor A — Algoritmo de ML (Factor de tratamiento)
| Nivel | Algoritmo | Hiperparámetros clave |
|---|---|---|
| A1 | Logistic Regression | C=1.0, penalty=l2, class_weight=balanced |
| A2 | Random Forest | n_estimators=400, class_weight=balanced_subsample |
| A3 | SVM (RBF) | C=10, gamma=scale, class_weight=balanced |
| A4 | Gradient Boosting | n_estimators=300, lr=0.05, max_depth=4 |

**Justificación niveles**: Los cuatro algoritmos cubren el espacio de hipótesis relevante para clasificación tabular:
- LogReg: modelo lineal, interpretable, baseline regulado
- RF: ensemble de árboles, maneja alta dimensionalidad, estable
- SVM: margen máximo, efectivo con datos de alta dimensión normalizados
- GradBoost: ensemble secuencial, mejor control del sobreajuste

### Factor B — Conjunto de características (Feature set)
| Nivel | Nombre | Nº features | Componentes |
|---|---|---|---|
| B1 | `bp_only` | 10 | Bandpower relativo (δ,θ,α,β,γ) × mean/std sobre canales |
| B2 | `bp_plus_rms` | 12 | B1 + RMS por canal (mean/std) |
| B3 | `bp_plus_rms_kurt` | 14 | B2 + Kurtosis por canal (mean/std) |
| B4 | `bp_plus_rms_kurt_skew` | 16 | B3 + Skewness por canal (mean/std) |

**Justificación**: Diseño incremental que permite evaluar el aporte marginal de cada tipo de feature. La hipótesis es que el RMS aporta valor de inflexión (observado en cap. 7) y que la kurtosis y skewness pueden capturar la asimetría y picos característicos de las descargas ictales.

### Factor de bloqueo — Sujeto (Blocking factor)
| Nivel | Sujeto | Nota |
|---|---|---|
| B_k1..B_k7 | chb05, chb09, chb14, chb16, chb20, chb22, chb23 | chb05 puede tener 0 positivos en un fold |

El sujeto es un **factor de ruido controlado** (blocking), no un factor de tratamiento. Se controla mediante GroupKFold por sujeto para garantizar que no haya datos del mismo paciente simultáneamente en entrenamiento y evaluación.

---

## 4. Clasificación del Diseño Experimental

### Diseño principal
**Diseño factorial completo 4 × 4 con bloques aleatorios por sujeto**  
(A × B = 4 × 4 = 16 combinaciones de tratamiento; 7 bloques = sujetos)

Este diseño pertenece a la categoría:
> **Diseño para estudiar el efecto de varios factores sobre una o más variables de respuesta**

Permite estimar:
- Efecto principal del algoritmo (Factor A)
- Efecto principal del feature set (Factor B)
- Interacción A×B (¿depende la efectividad del algoritmo del tipo de feature?)

### Clasificación formal aplicable

| Tipo de diseño | ¿Aplica? | Rol en este experimento |
|---|---|---|
| Comparación de 2+ tratamientos | ✅ | Los 4 algoritmos son los tratamientos; Wilcoxon compara pares |
| Factorial (varios factores → 1+ variables respuesta) | ✅ | **Diseño principal**: 4×4, métricas múltiples |
| Optimización de proceso (RSM) | No | No se busca superficie de respuesta continua |
| Diseño robusto (Taguchi) | Parcialmente | GroupKFold actúa como control de variabilidad inter-sujeto |
| Diseño de mezcla | No | Los features son independientes, no proporciones que sumen 1 |

### Validación cruzada como diseño de bloques
GroupKFold con n_splits=7 implementa un **diseño de bloques completos aleatorizados** donde:
- Cada bloque = un sujeto excluido en evaluación
- Cada tratamiento (combinación algoritmo×features) se evalúa en los mismos 7 bloques
- Se garantiza independencia entre folds al nivel de sujeto

---

## 5. Hipótesis Estadísticas

> **Nota epistemológica**: En estadística no existen verdades absolutas. Los enunciados de hipótesis describen lo que es *posible* o *probable* dado los datos observados. Los resultados se interpretan en términos de evidencia, no de certezas.

### Hipótesis principal (H0 conservadora)
- **H₀**: Es *posible* que no exista diferencia sistemática en PR-AUC entre ningún par de algoritmos cuando se evalúan sobre las mismas ventanas de EEG.
- **H₁**: Es *probable* que al menos un algoritmo muestre PR-AUC sistemáticamente superior a otro en la mayoría de los sujetos evaluados.

### Hipótesis por pares (Wilcoxon signed-rank, 6 comparaciones)
Para cada par (modelo_i, modelo_j):
- **H₀ᵢⱼ**: Es *posible* que las distribuciones de PR-AUC por fold de modelo_i y modelo_j sean equivalentes.
- **H₁ᵢⱼ**: Existe evidencia *probable* de que las distribuciones difieren sistemáticamente entre sujetos.

### Hipótesis sobre features
- **H₀_features**: Es *posible* que la adición de RMS, kurtosis o skewness no mejore PR-AUC respecto a bandpower puro.
- **H₁_features**: Es *probable* que el feature set `bp_plus_rms` (o superior) genere mejoras detectables en PR-AUC, dado que cap. 7 mostró que `bp_only` rinde como azar (AUC ≈ 0.466).

---

## 6. Elementos de Inferencia Estadística

### Población y Muestra
| Elemento | Descripción |
|---|---|
| **Población objetivo** | Niños con epilepsia focal de 6–10 años con registros EEG disponibles |
| **Muestra disponible** | 7 sujetos del CHB-MIT en el rango de edad (chb05, chb09, chb14, chb16, chb20, chb22, chb23) |
| **Sesgo de muestra** | Cohorte CHB-MIT: 80% femenino, distribución bimodal de edades (2–4 y 10–14 años) |
| **Representatividad** | **Limitada**: No se puede generalizar directamente; los resultados son válidos para esta cohorte |

### Pruebas de Hipótesis
**Wilcoxon Signed-Rank Test** (no paramétrico, pareado por fold):
- Condición de aplicación: ≥ 3 folds con métricas no idénticas
- Estadístico: W (suma de rangos de diferencias positivas)
- Umbral de significancia: α = 0.05
- Interpretación: p < 0.05 → *existe evidencia probable* de diferencia (no "el modelo A es mejor")

**Advertencia de potencia estadística**: Con solo 7 folds, la potencia del test Wilcoxon es baja (~0.3–0.5 para efectos medianos). Esto significa que incluso diferencias reales pueden no alcanzar significancia estadística. Los resultados no significativos deben interpretarse como *insuficiencia de evidencia*, no como *equivalencia demostrada*.

### Intervalos de Confianza (IC)
Para cada combinación (algoritmo, feature_set):
- Reportar: media ± desviación estándar (sobre 7 folds)
- IC aproximado al 95%: μ ± t_(0.025, 6) × σ/√7 ≈ μ ± 2.447 × σ/√7
- Interpretación: El verdadero PR-AUC *podría estar* en este rango con ~95% de probabilidad dadas estas observaciones

### Error Estadístico
| Tipo | Descripción | Consecuencia en este experimento |
|---|---|---|
| **Error Tipo I (α)** | Declarar diferencia cuando no existe (falso positivo) | Recomendar un modelo innecesariamente más complejo |
| **Error Tipo II (β)** | No detectar diferencia real (falso negativo) | Alta probabilidad dado el tamaño muestral (n=7 folds) |

Dado el tamaño muestral reducido, **se acepta mayor riesgo de Error Tipo II**. La interpretación se complementa con el tamaño del efecto (diferencia en medias) además del p-valor.

### Potencia Estadística
Con n=7 folds y α=0.05 (bilateral), la potencia para detectar un efecto medio (d=0.5) en Wilcoxon es ≈ 0.35. Esto implica:
- Un resultado significativo es informativo
- Un resultado no significativo **no descarta** la existencia de una diferencia real
- Se recomienda reportar tamaño del efecto (rank-biserial correlation) junto al p-valor

---

## 7. Enfoque Diferencial: Datos de Entrenamiento vs. Test

| Aspecto | Datos de Entrenamiento | Datos de Test (held-out fold) |
|---|---|---|
| **Rol** | Ajuste de pesos/parámetros del modelo | Evaluación imparcial del desempeño |
| **Sujetos incluidos** | 6 de los 7 sujetos de la cohorte | 1 sujeto excluido (el del fold actual) |
| **Riesgo de sobreajuste** | Alto si el modelo memoriza en lugar de generalizar | Detecta sobreajuste (train >> test en métricas) |
| **Desbalance** | Se maneja con `class_weight` en el modelo | Se reporta *sin* rebalanceo artificial |
| **Métricas reportadas** | No se reportan (solo para diagnóstico interno) | PR-AUC, AUC-ROC, sensibilidad, especificidad |
| **Fuga de información** | Los hiperparámetros se ajustan solo con datos de train (nested CV si --tune_hyperparams) | El fold de test nunca participa en la selección de hiperparámetros |

**Regla de oro**: Las métricas finales en `results.csv` y `cv_fold_metrics.csv` corresponden **exclusivamente** a las predicciones sobre el fold de test. Cualquier cifra de entrenamiento que aparezca en el análisis debe marcarse explícitamente para evitar confusión.

---

## 8. Plan Experimental y Organización del Trabajo

### Experimentos a ejecutar

| # | Algoritmo | Feature sets | Folds | Combinaciones |
|---|---|---|---|---|
| Run principal | LogReg, RF, SVM, GradBoost | bp_only, bp_plus_rms, bp_plus_rms_kurt, bp_plus_rms_kurt_skew | 7 (GroupKFold sujeto) | 4 × 4 = 16 |

**Total de entrenamientos**: 16 combinaciones × 7 folds = **112 entrenamientos**

### Estructura de salida organizada

```
out_thesis_final/
├── doc/
│   └── doe_experimental_design.md       ← Este documento
├── classical_all_models/                 ← --out_dir de chbmit_experiments.py
│   ├── results.csv                       ← 16 filas: media por combinación
│   ├── cv_fold_metrics.csv               ← 112 filas: métricas por fold
│   ├── windows_dataset.csv               ← ~52,244 ventanas
│   ├── run_manifest.json                 ← reproducibilidad completa
│   └── runs/
│       └── run_YYYYMMDD_HHMMSS/          ← snapshot versionado del run
│           ├── results.csv
│           ├── cv_fold_metrics.csv
│           ├── windows_dataset.csv
│           └── run_manifest.json
├── comparison_table.csv                  ← output de consolidate_results.py
└── reports/                              ← análisis estadístico post-experimento
    ├── wilcoxon_report.txt
    └── summary_figures/
```

### Comandos de ejecución (en orden)

**Paso 1 — Experimento principal**:
```powershell
python .\chbmit_experiments.py `
  --data_root ".\data" `
  --out_dir   ".\out_thesis_final\classical_all_models" `
  --subjects chb05 chb09 chb14 chb16 chb20 chb22 chb23 `
  --window_sec 5 `
  --overlap 0.5 `
  --n_splits 7 `
  --run_name "doe_v1_all_models"
```

**Paso 2 — Tabla comparativa**:
```powershell
python .\consolidate_results.py `
  --classical_dir ".\out_thesis_final\classical_all_models" `
  --out           ".\out_thesis_final\comparison_table.csv" `
  --feature_set   bp_plus_rms
```

---

## 9. Interpretación de Resultados (Guía)

1. **Rankear por PR-AUC** (variable primaria), no por accuracy ni AUC-ROC.
2. **Reportar con intervalos**: "El Random Forest *posiblemente* alcanza un PR-AUC de 0.XXX ± 0.YYY en esta cohorte".
3. **Wilcoxon**: "Existe evidencia *probable* de que RF supera a LogReg (W=XX, p=0.0YY)" — evitar "RF *es* mejor".
4. **Folds con NaN**: Si `folds_with_nan_auc` > 0, indicar que el AUC no pudo calcularse en algunos folds por ausencia de positivos; reportar por separado.
5. **Comparación con cap. 7**: La línea base es PR-AUC=0.112 con `bp_plus_rms`. Cualquier mejora > 0.02 es potencialmente relevante; <0.02 es ruido dado el σ=0.149.
6. **No inferir causalidad**: Los resultados son *asociaciones observadas* en esta cohorte; la generalización requiere validación externa.

---

*Documento generado: 2026-05-06*  
*Versión: 1.0 — pre-experimento*
