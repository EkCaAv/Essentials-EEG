# EEG Pediatric Seizure Detection — Pipeline Reproducible

Proyecto de investigación de la Maestría en Inteligencia Artificial de la Universidad de La Salle.
Automatiza la detección de crisis epilépticas en EEG pediátrico mediante Machine Learning clásico,
con énfasis en reproducibilidad y trazabilidad (Objetivo 3).

**Estudiante:** Erika Isabel Caita Avila | **Dataset:** CHB-MIT (7 sujetos pediátricos, 6–10 años)

---

## Estructura del proyecto

```
Essentials-EEG/
│
├── pipeline/                   ← PIPELINE REPRODUCIBLE DOE (Objetivo 3)
│   ├── 01_chbmit_experiments.py       Paso 1–3: extracción de features + 112 entrenamientos
│   ├── 02_consolidate_results.py      Paso 4: tabla comparativa + tests de Wilcoxon
│   ├── 03_generate_report_images.py   Paso 5: 9 figuras para el reporte (PDF)
│   ├── 04_generate_pdf_report.py      Paso 6: PDF final con resultados y análisis
│   └── run_doe_pipeline.ps1           Runner maestro — ejecuta los 4 pasos en orden
│
├── features/                   Extracción de características (vertical slices)
│   ├── signal_preprocessing/   Filtrado paso-banda, notch, resampling
│   ├── power_analysis/         Potencia relativa por banda (δ θ α β γ)
│   ├── spike_detection/        Detección de espigas epileptiformes
│   ├── connectivity_analysis/  Coherencia y Phase Lag Index (PLI)
│   ├── seizure_characterization/ Caracterización de patrones ictales
│   └── topographic_mapping/    Generación de mapas topográficos
│
├── shared/                     Infraestructura compartida
│   ├── data_access/            EDFLoader, SummaryParser (lectura CHB-MIT)
│   └── domain/                 Entidades: EEGRecording, FeatureExtractor
│
├── orchestration/              Orquestador del pipeline modular
│   └── pipeline_runner.py      EDF → preproceso → features → reporte
│
├── config/                     Configuración del pipeline
│   ├── base_config.py          Clases base: PreprocessConfig, BandDef, etc.
│   ├── subject_metadata.py     Metadatos demográficos de sujetos (SUBJECTS_DB)
│   └── experiments/            Variantes experimentales (pediatric, no_gamma, ...)
│
├── analysis/                   Análisis estadístico y visualizaciones
├── reporting/                  Generación de reportes HTML
├── experimental/               Exploración DL (CNN, LSTM) — fuera del alcance tesis
├── scripts/                    Scripts auxiliares y utilitarios
├── docs/                       Documentación del proyecto
│   ├── doe_experimental_design.md    DOE formal: hipótesis, factores, métricas
│   ├── copilot-instructions.md       Guía interna de revisión de código
│   └── SKILL_investigador_tutor.md   Skill de evaluación para defensa
│
├── data/                       Dataset CHB-MIT (registros EDF + anotaciones)
│
├── out_thesis_final/           Artefactos generados por el pipeline
│   ├── classical_all_models/   results.csv · cv_fold_metrics.csv · run_manifest.json
│   ├── report_images/          9 figuras PDF (heatmaps, barras, violins, Wilcoxon)
│   ├── comparison_table.csv    Tabla resumen filtrada por feature set
│   ├── reporte_experimentos.tex  Fuente LaTeX del reporte (compilable en Overleaf)
│   └── reporte_final_doe.pdf   PDF final del experimento
│
├── run_pipeline.py             Entrada al pipeline modular (Objetivo 1)
└── requirements.txt            Dependencias Python
```

---

## Reproducir el experimento DOE completo

### Prerrequisito

```powershell
py -3 -m pip install -r requirements.txt
```

### Ejecución completa (un solo comando)

```powershell
.\pipeline\run_doe_pipeline.ps1
```

Esto ejecuta los 4 pasos en orden y verifica que todos los artefactos se generen correctamente.

### Ejecución paso a paso

```powershell
# Paso 1-3: Extracción de features + 16 combos x 7 folds = 112 entrenamientos (~1-3 h)
py -3 .\pipeline\01_chbmit_experiments.py `
  --data_root ".\data" `
  --out_dir   ".\out_thesis_final\classical_all_models" `
  --subjects  chb05 chb09 chb14 chb16 chb20 chb22 chb23 `
  --window_sec 5 --overlap 0.5 --n_splits 7 `
  --run_name  "doe_v1_all_models"

# Paso 4: Tabla comparativa + Wilcoxon
py -3 -X utf8 .\pipeline\02_consolidate_results.py `
  --classical_dir ".\out_thesis_final\classical_all_models" `
  --out           ".\out_thesis_final\comparison_table.csv" `
  --feature_set   bp_plus_rms

# Paso 5: Figuras del reporte
py -3 -X utf8 .\pipeline\03_generate_report_images.py

# Paso 6: PDF final
py -3 -X utf8 .\pipeline\04_generate_pdf_report.py
```

### Artefactos generados

| Archivo | Descripción |
|---------|-------------|
| `out_thesis_final/classical_all_models/results.csv` | 16 filas: media ± std por combinación |
| `out_thesis_final/classical_all_models/cv_fold_metrics.csv` | 112 filas: métricas por fold (insumo Wilcoxon) |
| `out_thesis_final/classical_all_models/run_manifest.json` | Parámetros completos del run (reproducibilidad) |
| `out_thesis_final/comparison_table.csv` | Tabla filtrada por feature set de referencia |
| `out_thesis_final/report_images/` | 9 figuras PDF para el reporte |
| `out_thesis_final/reporte_final_doe.pdf` | PDF final con análisis y visualizaciones |

---

## Pipeline modular (Objetivo 1)

```powershell
python run_pipeline.py
python run_pipeline.py --config experiments.exp_pediatric
```

---

## Diseño experimental (DOE)

Ver [docs/doe_experimental_design.md](docs/doe_experimental_design.md) para:
- Hipótesis estadísticas (H₀, H₁, H₀_features)
- Factores y niveles (4 algoritmos × 4 feature sets)
- Estrategia de validación (GroupKFold por sujeto)
- Criterios de éxito (PR-AUC > 0.15)

---

## Licencia

El proyecto no incluye una licencia explícita en el repositorio.
