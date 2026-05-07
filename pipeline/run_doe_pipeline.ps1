# run_doe_pipeline.ps1
# Pipeline DOE reproducible — Deteccion de Crisis Epilepticas en EEG Pediatrico
# Universidad de La Salle | Maestria en Inteligencia Artificial
#
# Ejecutar desde la raiz del proyecto:
#   .\pipeline\run_doe_pipeline.ps1
#
# Prerrequisito: pip install -r requirements.txt

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "=========================================================="
Write-Host "  PIPELINE DOE — doe_v1_all_models"
Write-Host "  Sujetos: chb05 chb09 chb14 chb16 chb20 chb22 chb23"
Write-Host "  16 combinaciones x 7 folds = 112 entrenamientos"
Write-Host "=========================================================="
Write-Host ""

# ----------------------------------------------------------
# PASO 1 — Extraccion de features + experimentos ML
# ----------------------------------------------------------
Write-Host "[PASO 1/4] Experimentos ML (chbmit_experiments.py)..."
py -3 -X utf8 "$ROOT\pipeline\01_chbmit_experiments.py" `
  --data_root  "$ROOT\data" `
  --out_dir    "$ROOT\out_thesis_final\classical_all_models" `
  --subjects   chb05 chb09 chb14 chb16 chb20 chb22 chb23 `
  --window_sec 5 `
  --overlap    0.5 `
  --n_splits   7 `
  --run_name   "doe_v1_all_models"

if (-not $?) { Write-Error "PASO 1 fallo. Abortando."; exit 1 }
Write-Host "[PASO 1] OK`n"

# ----------------------------------------------------------
# PASO 2 — Tabla comparativa + Wilcoxon
# ----------------------------------------------------------
Write-Host "[PASO 2/4] Consolidacion y Wilcoxon (consolidate_results.py)..."
$env:PYTHONIOENCODING = "utf-8"
py -3 -X utf8 "$ROOT\pipeline\02_consolidate_results.py" `
  --classical_dir "$ROOT\out_thesis_final\classical_all_models" `
  --out           "$ROOT\out_thesis_final\comparison_table.csv" `
  --feature_set   bp_plus_rms

if (-not $?) { Write-Error "PASO 2 fallo. Abortando."; exit 1 }
Write-Host "[PASO 2] OK`n"

# ----------------------------------------------------------
# PASO 3 — Figuras del reporte
# ----------------------------------------------------------
Write-Host "[PASO 3/4] Generacion de figuras (generate_report_images.py)..."
py -3 -X utf8 "$ROOT\pipeline\03_generate_report_images.py"

if (-not $?) { Write-Error "PASO 3 fallo. Abortando."; exit 1 }
Write-Host "[PASO 3] OK`n"

# ----------------------------------------------------------
# PASO 4 — PDF final
# ----------------------------------------------------------
Write-Host "[PASO 4/4] PDF final (generate_pdf_report.py)..."
py -3 -X utf8 "$ROOT\pipeline\04_generate_pdf_report.py"

if (-not $?) { Write-Error "PASO 4 fallo. Abortando."; exit 1 }
Write-Host "[PASO 4] OK`n"

# ----------------------------------------------------------
# Resumen de artefactos generados
# ----------------------------------------------------------
Write-Host "=========================================================="
Write-Host "  PIPELINE COMPLETADO — Artefactos generados:"
Write-Host "=========================================================="
$artifacts = @(
    "$ROOT\out_thesis_final\classical_all_models\results.csv",
    "$ROOT\out_thesis_final\classical_all_models\cv_fold_metrics.csv",
    "$ROOT\out_thesis_final\classical_all_models\run_manifest.json",
    "$ROOT\out_thesis_final\comparison_table.csv",
    "$ROOT\out_thesis_final\reporte_final_doe.pdf"
)
foreach ($f in $artifacts) {
    if (Test-Path $f) {
        $size = [math]::Round((Get-Item $f).Length / 1KB, 1)
        Write-Host "  [OK] $(Split-Path $f -Leaf)  ($size KB)"
    } else {
        Write-Host "  [FALTA] $f"
    }
}
Write-Host ""
