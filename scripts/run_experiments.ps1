# run_experiments.ps1
# Ejecuta el experimento factorial 4x4 (algoritmo x feature_set) con GroupKFold por sujeto.
# Requiere: Python 3.11 + venv activado con dependencias instaladas.
#
# Uso:
#   .\run_experiments.ps1
#   .\run_experiments.ps1 -SkipDataBuild   # si windows_dataset.csv ya existe

param(
    [switch]$SkipDataBuild,
    [switch]$TuneHyperparams,
    [string]$RunName = "doe_v1_all_models"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Rutas ─────────────────────────────────────────────────────────────────────
$ROOT       = $PSScriptRoot
$DATA_ROOT  = Join-Path $ROOT "data"
$OUT_DIR    = Join-Path $ROOT "out_thesis_final\classical_all_models"
$DOC_DIR    = Join-Path $ROOT "out_thesis_final\doc"
$REPORT_DIR = Join-Path $ROOT "out_thesis_final\reports"

# ── Sujetos de la cohorte pediátrica 6–10 años ────────────────────────────────
$SUBJECTS = "chb05", "chb09", "chb14", "chb16", "chb20", "chb22", "chb23"

# ── Crear estructura de directorios ───────────────────────────────────────────
Write-Host "`n[SETUP] Creando estructura de directorios..."
New-Item -ItemType Directory -Force -Path $OUT_DIR    | Out-Null
New-Item -ItemType Directory -Force -Path $DOC_DIR    | Out-Null
New-Item -ItemType Directory -Force -Path $REPORT_DIR | Out-Null

# Copiar plan DoE al directorio de documentación
$DOE_SRC = Join-Path $ROOT "doe_experimental_design.md"
$DOE_DST = Join-Path $DOC_DIR "doe_experimental_design.md"
Copy-Item -Path $DOE_SRC -Destination $DOE_DST -Force
Write-Host "[SETUP] Plan DoE copiado a: $DOE_DST"

# ── Verificar Python ──────────────────────────────────────────────────────────
Write-Host "`n[CHECK] Verificando Python..."
$py = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python no encontrado. Instalar con: winget install Python.Python.3.11"
}
Write-Host "[CHECK] $py"

# ── Verificar dependencias críticas ──────────────────────────────────────────
Write-Host "[CHECK] Verificando dependencias..."
$deps = python -c "import mne, sklearn, numpy, pandas, scipy; print('OK')" 2>&1
if ($deps -ne "OK") {
    Write-Error "Dependencias faltantes. Ejecutar: pip install -r requirements.txt`n$deps"
}
Write-Host "[CHECK] Dependencias: OK"

# ── Paso 1: Experimento principal ─────────────────────────────────────────────
Write-Host "`n[STEP 1/2] Ejecutando experimento factorial 4x4..."
Write-Host "[INFO] Algoritmos : LogReg, RandomForest, SVM, GradientBoosting"
Write-Host "[INFO] Feature sets: bp_only, bp_plus_rms, bp_plus_rms_kurt, bp_plus_rms_kurt_skew"
Write-Host "[INFO] Folds       : 7 (GroupKFold por sujeto)"
Write-Host "[INFO] Sujetos     : $($SUBJECTS -join ', ')"
Write-Host "[INFO] Salida      : $OUT_DIR"
Write-Host ""

$args_list = @(
    ".\chbmit_experiments.py",
    "--data_root",  $DATA_ROOT,
    "--out_dir",    $OUT_DIR,
    "--subjects",   $SUBJECTS,
    "--window_sec", "5",
    "--overlap",    "0.5",
    "--n_splits",   "7",
    "--run_name",   $RunName
)

if ($TuneHyperparams) {
    $args_list += "--tune_hyperparams"
    Write-Host "[INFO] Nested CV activado (--tune_hyperparams)"
}

$t_start = Get-Date
python @args_list
if ($LASTEXITCODE -ne 0) {
    Write-Error "chbmit_experiments.py falló con código $LASTEXITCODE"
}
$elapsed = (Get-Date) - $t_start
Write-Host "`n[TIME] Experimento completado en $([int]$elapsed.TotalMinutes) min $($elapsed.Seconds) seg"

# ── Paso 2: Tabla comparativa ─────────────────────────────────────────────────
Write-Host "`n[STEP 2/2] Generando tabla comparativa..."
$COMPARISON_CSV = Join-Path $ROOT "out_thesis_final\comparison_table.csv"

python .\consolidate_results.py `
    --classical_dir $OUT_DIR `
    --out           $COMPARISON_CSV `
    --feature_set   "bp_plus_rms"

if ($LASTEXITCODE -ne 0) {
    Write-Warning "consolidate_results.py falló; la tabla comparativa no se generó."
} else {
    Write-Host "[OK] Tabla comparativa: $COMPARISON_CSV"
}

# ── Resumen final ─────────────────────────────────────────────────────────────
Write-Host "`n$(('=' * 70))"
Write-Host "EXPERIMENTO COMPLETADO"
Write-Host "$(('=' * 70))"
Write-Host "  Resultados     : $OUT_DIR\results.csv"
Write-Host "  Métricas/fold  : $OUT_DIR\cv_fold_metrics.csv"
Write-Host "  Manifiesto     : $OUT_DIR\run_manifest.json"
Write-Host "  Tabla tesis    : $COMPARISON_CSV"
Write-Host "  Plan DoE       : $DOC_DIR\doe_experimental_design.md"
Write-Host "$(('=' * 70))"
Write-Host ""
Write-Host "SIGUIENTE PASO: Revisar results.csv y ejecutar thesis_dashboard.py para visualización."
