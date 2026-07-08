# run_interface.ps1
# Lanza la interfaz explicativa del pipeline EEG pediátrico (Fase 1).
#
# Uso:
#   .\run_interface.ps1
#
# Requisitos: streamlit y pyyaml (incluidos en requirements.txt).

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Write-Host "Iniciando interfaz EEG Pediatrico..." -ForegroundColor Cyan
Write-Host "Se abrira en el navegador (http://localhost:8501)" -ForegroundColor Cyan

py -3 -m streamlit run (Join-Path $here "interface\app.py")
