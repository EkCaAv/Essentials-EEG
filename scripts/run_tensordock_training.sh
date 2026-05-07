#!/usr/bin/env bash
set -euo pipefail

# TensorDock one-shot runner for CHB-MIT experiments
# It creates a Python 3.11 venv, installs dependencies,
# runs baseline/current experiments, and builds a comparison CSV.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv311}"
OUT_DIR="${OUT_DIR:-./out_tensordock}"

SUBJECTS="${SUBJECTS:-chb05 chb09 chb14 chb16 chb20 chb22 chb23}"
WINDOW_SEC="${WINDOW_SEC:-5}"
BASE_OVERLAP="${BASE_OVERLAP:-0.5}"
CURR_OVERLAP="${CURR_OVERLAP:-0.25}"
N_SPLITS="${N_SPLITS:-5}"
MAX_INTERICTAL_PER_FILE="${MAX_INTERICTAL_PER_FILE:-300}"
# Activa CV anidado con RandomizedSearchCV (recomendado para TensorDock).
# Exporta TUNE_HYPERPARAMS=0 para desactivarlo y ahorrar tiempo de cómputo.
TUNE_HYPERPARAMS="${TUNE_HYPERPARAMS:-1}"
INNER_CV_SPLITS="${INNER_CV_SPLITS:-3}"

BASE_RUN_NAME="${BASE_RUN_NAME:-tensordock_baseline}"
CURR_RUN_NAME="${CURR_RUN_NAME:-tensordock_current}"

echo "[1/6] Creating virtual environment with ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv "$VENV_DIR"

echo "[2/6] Installing dependencies"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" -m pip install scikit-learn

mkdir -p "$OUT_DIR/logs"

echo "[3/6] Running baseline experiment: ${BASE_RUN_NAME}"
"$VENV_DIR/bin/python" chbmit_experiments.py \
  --data_root ./data \
  --out_dir "$OUT_DIR" \
  --subjects $SUBJECTS \
  --window_sec "$WINDOW_SEC" \
  --overlap "$BASE_OVERLAP" \
  --n_splits "$N_SPLITS" \
  --max_interictal_per_file "$MAX_INTERICTAL_PER_FILE" \
  --run_name "$BASE_RUN_NAME" \
  $( [ "$TUNE_HYPERPARAMS" = "1" ] && echo "--tune_hyperparams --inner_cv_splits $INNER_CV_SPLITS" ) \
  2>&1 | tee "$OUT_DIR/logs/${BASE_RUN_NAME}.log"

echo "[4/6] Running current experiment: ${CURR_RUN_NAME}"
"$VENV_DIR/bin/python" chbmit_experiments.py \
  --data_root ./data \
  --out_dir "$OUT_DIR" \
  --subjects $SUBJECTS \
  --window_sec "$WINDOW_SEC" \
  --overlap "$CURR_OVERLAP" \
  --n_splits "$N_SPLITS" \
  --max_interictal_per_file "$MAX_INTERICTAL_PER_FILE" \
  --run_name "$CURR_RUN_NAME" \
  --compare_to "$OUT_DIR/runs/$BASE_RUN_NAME/results.csv" \
  $( [ "$TUNE_HYPERPARAMS" = "1" ] && echo "--tune_hyperparams --inner_cv_splits $INNER_CV_SPLITS" ) \
  2>&1 | tee "$OUT_DIR/logs/${CURR_RUN_NAME}.log"

echo "[5/6] Building manual comparison CSV"
"$VENV_DIR/bin/python" compare_experiment_runs.py \
  --current "$OUT_DIR/runs/$CURR_RUN_NAME/results.csv" \
  --baseline "$OUT_DIR/runs/$BASE_RUN_NAME/results.csv" \
  --output "$OUT_DIR/runs/$CURR_RUN_NAME/comparison_manual.csv"

echo "[6/6] Done"
echo "Baseline results: $OUT_DIR/runs/$BASE_RUN_NAME/results.csv"
echo "Current results:  $OUT_DIR/runs/$CURR_RUN_NAME/results.csv"
echo "Auto compare:     $OUT_DIR/runs/$CURR_RUN_NAME/comparison_vs_baseline.csv"
echo "Manual compare:   $OUT_DIR/runs/$CURR_RUN_NAME/comparison_manual.csv"
