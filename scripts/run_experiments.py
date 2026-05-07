#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🧪 Ejecutor de Experimentos en Batch

Ejecuta múltiples configuraciones de forma secuencial
"""

import subprocess
import sys
from pathlib import Path

EXPERIMENTS = [
    "base_config",
    "experiments.exp_no_gamma",
    "experiments.exp_high_spike_threshold",
    "experiments.exp_pli_only"
]

def run_experiment(config_name: str):
    """Ejecuta un experimento"""
    print(f"\n{'='*70}")
    print(f"🧪 Ejecutando experimento: {config_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--config", config_name],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"❌ Error en experimento: {config_name}")
        return False
    
    print(f"\n✅ Experimento completado: {config_name}\n")
    return True

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║       🧪 EJECUTOR DE EXPERIMENTOS EN BATCH                ║
    ║       Pipeline de Análisis EEG                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        results[exp] = "✅ Exitoso" if success else "❌ Falló"
    
    # Resumen
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE EXPERIMENTOS")
    print(f"{'='*70}\n")
    
    for exp, status in results.items():
        print(f"  {status}  {exp}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()