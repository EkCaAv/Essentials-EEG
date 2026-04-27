# config/experiments/exp_smoke_test.py
"""
Configuración para Smoke Test: un solo archivo de un solo sujeto.
"""

from pathlib import Path
from dataclasses import field
from config.base_config import BaseConfig


class SmokeTestConfig(BaseConfig):
    """Configuración para test de humo con un solo archivo"""
    
    def __init__(self):
        super().__init__()
        
        # 1. Limitar a un solo sujeto
        self.subjects = ["chb01"]
        
        # 2. Especificar un solo archivo a procesar
        self.target_files = ["chb01_03.edf"]
        
        # 3. Guardar resultados en una carpeta separada
        self.results_dir = Path("./results_smoke_test").resolve()
        
        # 4. Nombre del experimento
        self.experiment_name = "smoke_test_single_file"


# ¡IMPORTANTE! Exponer la configuración como variable
CONFIG = SmokeTestConfig()