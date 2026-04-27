# config/experiments/exp_no_gamma.py
from config.base_config import BaseConfig, BandDefinitions
from pathlib import Path

# Configuración modificada: Sin banda Gamma
class NoGammaConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # Sobrescribir bandas (sin gamma)
        self.bands = BandDefinitions(
            delta=(0.5, 4),
            theta=(4, 8),
            alpha=(8, 13),
            beta=(13, 30),
            gamma=(0, 0)  # Desactivada
        )
        
        # Modificar directorio de resultados
        self.results_dir = Path("./results_no_gamma").resolve()
        self.experiment_name = "no_gamma"

CONFIG = NoGammaConfig()