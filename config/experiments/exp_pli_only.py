# config/experiments/exp_pli_only.py
from config.base_config import BaseConfig, ConnectivityConfig
from pathlib import Path

class PLIOnlyConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # Solo PLI, sin coherencia
        self.connectivity = ConnectivityConfig(
            coherence_band=(4, 30),
            compute_pli=True,
            pli_bands={
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30)
            }
        )
        
        self.results_dir = Path("./results_pli_only").resolve()
        self.experiment_name = "pli_only_all_bands"

CONFIG = PLIOnlyConfig()