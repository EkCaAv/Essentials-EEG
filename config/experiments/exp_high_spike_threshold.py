# config/experiments/exp_high_spike_threshold.py
from config.base_config import BaseConfig, SpikeDetectionConfig
from pathlib import Path

class HighSpikeThresholdConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # Umbral más alto para spikes
        self.spike_detection = SpikeDetectionConfig(
            z_threshold=7.0,  # Base: 5.0
            min_distance_ms=20.0,
            filter_band=(14.0, 70.0)
        )
        
        self.results_dir = Path("./results_high_spike_thresh").resolve()
        self.experiment_name = "high_spike_threshold_7"

CONFIG = HighSpikeThresholdConfig()