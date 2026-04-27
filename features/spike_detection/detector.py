# features/spike_detection/detector.py
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Tuple, Dict
from config.base_config import SpikeDetectionConfig

class SpikeDetector:
    """
    VERTICAL SLICE: Detección de espigas epileptiformes
    
    Responsabilidad: Identificar eventos tipo espiga en señales EEG
    """
    
    def __init__(self, config: SpikeDetectionConfig):
        self.config = config
    
    def detect(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, Dict]:
        """
        Detecta espigas usando umbral de Z-score
        
        Args:
            signal: Señal de un canal
            fs: Frecuencia de muestreo
        
        Returns:
            peaks: Índices de los picos detectados
            stats: Estadísticas de detección
        """
        # Filtro pasa-banda para resaltar espigas
        lo, hi = self.config.filter_band
        b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        filtered = filtfilt(b, a, signal)
        
        # Normalización Z-score
        z_scores = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)
        
        # Detección de picos
        min_distance_samples = int(fs * (self.config.min_distance_ms / 1000.0))
        peaks, _ = find_peaks(
            np.abs(z_scores),
            height=self.config.z_threshold,
            distance=min_distance_samples
        )
        
        stats = {
            "z_mean_abs": float(np.mean(np.abs(z_scores))),
            "n_peaks": int(len(peaks)),
            "peak_rate_per_min": float(len(peaks) / (len(signal) / fs / 60))
        }
        
        return peaks, stats