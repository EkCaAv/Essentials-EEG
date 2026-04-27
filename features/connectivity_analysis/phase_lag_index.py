# features/connectivity_analysis/phase_lag_index.py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from typing import Tuple

class PLIAnalyzer:
    """
    VERTICAL SLICE: Análisis de Phase-Lag Index (PLI)
    
    Responsabilidad: Calcular PLI para evaluar sincronización de fase
    """
    
    def __init__(self, band: Tuple[float, float]):
        self.band = band
    
    def _bandpass_filter(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Aplica filtro pasa-banda"""
        lo, hi = self.band
        b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return filtfilt(b, a, signal)
    
    def compute_matrix(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Calcula matriz de PLI
        
        Args:
            data: Datos multicanal
            fs: Frecuencia de muestreo
        
        Returns:
            Matriz PLI
        """
        n_channels = data.shape[0]
        
        # Filtrar todas las señales en la banda
        filtered = np.vstack([self._bandpass_filter(ch, fs) for ch in data])
        
        # Calcular fase instantánea
        phases = np.angle(hilbert(filtered, axis=1))
        
        # Matriz PLI
        pli_matrix = np.eye(n_channels)
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phases[i] - phases[j]
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli_matrix[i, j] = pli_matrix[j, i] = float(pli)
        
        return pli_matrix