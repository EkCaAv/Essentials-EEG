# features/connectivity_analysis/coherence.py
import numpy as np
from scipy.signal import coherence as scipy_coherence
from typing import Tuple

class CoherenceAnalyzer:
    """
    VERTICAL SLICE: Análisis de coherencia entre canales
    
    Responsabilidad: Calcular matrices de coherencia
    """
    
    def __init__(self, band: Tuple[float, float], nperseg_sec: float = 2.0, noverlap_sec: float = 1.0):
        self.band = band
        self.nperseg_sec = nperseg_sec
        self.noverlap_sec = noverlap_sec
    
    def compute_matrix(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Calcula matriz de coherencia para todos los pares de canales
        
        Args:
            data: Datos multicanal (n_channels x n_samples)
            fs: Frecuencia de muestreo
        
        Returns:
            Matriz de coherencia (n_channels x n_channels)
        """
        n_channels = data.shape[0]
        coh_matrix = np.eye(n_channels)
        
        nperseg = int(fs * self.nperseg_sec)
        noverlap = int(fs * self.noverlap_sec)
        lo, hi = self.band
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                f, Cxy = scipy_coherence(
                    data[i], data[j],
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
                
                mask = (f >= lo) & (f <= hi)
                mean_coh = float(np.nanmean(Cxy[mask])) if np.any(mask) else 0.0
                
                coh_matrix[i, j] = coh_matrix[j, i] = mean_coh
        
        return coh_matrix