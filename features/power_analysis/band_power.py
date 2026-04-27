# features/power_analysis/band_power.py
import numpy as np
import pandas as pd
from scipy.signal import welch
from typing import Dict
from config.base_config import BandDefinitions

class BandPowerAnalyzer:
    """
    VERTICAL SLICE: Análisis de potencia por bandas
    
    Responsabilidad: Calcular potencia absoluta y relativa en bandas de frecuencia
    """
    
    def __init__(self, bands: BandDefinitions, nperseg_sec: float = 2.0, noverlap_sec: float = 1.0):
        self.bands = bands.to_dict()
        self.nperseg_sec = nperseg_sec
        self.noverlap_sec = noverlap_sec
    
    def compute_bandpower(self, signal: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Calcula potencia de banda usando método de Welch
        
        Args:
            signal: Señal de un canal (1D array)
            fs: Frecuencia de muestreo
        
        Returns:
            Dict con potencias absolutas y relativas por banda
        """
        nperseg = int(fs * self.nperseg_sec)
        noverlap = int(fs * self.noverlap_sec)
        
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        total_power = np.trapz(Pxx, f) + 1e-12
        
        results = {}
        for band_name, (lo, hi) in self.bands.items():
            mask = (f >= lo) & (f <= hi)
            band_power = np.trapz(Pxx[mask], f[mask])
            
            results[band_name] = band_power
            results[f"{band_name}_rel"] = band_power / total_power
        
        return results
    
    def compute_temporal_evolution(self, data: np.ndarray, fs: float, 
                                   segment_duration_sec: float = 10) -> pd.DataFrame:
        """
        Calcula evolución temporal de potencia de banda
        
        Args:
            data: Datos multicanal (n_channels x n_samples)
            fs: Frecuencia de muestreo
            segment_duration_sec: Duración de cada segmento
        
        Returns:
            DataFrame con evolución temporal
        """
        n_channels, n_samples = data.shape
        seg_len = int(fs * segment_duration_sec)
        n_segs = max(1, n_samples // seg_len)
        
        rows = []
        for seg_idx in range(n_segs):
            a = seg_idx * seg_len
            b = a + seg_len
            if b > n_samples:
                break
            
            # Promediar sobre todos los canales
            avg_powers = {k: 0.0 for k in self.bands}
            avg_powers_rel = {f"{k}_rel": 0.0 for k in self.bands}
            
            for ch_idx in range(n_channels):
                bp = self.compute_bandpower(data[ch_idx, a:b], fs)
                for k in self.bands:
                    avg_powers[k] += bp[k]
                    avg_powers_rel[f"{k}_rel"] += bp[f"{k}_rel"]
            
            for k in self.bands:
                avg_powers[k] /= n_channels
                avg_powers_rel[f"{k}_rel"] /= n_channels
            
            row = {"time_sec": a / fs}
            row.update(avg_powers)
            row.update(avg_powers_rel)
            rows.append(row)
        
        return pd.DataFrame(rows)