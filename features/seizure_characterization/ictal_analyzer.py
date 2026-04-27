# features/seizure_characterization/ictal_analyzer.py
"""
Análisis especializado de ventanas ictales
"""

import numpy as np
from typing import Dict, List
from scipy.signal import welch
from dataclasses import dataclass

@dataclass
class IctalCharacteristics:
    """Características extraídas de una ventana ictal"""
    seizure_index: int
    duration_sec: float
    
    # Potencia espectral promedio
    avg_delta_power: float
    avg_theta_power: float
    avg_alpha_power: float
    avg_beta_power: float
    avg_gamma_power: float
    
    # Relación de potencias
    delta_theta_ratio: float
    alpha_beta_ratio: float
    
    # Evolución temporal
    power_trend: str  # 'increasing', 'decreasing', 'stable'
    dominant_frequency_hz: float
    
    # Sincronización
    mean_coherence: float
    max_coherence: float
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "seizure_index": self.seizure_index,
            "duration_sec": self.duration_sec,
            "avg_delta_power": self.avg_delta_power,
            "avg_theta_power": self.avg_theta_power,
            "avg_alpha_power": self.avg_alpha_power,
            "avg_beta_power": self.avg_beta_power,
            "avg_gamma_power": self.avg_gamma_power,
            "delta_theta_ratio": self.delta_theta_ratio,
            "alpha_beta_ratio": self.alpha_beta_ratio,
            "power_trend": self.power_trend,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "mean_coherence": self.mean_coherence,
            "max_coherence": self.max_coherence
        }

class IctalAnalyzer:
    """
    VERTICAL SLICE: Analizador de ventanas ictales
    
    Responsabilidad: Extraer características específicas de crisis
    """
    
    def __init__(self, bands: Dict[str, tuple]):
        """
        Args:
            bands: Definición de bandas de frecuencia
        """
        self.bands = bands
    
    def analyze(self, ictal_data: np.ndarray, fs: float, 
                seizure_index: int = 0) -> IctalCharacteristics:
        """
        Analiza una ventana ictal completa
        
        Args:
            ictal_data: Datos de la crisis (n_channels, n_samples)
            fs: Frecuencia de muestreo
            seizure_index: Índice de la crisis
        
        Returns:
            IctalCharacteristics con todas las métricas
        """
        n_channels, n_samples = ictal_data.shape
        duration_sec = n_samples / fs
        
        # 1. Potencia promedio por banda
        band_powers = self._compute_band_powers(ictal_data, fs)
        
        # 2. Frecuencia dominante
        dominant_freq = self._compute_dominant_frequency(ictal_data, fs)
        
        # 3. Tendencia de potencia temporal
        power_trend = self._compute_power_trend(ictal_data, fs)
        
        # 4. Coherencia promedio (simplificada)
        mean_coh, max_coh = self._compute_mean_coherence(ictal_data, fs)
        
        return IctalCharacteristics(
            seizure_index=seizure_index,
            duration_sec=duration_sec,
            avg_delta_power=band_powers['delta'],
            avg_theta_power=band_powers['theta'],
            avg_alpha_power=band_powers['alpha'],
            avg_beta_power=band_powers['beta'],
            avg_gamma_power=band_powers['gamma'],
            delta_theta_ratio=band_powers['delta'] / (band_powers['theta'] + 1e-12),
            alpha_beta_ratio=band_powers['alpha'] / (band_powers['beta'] + 1e-12),
            power_trend=power_trend,
            dominant_frequency_hz=dominant_freq,
            mean_coherence=mean_coh,
            max_coherence=max_coh
        )
    
    def _compute_band_powers(self, data: np.ndarray, fs: float) -> Dict[str, float]:
        """Calcula potencia promedio por banda (sobre todos los canales)"""
        n_channels = data.shape[0]
        
        band_sums = {band: 0.0 for band in self.bands}
        
        for ch_idx in range(n_channels):
            f, Pxx = welch(data[ch_idx], fs=fs, nperseg=int(fs*2), noverlap=int(fs))
            
            for band_name, (lo, hi) in self.bands.items():
                mask = (f >= lo) & (f <= hi)
                power = np.trapz(Pxx[mask], f[mask])
                band_sums[band_name] += power
        
        # Promediar sobre canales
        return {band: power / n_channels for band, power in band_sums.items()}
    
    def _compute_dominant_frequency(self, data: np.ndarray, fs: float) -> float:
        """Encuentra la frecuencia dominante"""
        # Promediar PSD sobre todos los canales
        n_channels = data.shape[0]
        
        f, Pxx_avg = welch(data[0], fs=fs, nperseg=int(fs*2), noverlap=int(fs))
        for ch_idx in range(1, n_channels):
            _, Pxx = welch(data[ch_idx], fs=fs, nperseg=int(fs*2), noverlap=int(fs))
            Pxx_avg += Pxx
        
        Pxx_avg /= n_channels
        
        # Encontrar pico en rango 1-30 Hz (rango típico de crisis)
        mask = (f >= 1) & (f <= 30)
        f_range = f[mask]
        Pxx_range = Pxx_avg[mask]
        
        dominant_idx = np.argmax(Pxx_range)
        return float(f_range[dominant_idx])
    
    def _compute_power_trend(self, data: np.ndarray, fs: float) -> str:
        """
        Determina si la potencia aumenta, disminuye o es estable durante la crisis
        """
        n_channels, n_samples = data.shape
        
        # Dividir en 3 tercios
        third = n_samples // 3
        
        power_first = np.mean(np.abs(data[:, :third]) ** 2)
        power_last = np.mean(np.abs(data[:, -third:]) ** 2)
        
        ratio = power_last / (power_first + 1e-12)
        
        if ratio > 1.2:
            return 'increasing'
        elif ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _compute_mean_coherence(self, data: np.ndarray, fs: float) -> tuple:
        """
        Calcula coherencia promedio entre pares de canales (simplificada)
        """
        from scipy.signal import coherence
        
        n_channels = data.shape[0]
        coherences = []
        
        # Calcular para algunos pares representativos (no todos para eficiencia)
        for i in range(min(5, n_channels)):
            for j in range(i + 1, min(i + 3, n_channels)):
                f, Cxy = coherence(data[i], data[j], fs=fs, 
                                  nperseg=int(fs*2), noverlap=int(fs))
                # Coherencia en banda 4-30 Hz
                mask = (f >= 4) & (f <= 30)
                coherences.append(np.mean(Cxy[mask]))
        
        if not coherences:
            return 0.0, 0.0
        
        return float(np.mean(coherences)), float(np.max(coherences))