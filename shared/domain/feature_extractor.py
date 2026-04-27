# shared/domain/feature_extractor.py
"""
Servicio de extracción de características numéricas (Feature Engineering)
Versión Mejorada: Soporta bandas dinámicas y conteo de spikes.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from typing import Dict, List, Optional
from .eeg_recording import EEGRecording

class FeatureExtractor:
    """
    Calcula métricas cuantitativas de la señal EEG.
    Prepara el terreno para el Análisis Descriptivo y ML.
    """
    
    def __init__(self, bands_config=None):
        """
        Args:
            bands_config: Objeto BandsConfig o dict con rangos de frecuencia.
                          Si es None, usa valores por defecto estándar.
        """
        if bands_config:
            # Si viene del objeto de configuración, lo convertimos a dict si es necesario
            self.bands = bands_config.to_dict() if hasattr(bands_config, 'to_dict') else bands_config
        else:
            # Bandas por defecto si no se especifica nada
            self.bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }

    def extract_features(self, recording: EEGRecording, spike_events: List[dict] = None) -> Dict:
        """
        Extrae un diccionario completo de características de la grabación.
        
        Args:
            recording: Objeto EEGRecording con los datos.
            spike_events: Lista de spikes detectados (opcional).
        """
        features = {}
        data = recording.data  # (n_channels, n_samples)
        sfreq = recording.sampling_rate
        duration = recording.duration_sec
        
        # 1. Metadatos Básicos
        features['file_id'] = recording.file_id
        features['duration_sec'] = duration
        features['has_seizures'] = recording.has_seizures()
        features['n_seizures'] = len(recording.seizure_windows)

        # 2. Estadísticas de Spikes (¡Nuevo!)
        if spike_events is not None:
            n_spikes = len(spike_events)
            features['n_spikes'] = n_spikes
            features['spike_rate_min'] = (n_spikes / duration) * 60 if duration > 0 else 0
            
            # Amplitud promedio de los spikes (si existe el dato)
            amplitudes = [s.get('max_amplitude', 0) for s in spike_events]
            features['mean_spike_amp'] = np.mean(amplitudes) if amplitudes else 0
        else:
            features['n_spikes'] = 0
            features['spike_rate_min'] = 0

        # 3. Estadísticas Temporales (Promedio global)
        # Nota: Usamos nanmean para seguridad
        features['rms_mean'] = np.mean(np.sqrt(np.mean(data**2, axis=1)))
        features['kurtosis_mean'] = np.mean(kurtosis(data, axis=1))
        features['skewness_mean'] = np.mean(skew(data, axis=1))
        
        # 4. Características Espectrales (Band Powers)
        nperseg = min(int(2 * sfreq), data.shape[1])  # Ventana de 2s
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
        
        # Potencia total por canal
        total_power_ch = np.sum(psd, axis=1)
        # Evitar división por cero
        total_power_ch[total_power_ch == 0] = 1e-10
        
        features['total_power'] = np.mean(total_power_ch)
        
        for band_name, (fmin, fmax) in self.bands.items():
            if fmin is None or fmax is None: continue # Saltar bandas desactivadas

            # Índices de frecuencia para la banda
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            if np.sum(idx_band) == 0:
                features[f'abs_power_{band_name}'] = 0
                features[f'rel_power_{band_name}'] = 0
                continue

            # Potencia absoluta en la banda (por canal)
            band_power_ch = np.sum(psd[:, idx_band], axis=1)
            
            # Promediar entre todos los canales para tener un escalar
            features[f'abs_{band_name}'] = np.mean(band_power_ch)
            
            # Potencia relativa (por canal y luego promedio)
            rel_power_ch = band_power_ch / total_power_ch
            features[f'rel_{band_name}'] = np.mean(rel_power_ch)
            
        return features