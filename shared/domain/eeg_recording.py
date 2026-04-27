# shared/domain/eeg_recording.py
"""
Entidades de dominio para grabaciones EEG
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path
import numpy as np
import mne


@dataclass
class SeizureWindow:
    """Representa una ventana de crisis epiléptica"""
    start_sec: float
    end_sec: float
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class EEGRecording:
    """Entidad de dominio: grabación de EEG"""
    file_path: str
    raw: mne.io.BaseRaw
    seizure_windows: List[SeizureWindow]
    
    @property
    def file_id(self) -> str:
        """Retorna solo el nombre del archivo (sin ruta)"""
        return Path(self.file_path).name
    
    @property
    def sampling_rate(self) -> float:
        return float(self.raw.info['sfreq'])
    
    @property
    def channels(self) -> List[str]:
        return list(self.raw.ch_names)
    
    @property
    def n_channels(self) -> int:
        return len(self.channels)
    
    @property
    def duration_sec(self) -> float:
        return self.raw.times[-1]
    
    @property
    def data(self) -> np.ndarray:
        """Retorna los datos de la señal (n_channels x n_samples)"""
        return self.raw.get_data()
    
    def has_seizures(self) -> bool:
        """Indica si la grabación tiene crisis anotadas"""
        return len(self.seizure_windows) > 0