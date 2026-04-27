# shared/data_access/edf_loader.py
from pathlib import Path
from typing import List, Optional
import mne
from ..domain.eeg_recording import EEGRecording, SeizureWindow

class EDFLoader:
    """Carga archivos EDF y los convierte en entidades de dominio"""
    
    @staticmethod
    def load(edf_path: Path, seizure_windows: Optional[List[SeizureWindow]] = None) -> EEGRecording:
        """
        Carga un archivo EDF
        
        Args:
            edf_path: Ruta al archivo EDF
            seizure_windows: Lista de ventanas de crisis (opcional)
        
        Returns:
            EEGRecording: Entidad de dominio
        """
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        
        return EEGRecording(
            file_path=str(edf_path),
            raw=raw,
            seizure_windows=seizure_windows or []
        )