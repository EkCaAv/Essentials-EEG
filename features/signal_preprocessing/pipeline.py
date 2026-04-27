# features/signal_preprocessing/pipeline.py
import mne
from config.base_config import PreprocessingConfig

class SignalPreprocessor:
    """
    VERTICAL SLICE: Preprocesamiento de señal EEG
    
    Responsabilidad: Aplicar filtros y limpieza a señales crudas
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def preprocess(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Pipeline completo de preprocesamiento
        
        Args:
            raw: Señal cruda de MNE
        
        Returns:
            raw: Señal preprocesada
        """
        # Filtro notch
        raw.notch_filter(
            [self.config.notch_freq],
            method=self.config.method,
            verbose="ERROR"
        )
        
        # Filtro pasa-banda
        raw.filter(
            self.config.filter_low,
            self.config.filter_high,
            method=self.config.method,
            verbose="ERROR"
        )
        
        return raw