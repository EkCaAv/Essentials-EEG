# config/base_config.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

@dataclass
class PreprocessingConfig:
    """Configuración de preprocesamiento de señal"""
    notch_freq: float = 60.0
    filter_low: float = 0.5
    filter_high: float = 40.0
    method: str = 'iir'

@dataclass
class BandDefinitions:
    """Definición de bandas de frecuencia"""
    delta: Tuple[float, float] = (0.5, 4)
    theta: Tuple[float, float] = (4, 8)
    alpha: Tuple[float, float] = (8, 13)
    beta: Tuple[float, float] = (13, 30)
    gamma: Tuple[float, float] = (30, 45)
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }

@dataclass
class SpikeDetectionConfig:
    """Configuración para detección de espigas"""
    z_threshold: float = 5.0
    min_distance_ms: float = 20.0
    filter_band: Tuple[float, float] = (14.0, 70.0)

@dataclass
class ConnectivityConfig:
    """Configuración de análisis de conectividad"""
    coherence_band: Tuple[float, float] = (4, 30)
    compute_pli: bool = True
    pli_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30)
    })

@dataclass
class VisualizationConfig:
    """Configuración de visualizaciones"""
    preferred_channels: List[str] = field(default_factory=lambda: [
        "F7-T7", "T7-P7", "F8-T8", "T8-P8", "FZ-CZ", "CZ-PZ"
    ])
    figure_dpi: int = 130
    spectrogram_window_sec: float = 2.0
    spectrogram_overlap_sec: float = 1.0

@dataclass
class BaseConfig:
    """Configuración completa del pipeline"""
    
    # 🗂️ RUTAS DE DATOS
    base_dir: Path = Path("./data").resolve()
    
    # Carpeta alternativa para summaries (si no están en subcarpetas de sujetos)
    summary_fallback_dir: Path = Path("./data/SUBJECT-INFO").resolve()
    
    # Carpeta de resultados
    results_dir: Path = Path("./results").resolve()
    
    # 👥 SUJETOS A PROCESAR
    subjects: List[str] = field(default_factory=lambda: 
        [f"chb{str(i).zfill(2)}" for i in range(1, 11)]
    )
    
    # ⚙️ CONFIGURACIONES DE FEATURES
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    bands: BandDefinitions = field(default_factory=BandDefinitions)
    spike_detection: SpikeDetectionConfig = field(default_factory=SpikeDetectionConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # 🏷️ NOMBRE DEL EXPERIMENTO
    experiment_name: str = "base"
    
    def __post_init__(self):
        """Validación y creación de directorios"""
        # Verificar que existe la carpeta de datos
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"❌ La carpeta de datos no existe: {self.base_dir}\n"
                f"   Crea la carpeta y coloca los datos de CHB-MIT allí."
            )
        
        # Crear carpeta de resultados
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar que existen sujetos
        found_subjects = [d.name for d in self.base_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('chb')]
        
        if not found_subjects:
            print(f"⚠️  ADVERTENCIA: No se encontraron carpetas de sujetos en {self.base_dir}")
            print(f"   Se esperaban carpetas como: chb01/, chb02/, ..., chb10/")