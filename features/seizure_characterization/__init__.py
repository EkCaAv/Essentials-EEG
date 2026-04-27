# features/seizure_characterization/__init__.py
"""
VERTICAL SLICE: Caracterización de Crisis Epilépticas

Responsabilidad: 
- Parsear anotaciones de crisis
- Segmentar fases (preictal, ictal, postictal, interictal)
- Extraer características específicas de crisis
- Generar resúmenes de crisis
"""

from .window_parser import SeizureWindowParser
from .ictal_analyzer import IctalAnalyzer
from .phase_segmenter import PhaseSegmenter
from .summary_builder import SeizureSummaryBuilder

__all__ = [
    'SeizureWindowParser',
    'IctalAnalyzer',
    'PhaseSegmenter',
    'SeizureSummaryBuilder'
]