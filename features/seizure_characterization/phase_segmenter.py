# features/seizure_characterization/phase_segmenter.py
"""
Segmentación de fases: preictal, ictal, postictal, interictal
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from shared.domain.eeg_recording import EEGRecording

@dataclass
class PhaseSegment:
    """Segmento de una fase específica"""
    phase: str  # 'preictal', 'ictal', 'postictal', 'interictal'
    start_sec: float
    end_sec: float
    data: np.ndarray  # (n_channels, n_samples)
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

class PhaseSegmenter:
    """
    VERTICAL SLICE: Segmentador de fases de crisis
    
    Responsabilidad: Extraer ventanas temporales de diferentes fases
    """
    
    def __init__(self, 
                 preictal_duration_sec: float = 60.0,
                 postictal_duration_sec: float = 60.0):
        """
        Args:
            preictal_duration_sec: Duración de ventana preictal (antes de crisis)
            postictal_duration_sec: Duración de ventana postictal (después de crisis)
        """
        self.preictal_duration = preictal_duration_sec
        self.postictal_duration = postictal_duration_sec
    
    def segment(self, recording: EEGRecording) -> Dict[str, List[PhaseSegment]]:
        """
        Segmenta todas las fases de una grabación
        
        Returns:
            Dict con listas de segmentos por fase:
            {
                'preictal': [PhaseSegment, ...],
                'ictal': [PhaseSegment, ...],
                'postictal': [PhaseSegment, ...],
                'interictal': [PhaseSegment, ...]
            }
        """
        fs = recording.sampling_rate
        data = recording.data
        
        segments = {
            'preictal': [],
            'ictal': [],
            'postictal': [],
            'interictal': []
        }
        
        if not recording.has_seizures():
            # Toda la grabación es interictal
            segments['interictal'].append(PhaseSegment(
                phase='interictal',
                start_sec=0,
                end_sec=recording.duration_sec,
                data=data
            ))
            return segments
        
        # Procesar cada crisis
        for seizure in recording.seizure_windows:
            # === PREICTAL ===
            preictal_start = max(0, seizure.start_sec - self.preictal_duration)
            preictal_end = seizure.start_sec
            
            if preictal_end > preictal_start:
                a = int(preictal_start * fs)
                b = int(preictal_end * fs)
                if b <= data.shape[1]:
                    segments['preictal'].append(PhaseSegment(
                        phase='preictal',
                        start_sec=preictal_start,
                        end_sec=preictal_end,
                        data=data[:, a:b]
                    ))
            
            # === ICTAL ===
            ictal_start = seizure.start_sec
            ictal_end = seizure.end_sec
            
            a = int(ictal_start * fs)
            b = int(ictal_end * fs)
            if b <= data.shape[1]:
                segments['ictal'].append(PhaseSegment(
                    phase='ictal',
                    start_sec=ictal_start,
                    end_sec=ictal_end,
                    data=data[:, a:b]
                ))
            
            # === POSTICTAL ===
            postictal_start = seizure.end_sec
            postictal_end = min(recording.duration_sec, 
                               seizure.end_sec + self.postictal_duration)
            
            if postictal_end > postictal_start:
                a = int(postictal_start * fs)
                b = int(postictal_end * fs)
                if b <= data.shape[1]:
                    segments['postictal'].append(PhaseSegment(
                        phase='postictal',
                        start_sec=postictal_start,
                        end_sec=postictal_end,
                        data=data[:, a:b]
                    ))
        
        # === INTERICTAL ===
        # Extraer segmentos que no están en ninguna otra fase
        segments['interictal'] = self._extract_interictal(recording, segments)
        
        return segments
    
    def _extract_interictal(self, recording: EEGRecording, 
                           phase_segments: Dict) -> List[PhaseSegment]:
        """
        Extrae segmentos interictales (que no se solapan con otras fases)
        """
        fs = recording.sampling_rate
        data = recording.data
        duration = recording.duration_sec
        
        # Crear máscara de tiempo ocupado
        occupied_mask = np.zeros(int(duration * fs), dtype=bool)
        
        for phase in ['preictal', 'ictal', 'postictal']:
            for segment in phase_segments[phase]:
                a = int(segment.start_sec * fs)
                b = int(segment.end_sec * fs)
                occupied_mask[a:b] = True
        
        # Encontrar regiones libres
        interictal_segments = []
        in_free = False
        start_idx = 0
        
        for i, is_occupied in enumerate(occupied_mask):
            if not is_occupied and not in_free:
                # Inicio de región libre
                in_free = True
                start_idx = i
            elif is_occupied and in_free:
                # Fin de región libre
                in_free = False
                if i - start_idx > fs * 10:  # Al menos 10 segundos
                    interictal_segments.append(PhaseSegment(
                        phase='interictal',
                        start_sec=start_idx / fs,
                        end_sec=i / fs,
                        data=data[:, start_idx:i]
                    ))
        
        # Última región si quedó abierta
        if in_free and len(occupied_mask) - start_idx > fs * 10:
            interictal_segments.append(PhaseSegment(
                phase='interictal',
                start_sec=start_idx / fs,
                end_sec=len(occupied_mask) / fs,
                data=data[:, start_idx:]
            ))
        
        return interictal_segments
    
    def get_statistics(self, segments: Dict[str, List[PhaseSegment]]) -> Dict:
        """Retorna estadísticas de la segmentación"""
        stats = {}
        
        for phase in ['preictal', 'ictal', 'postictal', 'interictal']:
            phase_segments = segments[phase]
            
            if not phase_segments:
                stats[phase] = {
                    "count": 0,
                    "total_duration_sec": 0,
                    "avg_duration_sec": 0
                }
            else:
                durations = [s.duration_sec for s in phase_segments]
                stats[phase] = {
                    "count": len(phase_segments),
                    "total_duration_sec": sum(durations),
                    "avg_duration_sec": np.mean(durations),
                    "min_duration_sec": min(durations),
                    "max_duration_sec": max(durations)
                }
        
        return stats