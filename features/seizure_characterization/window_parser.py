# features/seizure_characterization/window_parser.py
"""
Parser de ventanas de crisis desde archivos summary
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SeizureAnnotation:
    """Anotación de una crisis epiléptica"""
    file_name: str
    seizure_number: int
    start_sec: float
    end_sec: float
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
    
    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60.0
    
    def __repr__(self) -> str:
        return (f"SeizureAnnotation(file={self.file_name}, "
                f"seizure#{self.seizure_number}, "
                f"{self.start_sec:.1f}s-{self.end_sec:.1f}s, "
                f"duration={self.duration_sec:.1f}s)")

class SeizureWindowParser:
    """
    VERTICAL SLICE: Parser de ventanas de crisis
    
    Responsabilidad: Extraer anotaciones de crisis desde archivos summary
    """
    
    def __init__(self):
        self.annotations: Dict[str, List[SeizureAnnotation]] = {}
        self.sampling_rate: Optional[float] = None
    
    def parse(self, summary_path: Path) -> Dict[str, List[SeizureAnnotation]]:
        """
        Parsea un archivo chbXX-summary.txt
        
        Args:
            summary_path: Ruta al archivo summary
        
        Returns:
            Dict[file_name, List[SeizureAnnotation]]
        """
        if not summary_path or not summary_path.exists():
            return {}
        
        content = summary_path.read_text(errors="ignore").splitlines()
        
        current_file = None
        pending_starts = []
        seizure_counter = {}
        
        for line in content:
            line = line.strip()
            
            # Detectar frecuencia de muestreo
            if not self.sampling_rate:
                match = re.match(r"Sampling Rate:\s*([\d.]+)\s*Hz", line, re.I)
                if match:
                    self.sampling_rate = float(match.group(1))
                    continue
            
            # Detectar nombre de archivo
            match = re.match(r"File Name:\s*(\S+)", line)
            if match:
                current_file = match.group(1)
                self.annotations.setdefault(current_file, [])
                seizure_counter.setdefault(current_file, 0)
                pending_starts = []
                continue
            
            # Detectar inicio de crisis
            match = re.match(r"Seizure(?:\s+(\d+))?\s+Start Time:\s*([\d.]+)\s*seconds?", line, re.I)
            if match and current_file:
                start_time = float(match.group(2))
                pending_starts.append(start_time)
                continue
            
            # Detectar fin de crisis
            match = re.match(r"Seizure(?:\s+(\d+))?\s+End Time:\s*([\d.]+)\s*seconds?", line, re.I)
            if match and current_file and pending_starts:
                end_time = float(match.group(2))
                start_time = pending_starts.pop(0)
                
                seizure_counter[current_file] += 1
                
                annotation = SeizureAnnotation(
                    file_name=current_file,
                    seizure_number=seizure_counter[current_file],
                    start_sec=start_time,
                    end_sec=end_time
                )
                
                self.annotations[current_file].append(annotation)
        
        return self.annotations
    
    def get_all_seizures(self) -> List[SeizureAnnotation]:
        """Retorna lista plana de todas las crisis"""
        all_seizures = []
        for seizures in self.annotations.values():
            all_seizures.extend(seizures)
        return all_seizures
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de las crisis parseadas"""
        all_seizures = self.get_all_seizures()
        
        if not all_seizures:
            return {
                "total_seizures": 0,
                "total_files_with_seizures": 0,
                "avg_duration_sec": 0,
                "min_duration_sec": 0,
                "max_duration_sec": 0
            }
        
        durations = [s.duration_sec for s in all_seizures]
        
        return {
            "total_seizures": len(all_seizures),
            "total_files_with_seizures": len(self.annotations),
            "avg_duration_sec": sum(durations) / len(durations),
            "min_duration_sec": min(durations),
            "max_duration_sec": max(durations),
            "sampling_rate_hz": self.sampling_rate
        }