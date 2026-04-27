# shared/data_access/summary_parser.py
from pathlib import Path
from typing import Dict, List, Optional
import re
from ..domain.eeg_recording import SeizureWindow

class SummaryParser:
    """Parsea archivos chbXX-summary.txt"""
    
    @staticmethod
    def parse(summary_path: Path) -> Dict[str, List[SeizureWindow]]:
        """
        Parsea un archivo summary
        
        Returns:
            Dict[file_name, List[SeizureWindow]]
        """
        if not summary_path or not summary_path.exists():
            return {}
        
        txt = summary_path.read_text(errors="ignore").splitlines()
        windows_by_file: Dict[str, List[SeizureWindow]] = {}
        current_file = None
        pending_starts = []
        
        for line in txt:
            line = line.strip()
            
            # Detectar nombre de archivo
            m = re.match(r"File Name:\s*(\S+)", line)
            if m:
                current_file = m.group(1)
                windows_by_file.setdefault(current_file, [])
                pending_starts = []
                continue
            
            # Detectar inicio de crisis
            m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)\s*seconds?", line, re.I)
            if m and current_file:
                pending_starts.append(float(m.group(1)))
                continue
            
            # Detectar fin de crisis
            m = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)\s*seconds?", line, re.I)
            if m and current_file and pending_starts:
                start = pending_starts.pop(0)
                end = float(m.group(1))
                windows_by_file[current_file].append(
                    SeizureWindow(start_sec=start, end_sec=end)
                )
        
        return windows_by_file