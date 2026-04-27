# features/seizure_characterization/summary_builder.py
"""
Constructor de resúmenes de crisis
"""

import pandas as pd
from typing import List, Dict
from pathlib import Path
from .ictal_analyzer import IctalCharacteristics

class SeizureSummaryBuilder:
    """
    VERTICAL SLICE: Constructor de resúmenes de crisis
    
    Responsabilidad: Generar reportes tabulares de características de crisis
    """
    
    def __init__(self):
        self.characteristics: List[IctalCharacteristics] = []
    
    def add_seizure(self, characteristics: IctalCharacteristics):
        """Agrega características de una crisis"""
        self.characteristics.append(characteristics)
    
    def build_dataframe(self) -> pd.DataFrame:
        """Construye DataFrame con todas las crisis"""
        if not self.characteristics:
            return pd.DataFrame()
        
        rows = [char.to_dict() for char in self.characteristics]
        return pd.DataFrame(rows)
    
    def save_csv(self, output_path: Path):
        """Guarda resumen en CSV"""
        df = self.build_dataframe()
        df.to_csv(output_path, index=False)
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas agregadas de todas las crisis"""
        df = self.build_dataframe()
        
        if df.empty:
            return {}
        
        stats = {
            "total_seizures": len(df),
            "avg_duration_sec": df['duration_sec'].mean(),
            "std_duration_sec": df['duration_sec'].std(),
            "avg_dominant_freq_hz": df['dominant_frequency_hz'].mean(),
            "most_common_trend": df['power_trend'].mode()[0] if not df.empty else None,
            "avg_coherence": df['mean_coherence'].mean(),
            
            # Por banda
            "avg_delta": df['avg_delta_power'].mean(),
            "avg_theta": df['avg_theta_power'].mean(),
            "avg_alpha": df['avg_alpha_power'].mean(),
            "avg_beta": df['avg_beta_power'].mean(),
            "avg_gamma": df['avg_gamma_power'].mean(),
        }
        
        return stats
    
    def generate_text_report(self) -> str:
        """Genera reporte textual"""
        stats = self.get_statistics()
        
        if not stats:
            return "No hay crisis analizadas."
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║          RESUMEN DE CARACTERÍSTICAS DE CRISIS            ║
╚═══════════════════════════════════════════════════════════╝

📊 ESTADÍSTICAS GENERALES
─────────────────────────────────────────────────────────────
  • Total de crisis analizadas: {stats['total_seizures']}
  • Duración promedio: {stats['avg_duration_sec']:.1f} ± {stats['std_duration_sec']:.1f} s
  • Frecuencia dominante promedio: {stats['avg_dominant_freq_hz']:.1f} Hz
  • Tendencia más común: {stats['most_common_trend']}
  • Coherencia promedio: {stats['avg_coherence']:.3f}

🌊 POTENCIA POR BANDAS (Promedio)
─────────────────────────────────────────────────────────────
  • Delta (0.5-4 Hz):   {stats['avg_delta']:.2e}
  • Theta (4-8 Hz):     {stats['avg_theta']:.2e}
  • Alpha (8-13 Hz):    {stats['avg_alpha']:.2e}
  • Beta (13-30 Hz):    {stats['avg_beta']:.2e}
  • Gamma (30-45 Hz):   {stats['avg_gamma']:.2e}
"""
        return report