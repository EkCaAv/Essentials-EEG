# config/experiments/exp_pediatric.py
"""
Configuración para el Estudio Pediátrico (< 18 años).
Excluye sujetos adultos (chb04, chb18, chb19).
"""

from pathlib import Path
from config.base_config import BaseConfig
from config.subject_metadata import get_subjects

class PediatricStudyConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # FILTRO: Estrictamente menores de 18 años
        self.subjects = get_subjects(max_age=17.9)
        
        # Carpeta de resultados específica para esta población
        self.results_dir = Path("./results_pediatric_study").resolve()
        
        self.experiment_name = "thesis_pediatric_population"
        
        # Imprimir resumen para el registro
        print(f"\n🧸 CONFIGURACIÓN PEDIÁTRICA (< 18 años)")
        print(f"   Total sujetos incluidos: {len(self.subjects)}")
        print(f"   Sujetos: {self.subjects}")
        
        # Opcional: Si quieres procesar SOLO un archivo de prueba de cada uno para verificar
        # self.target_files = [f"{s}_03.edf" for s in self.subjects] # Descomentar para prueba rápida

CONFIG = PediatricStudyConfig()