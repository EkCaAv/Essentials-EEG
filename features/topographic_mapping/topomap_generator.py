# features/topographic_mapping/topomap_generator.py
"""
Generador de mapas topográficos para derivaciones bipolares
Compatible con MNE 1.0+
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings


class TopomapGenerator:
    """
    VERTICAL SLICE: Generación de mapas topográficos
    
    Responsabilidad: Crear topomaps para derivaciones bipolares
    Compatible con versiones modernas de MNE (1.0+)
    """
    
    def __init__(self):
        """Inicializa el generador cargando posiciones estándar"""
        # Cargar montage estándar 10-20
        self.std_montage = mne.channels.make_standard_montage("standard_1020")
        self.std_positions = self.std_montage.get_positions()["ch_pos"]
    
    @staticmethod
    def _parse_bipolar_channel(ch: str) -> Optional[Tuple[str, str]]:
        """
        Parsea un nombre de canal bipolar, manejando sufijos numéricos
        
        Ejemplos:
            "F7-T7"     -> ("F7", "T7")
            "T8-P8-0"   -> ("T8", "P8")  # Ignora sufijo
            "FZ-CZ"     -> ("FZ", "CZ")
        
        Returns:
            Tupla (electrodo1, electrodo2) o None si no es válido
        """
        ch_clean = ch.strip().upper()
        
        if "-" not in ch_clean:
            return None
        
        parts = ch_clean.split("-")
        
        if len(parts) == 2:
            return (parts[0], parts[1])
        
        elif len(parts) == 3:
            # Caso con sufijo: X-Y-0 o X-Y-1
            try:
                int(parts[2])
                return (parts[0], parts[1])
            except ValueError:
                return None
        
        return None
    
    def _find_electrode_position(self, electrode: str) -> Optional[np.ndarray]:
        """
        Busca la posición de un electrodo en el sistema 10-20
        Intenta variaciones de nombre comunes
        """
        # Variaciones a intentar
        variations = [
            electrode,
            electrode.upper(),
            electrode.capitalize(),
            electrode.lower(),
            electrode.replace("FP", "Fp"),
            electrode.replace("FZ", "Fz"),
            electrode.replace("CZ", "Cz"),
            electrode.replace("PZ", "Pz"),
            electrode.replace("OZ", "Oz"),
        ]
        
        for var in variations:
            if var in self.std_positions:
                return self.std_positions[var]
        
        return None
    
    def _compute_bipolar_positions(self, ch_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Calcula posiciones para canales bipolares (punto medio)
        
        Returns:
            Dict[channel_name, xyz_position]
        """
        positions = {}
        processed_pairs = set()
        
        for ch in ch_names:
            parsed = self._parse_bipolar_channel(ch)
            
            if parsed is None:
                continue
            
            a, b = parsed
            canonical = f"{a}-{b}"
            
            if canonical in processed_pairs:
                continue
            
            pos_a = self._find_electrode_position(a)
            pos_b = self._find_electrode_position(b)
            
            if pos_a is not None and pos_b is not None:
                midpoint = (pos_a + pos_b) / 2.0
                positions[ch] = midpoint
                processed_pairs.add(canonical)
        
        return positions
    
    def generate(self, 
                 data_vec: np.ndarray, 
                 ch_names: List[str],
                 output_path: Path, 
                 title: str, 
                 dpi: int = 130):
        """
        Genera y guarda un topomap
        
        Args:
            data_vec: Vector de datos (un valor por canal)
            ch_names: Nombres de canales
            output_path: Ruta de salida
            title: Título del mapa
            dpi: Resolución
        """
        try:
            # Calcular posiciones bipolares
            bipolar_positions = self._compute_bipolar_positions(ch_names)
            
            if not bipolar_positions:
                self._generate_error_figure(
                    output_path, title, dpi,
                    f"No se pudieron mapear canales bipolares\n\n"
                    f"Canales disponibles: {len(ch_names)}\n"
                    f"Canales mapeables: 0"
                )
                return
            
            # Filtrar datos para canales válidos
            valid_indices = [i for i, ch in enumerate(ch_names) if ch in bipolar_positions]
            use_names = [ch_names[i] for i in valid_indices]
            use_data = data_vec[valid_indices]
            
            # Crear array de posiciones XY (solo usamos x,y para topomap 2D)
            pos_array = np.array([bipolar_positions[ch][:2] for ch in use_names])
            
            # Generar topomap usando matplotlib directamente
            fig, ax = plt.subplots(figsize=(5.6, 4.6))
            
            # Crear topomap con interpolación
            self._plot_topomap_simple(ax, use_data, pos_array, use_names)
            
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi)
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error generando topomap para {title}: {str(e)}")
            self._generate_error_figure(output_path, title, dpi, str(e))
    
    def _plot_topomap_simple(self, ax, data: np.ndarray, pos: np.ndarray, ch_names: List[str]):
        """
        Genera un topomap simple usando scatter + interpolación
        
        Args:
            ax: Axes de matplotlib
            data: Datos a graficar
            pos: Posiciones XY de los canales
            ch_names: Nombres de los canales
        """
        from scipy.interpolate import griddata
        
        # Normalizar posiciones
        pos_norm = pos.copy()
        pos_norm[:, 0] = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min() + 1e-10)
        pos_norm[:, 1] = (pos[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min() + 1e-10)
        
        # Crear grid para interpolación
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
        
        # Interpolar datos
        try:
            grid_data = griddata(pos_norm, data, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        except Exception:
            grid_data = griddata(pos_norm, data, (grid_x, grid_y), method='linear', fill_value=np.nan)
        
        # Crear máscara circular
        center = (0.5, 0.5)
        radius = 0.5
        mask = (grid_x - center[0])**2 + (grid_y - center[1])**2 > radius**2
        grid_data[mask] = np.nan
        
        # Graficar
        im = ax.contourf(grid_x, grid_y, grid_data, levels=20, cmap='RdBu_r')
        
        # Dibujar círculo de la cabeza
        circle = plt.Circle(center, radius, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # Marcar posiciones de electrodos
        ax.scatter(pos_norm[:, 0], pos_norm[:, 1], c='black', s=20, zorder=5)
        
        # Agregar colorbar
        plt.colorbar(im, ax=ax, label='Potencia (log)')
        
        # Configurar ejes
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Agregar marcadores de orientación
        ax.text(0.5, 1.05, 'Anterior', ha='center', va='bottom', fontsize=8)
        ax.text(0.5, -0.05, 'Posterior', ha='center', va='top', fontsize=8)
        ax.text(-0.05, 0.5, 'L', ha='right', va='center', fontsize=8)
        ax.text(1.05, 0.5, 'R', ha='left', va='center', fontsize=8)
    
    def _generate_error_figure(self, output_path: Path, title: str, dpi: int, message: str):
        """Genera una figura de error/advertencia"""
        plt.figure(figsize=(5.6, 4.6))
        plt.text(0.5, 0.5, message,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                wrap=True)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()