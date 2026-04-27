# reporting/html_builder.py
"""
Generador de reportes HTML para análisis EEG
"""

from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ReportBuilder:
    """
    VERTICAL SLICE: Generación de reportes HTML
    
    Responsabilidad: Crear reportes HTML consolidados con resultados del análisis
    """
    
    def __init__(self, config):
        """
        Inicializa el generador de reportes
        
        Args:
            config: Configuración del pipeline
        """
        self.config = config
    
    def build(self, subject_id: str, file_results: List[Dict], output_dir: Path) -> Path:
        """
        Construye el reporte HTML para un sujeto
        
        Args:
            subject_id: Identificador del sujeto
            file_results: Lista de resultados por archivo
            output_dir: Directorio de salida
        
        Returns:
            Path al archivo HTML generado
        """
        # Generar contenido HTML
        html_content = self._build_html(subject_id, file_results, output_dir)
        
        # Guardar archivo
        report_path = output_dir / f"{subject_id}_report.html"
        report_path.write_text(html_content, encoding='utf-8')
        
        return report_path
    
    def _build_html(self, subject_id: str, file_results: List[Dict], output_dir: Path) -> str:
        """
        Genera el contenido HTML completo
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte EEG - {subject_id}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <header>
        <h1>🧠 Reporte de Análisis EEG</h1>
        <p class="subtitle">Sujeto: <strong>{subject_id}</strong></p>
        <p class="timestamp">Generado: {timestamp}</p>
    </header>
    
    <nav class="toc">
        <h2>📋 Contenido</h2>
        <ul>
"""
        # Table of contents
        for result in file_results:
            file_id = self._extract_file_id(result.get('file', 'unknown'))
            has_seizures = result.get('has_seizures', False)
            seizure_icon = "🔴" if has_seizures else "🟢"
            html += f'            <li><a href="#{file_id}">{seizure_icon} {file_id}</a></li>\n'
        
        html += """        </ul>
    </nav>
    
    <main>
"""
        
        # Secciones por archivo
        for result in file_results:
            html += self._build_file_section(result, output_dir)
        
        # Footer
        html += f"""
    </main>
    
    <footer>
        <p>Pipeline EEG - Arquitectura Vertical Slices</p>
        <p>Experimento: {self.config.experiment_name}</p>
    </footer>
</body>
</html>
"""
        return html
    
    def _extract_file_id(self, file_path) -> str:
        """
        Extrae solo el nombre del archivo de una ruta
        
        Args:
            file_path: Puede ser str o Path, ruta completa o solo nombre
        
        Returns:
            Solo el nombre del archivo (sin extensión de ruta)
        """
        if file_path is None:
            return "unknown"
        
        # Convertir a string si es Path
        file_str = str(file_path)
        
        # Usar Path para extraer solo el nombre
        return Path(file_str).name
    
    def _build_file_section(self, result: Dict, output_dir: Path) -> str:
        """
        Construye la sección HTML para un archivo
        """
        # Extraer file_id de forma segura
        file_id = self._extract_file_id(result.get('file', 'unknown'))
        
        has_seizures = result.get('has_seizures', False)
        duration = result.get('duration_sec', 0)
        fs = result.get('fs', 0)
        n_channels = result.get('n_channels', 0)
        
        seizure_badge = '<span class="badge seizure">CON CRISIS</span>' if has_seizures else '<span class="badge normal">SIN CRISIS</span>'
        
        html = f"""
        <section id="{file_id}" class="file-section">
            <h2>📁 {file_id} {seizure_badge}</h2>
            
            <div class="metadata">
                <span>⏱️ Duración: {duration:.1f}s</span>
                <span>📊 Fs: {fs:.0f} Hz</span>
                <span>📡 Canales: {n_channels}</span>
            </div>
            
            <h3>🔍 Análisis Global</h3>
            <div class="gallery">
"""
        
        # Imágenes globales
        global_images = [
            ('psd', 'Densidad Espectral de Potencia'),
            ('spectrogram', 'Espectrograma'),
            ('bandpower_evolution', 'Evolución de Bandpower'),
        ]
        
        for suffix, title in global_images:
            img_name = f"{file_id}_{suffix}.png"
            img_path = output_dir / img_name
            if img_path.exists():
                html += f"""
                <div class="image-card">
                    <img src="{img_name}" alt="{title}">
                    <p>{title}</p>
                </div>
"""
        
        # Imágenes de spikes
        spike_images = list(output_dir.glob(f"{file_id}_spikes_*.png"))
        if spike_images:
            html += """
            </div>
            <h3>⚡ Detección de Spikes</h3>
            <div class="gallery">
"""
            for img_path in sorted(spike_images):
                ch_name = img_path.stem.replace(f"{file_id}_spikes_", "")
                html += f"""
                <div class="image-card">
                    <img src="{img_path.name}" alt="Spikes {ch_name}">
                    <p>Spikes - {ch_name}</p>
                </div>
"""
        
        html += "            </div>\n"
        
        # Análisis ictal si hay crisis
        ictal_summaries = result.get('ictal_summaries', [])
        if ictal_summaries:
            html += """
            <h3>🔴 Análisis Ictal</h3>
"""
            for ictal in ictal_summaries:
                idx = ictal.get('index', 0)
                start_s = ictal.get('start_s', 0)
                end_s = ictal.get('end_s', 0)
                duration_s = ictal.get('duration_s', 0)
                
                html += f"""
            <div class="ictal-section">
                <h4>Crisis #{idx + 1}</h4>
                <div class="metadata">
                    <span>⏱️ Inicio: {start_s:.1f}s</span>
                    <span>⏱️ Fin: {end_s:.1f}s</span>
                    <span>⏱️ Duración: {duration_s:.1f}s</span>
                </div>
                
                <div class="gallery">
"""
                
                # Espectrograma ictal
                spec_path = output_dir / f"{file_id}_ictal{idx}_spectrogram.png"
                if spec_path.exists():
                    html += f"""
                    <div class="image-card">
                        <img src="{spec_path.name}" alt="Espectrograma Ictal">
                        <p>Espectrograma Ictal</p>
                    </div>
"""
                
                # Topomaps
                topo_images = list(output_dir.glob(f"{file_id}_ictal{idx}_topomap_*.png"))
                for img_path in sorted(topo_images):
                    band = img_path.stem.split('_')[-1]
                    html += f"""
                    <div class="image-card">
                        <img src="{img_path.name}" alt="Topomap {band}">
                        <p>Topomap - {band.upper()}</p>
                    </div>
"""
                
                html += """
                </div>
                
                <h5>Conectividad</h5>
                <div class="gallery">
"""
                
                # Coherencia
                coh_path = output_dir / f"{file_id}_ictal{idx}_coherence_heatmap.png"
                if coh_path.exists():
                    html += f"""
                    <div class="image-card">
                        <img src="{coh_path.name}" alt="Coherencia">
                        <p>Coherencia</p>
                    </div>
"""
                
                # PLI
                pli_images = list(output_dir.glob(f"{file_id}_ictal{idx}_pli_*_heatmap.png"))
                for img_path in sorted(pli_images):
                    band = img_path.stem.replace(f"{file_id}_ictal{idx}_pli_", "").replace("_heatmap", "")
                    html += f"""
                    <div class="image-card">
                        <img src="{img_path.name}" alt="PLI {band}">
                        <p>PLI - {band.upper()}</p>
                    </div>
"""
                
                html += """
                </div>
            </div>
"""
        
        html += "        </section>\n"
        return html
    
    def _get_css(self) -> str:
        """Retorna los estilos CSS del reporte"""
        return """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.3em;
            opacity: 0.9;
        }
        
        .timestamp {
            font-size: 0.9em;
            opacity: 0.7;
            margin-top: 10px;
        }
        
        nav.toc {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        nav.toc h2 {
            margin-bottom: 15px;
            color: #667eea;
        }
        
        nav.toc ul {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        nav.toc li a {
            display: block;
            padding: 8px 15px;
            background: #f0f0f0;
            border-radius: 5px;
            text-decoration: none;
            color: #333;
            transition: all 0.3s;
        }
        
        nav.toc li a:hover {
            background: #667eea;
            color: white;
        }
        
        main {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .file-section {
            margin-bottom: 40px;
            padding-bottom: 40px;
            border-bottom: 2px solid #eee;
        }
        
        .file-section:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        .file-section h2 {
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .badge {
            font-size: 0.6em;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: normal;
        }
        
        .badge.seizure {
            background: #ff4757;
            color: white;
        }
        
        .badge.normal {
            background: #2ed573;
            color: white;
        }
        
        .metadata {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            padding: 10px 15px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.95em;
        }
        
        h3 {
            color: #667eea;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        h4 {
            color: #764ba2;
            margin: 20px 0 10px 0;
        }
        
        h5 {
            color: #555;
            margin: 15px 0 10px 0;
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 15px 0;
        }
        
        .image-card {
            background: #fafafa;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .image-card p {
            padding: 10px;
            text-align: center;
            font-weight: 500;
            color: #555;
        }
        
        .ictal-section {
            background: #fff5f5;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #ff4757;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            header {
                padding: 20px;
            }
            
            header h1 {
                font-size: 1.8em;
            }
            
            .gallery {
                grid-template-columns: 1fr;
            }
            
            .metadata {
                flex-direction: column;
                gap: 5px;
            }
        }
"""