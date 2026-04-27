from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, welch
import matplotlib.pyplot as plt

from config.base_config import BaseConfig
from shared.domain.eeg_recording import EEGRecording
from shared.data_access.edf_loader import EDFLoader
from shared.data_access.summary_parser import SummaryParser
from features.signal_preprocessing.pipeline import SignalPreprocessor
from features.power_analysis.band_power import BandPowerAnalyzer
from features.spike_detection.detector import SpikeDetector
from features.connectivity_analysis.coherence import CoherenceAnalyzer
from features.connectivity_analysis.phase_lag_index import PLIAnalyzer
from features.topographic_mapping.topomap_generator import TopomapGenerator
from reporting.html_builder import ReportBuilder
from shared.domain.feature_extractor import FeatureExtractor


class PipelineRunner:
    """
    Orquestador del pipeline completo de análisis EEG.
    CORREGIDO: Flujo de datos de spikes y configuración de bandas.
    """
    
    def __init__(self, config: BaseConfig):
        self.config = config
        
        # Inicializar componentes
        self.preprocessor = SignalPreprocessor(config.preprocessing)
        self.bandpower_analyzer = BandPowerAnalyzer(config.bands)
        self.spike_detector = SpikeDetector(config.spike_detection)
        self.topomap_generator = TopomapGenerator()
        
        # CORRECCIÓN 1: Pasar configuración de bandas al extractor
        self.feature_extractor = FeatureExtractor(bands_config=config.bands)
        
        # Crear directorio de resultados
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_for_subject(self, subject_id: str):
        """Procesa todos los archivos de un sujeto"""
        print(f"\n{'='*60}")
        print(f"Procesando sujeto: {subject_id}")
        print(f"{'='*60}")
        
        subject_dir = self.config.results_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        reports_dir = subject_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        file_results = []
        all_features = []
        
        # 1. Buscar archivo summary
        summary_path = self._find_summary(subject_id)
        if not summary_path:
            print(f"❌ No se encontró archivo summary para {subject_id}")
            return
        
        print(f"📋 Summary encontrado: {summary_path}")
        
        # 2. Parsear summary
        seizure_info = SummaryParser.parse(summary_path)
        
        # 3. Obtener archivos EDF
        subject_data_dir = self.config.base_dir / subject_id
        edf_files = sorted(subject_data_dir.glob("*.edf"))
        
        # Filtrar por target_files
        if hasattr(self.config, 'target_files') and self.config.target_files:
            target_names = [Path(f).name for f in self.config.target_files]
            print(f"🎯 Filtrando por target_files: {target_names}")
            edf_files = [f for f in edf_files if f.name in target_names]
        
        print(f"📂 Archivos a procesar: {len(edf_files)}")
        
        for edf_path in edf_files:
            try:
                file_id = edf_path.name
                print(f"\n  📄 Procesando: {file_id}")
                
                seizure_windows = seizure_info.get(file_id, [])
                if seizure_windows:
                    print(f"     ⚡ Crisis detectadas: {len(seizure_windows)}")
                
                # Cargar y Preprocesar
                recording = EDFLoader.load(edf_path, seizure_windows)
                recording.raw = self.preprocessor.preprocess(recording.raw)
                
                # === CORRECCIÓN 2: Detectar Spikes ANTES del análisis ===
                # Detectamos en todos los canales y guardamos la lista completa
                all_spike_events = self._detect_all_spikes(recording)
                print(f"     ⚡ Total spikes detectados: {len(all_spike_events)}")
                
                # Análisis global (Pasa los spikes para graficar)
                self._run_global_analysis(recording, reports_dir, all_spike_events)
                
                # Análisis ictal
                ictal_results = []
                if recording.has_seizures():
                    ictal_results = self._run_ictal_analysis(recording, reports_dir)
                
                # === CORRECCIÓN 3: Extracción con Spikes ===
                features = self.feature_extractor.extract_features(
                    recording, 
                    spike_events=all_spike_events
                )
                all_features.append(features)
                
                # Guardar resultado preliminar
                file_results.append({
                    "file": file_id,
                    "fs": recording.sampling_rate,
                    "duration_sec": recording.duration_sec,
                    "n_channels": recording.n_channels,
                    "has_seizures": recording.has_seizures(),
                    "n_seizures": len(seizure_windows),
                    "n_spikes": len(all_spike_events), # Dato útil para el reporte HTML
                    "ictal_summaries": ictal_results
                })
                
                print(f"  ✅ {file_id} completado")
                
            except Exception as e:
                print(f"  ❌ Error procesando {edf_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generar reporte HTML
        if file_results:
            report_builder = ReportBuilder(self.config)
            report_path = report_builder.build(subject_id, file_results, reports_dir)
            print(f"\n📄 Reporte generado: {report_path}")
        
        # Exportar features a CSV
        if all_features:
            df_features = pd.DataFrame(all_features)
            csv_path = reports_dir / f"{subject_id}_features.csv"
            df_features.to_csv(csv_path, index=False)
            print(f"📊 Datos numéricos exportados: {csv_path}")

    def _find_summary(self, subject_id: str) -> Path:
        candidates = [
            self.config.base_dir / subject_id / f"{subject_id}-summary.txt",
            self.config.base_dir / f"{subject_id}-summary.txt",
        ]
        if hasattr(self.config, 'summary_fallback_dir') and self.config.summary_fallback_dir:
            if self.config.summary_fallback_dir.exists():
                candidates.append(self.config.summary_fallback_dir / f"{subject_id}-summary.txt")
        
        for path in candidates:
            if path.exists(): return path
        return None

    def _detect_all_spikes(self, recording: EEGRecording) -> List[dict]:
        """
        Detecta spikes en todos los canales y retorna una lista plana de eventos.
        Usado para alimentar FeatureExtractor.
        """
        fs = recording.sampling_rate
        data = recording.data
        all_events = []
        
        # Iterar todos los canales
        for i, ch_name in enumerate(recording.channels):
            peaks, stats = self.spike_detector.detect(data[i], fs)
            
            for p in peaks:
                all_events.append({
                    'channel': ch_name,
                    'time_sec': p / fs,
                    'sample_idx': p,
                    'max_amplitude': data[i][p] # Amplitud en el pico
                })
        
        return all_events

    def _run_global_analysis(self, recording: EEGRecording, output_dir: Path, spike_events: List[dict]):
        """Análisis global del archivo"""
        file_id = recording.file_id
        
        # 1. PSD
        self._plot_psd(recording, output_dir)
        
        # 2. Espectrograma
        self._plot_spectrogram(recording, output_dir)
        
        # 3. Evolución temporal
        bp_df = self.bandpower_analyzer.compute_temporal_evolution(
            recording.data, recording.sampling_rate
        )
        self._plot_bandpower_evolution(bp_df, file_id, output_dir)
        
        # 4. Graficar Spikes (usando los ya detectados)
        self._plot_spikes_visualization(recording, spike_events, output_dir)

    def _plot_spikes_visualization(self, recording: EEGRecording, spike_events: List[dict], output_dir: Path):
        """Grafica los spikes detectados (solo canales preferidos)"""
        fs = recording.sampling_rate
        data = recording.data
        
        pref_idx = [i for i, ch in enumerate(recording.channels) 
                   if ch in self.config.visualization.preferred_channels]
        if not pref_idx:
            pref_idx = list(range(min(4, recording.n_channels)))
        
        for i in pref_idx:
            ch_name = recording.channels[i]
            # Filtrar eventos de este canal
            ch_spikes = [e['time_sec'] for e in spike_events if e['channel'] == ch_name]
            
            plt.figure(figsize=(12, 3))
            time_axis = np.arange(len(data[i])) / fs
            plt.plot(time_axis, data[i], 'b-', linewidth=0.5)
            
            if ch_spikes:
                spike_indices = (np.array(ch_spikes) * fs).astype(int)
                # Asegurar índices válidos
                spike_indices = spike_indices[spike_indices < len(data[i])]
                plt.plot(np.array(ch_spikes), data[i][spike_indices], "rx", markersize=8, 
                        label=f"Spikes (n={len(ch_spikes)})")
            
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud (µV)")
            plt.title(f"Detección de Spikes — {recording.file_id} — {ch_name}")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            ch_safe = ch_name.replace('/', '-').replace(' ', '_')
            plt.savefig(output_dir / f"{recording.file_id}_spikes_{ch_safe}.png",
                       dpi=self.config.visualization.figure_dpi)
            plt.close()

    # ... (El resto de métodos _plot_psd, _plot_spectrogram, _run_ictal_analysis se mantienen igual)
    # ... Solo asegúrate de copiar los métodos helpers del código anterior si no están aquí explícitamente.
    # ... Por brevedad, asumo que mantienes los métodos de ploteo que ya tenías bien.

    def _plot_psd(self, recording: EEGRecording, output_dir: Path):
        # (Mismo código que antes)
        fs = recording.sampling_rate
        data = recording.data
        pref_idx = [i for i, ch in enumerate(recording.channels) if ch in self.config.visualization.preferred_channels]
        if not pref_idx: pref_idx = list(range(min(4, recording.n_channels)))
        plt.figure(figsize=(10, 5))
        for i in pref_idx:
            f, Pxx = welch(data[i], fs=fs, nperseg=int(fs*2), noverlap=int(fs))
            plt.semilogy(f, Pxx, label=recording.channels[i])
        plt.xlim(0.5, 45)
        plt.xlabel("Hz"); plt.ylabel("PSD")
        plt.title(f"PSD — {recording.file_id}")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{recording.file_id}_psd.png", dpi=self.config.visualization.figure_dpi)
        plt.close()

    def _plot_spectrogram(self, recording: EEGRecording, output_dir: Path):
        # (Mismo código que antes)
        fs = recording.sampling_rate
        data = recording.data
        pref_idx = [i for i, ch in enumerate(recording.channels) if ch in self.config.visualization.preferred_channels]
        ch_idx = pref_idx[0] if pref_idx else 0
        win = int(fs * self.config.visualization.spectrogram_window_sec)
        nover = int(fs * self.config.visualization.spectrogram_overlap_sec)
        f, t, Sxx = spectrogram(data[ch_idx], fs=fs, nperseg=win, noverlap=nover)
        plt.figure(figsize=(12, 5))
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading="auto", cmap='viridis')
        plt.ylim(0, 45)
        plt.title(f"Espectrograma — {recording.file_id}")
        plt.savefig(output_dir / f"{recording.file_id}_spectrogram.png", dpi=self.config.visualization.figure_dpi)
        plt.close()

    def _plot_bandpower_evolution(self, bp_df, file_id: str, output_dir: Path):
        # (Mismo código que antes)
        plt.figure(figsize=(12, 5))
        colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'orange', 'beta': 'red', 'gamma': 'purple'}
        for band in self.config.bands.to_dict().keys():
            col = f"{band}_rel"
            if col in bp_df.columns:
                plt.plot(bp_df["time_sec"], bp_df[col], label=band, color=colors.get(band, 'gray'))
        plt.title(f"Evolución Bandpower — {file_id}")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(output_dir / f"{file_id}_bandpower_evolution.png", dpi=self.config.visualization.figure_dpi)
        plt.close()

    def _run_ictal_analysis(self, recording: EEGRecording, output_dir: Path) -> List[Dict]:
        results = []
        fs = recording.sampling_rate
        data = recording.data
        for idx, window in enumerate(recording.seizure_windows):
            a = int(window.start_sec * fs)
            b = int(window.end_sec * fs)
            if b <= a or b > data.shape[1]:
                continue
            seg_data = data[:, a:b]
            self._plot_ictal_spectrogram(seg_data, recording, idx, output_dir)
            self._generate_ictal_topomaps(seg_data, recording, idx, output_dir)
            self._compute_ictal_connectivity(seg_data, fs, recording, idx, output_dir)
            results.append(
                {
                    "index": idx,
                    "start_s": float(window.start_sec),
                    "end_s": float(window.end_sec),
                    "duration_s": float(window.duration_sec),
                }
            )
        return results

    def _plot_ictal_spectrogram(self, seg_data, recording, idx, output_dir):
        fs = recording.sampling_rate
        pref_idx = [i for i, ch in enumerate(recording.channels) if ch in self.config.visualization.preferred_channels]
        ch_idx = pref_idx[0] if pref_idx else 0

        win = int(fs * self.config.visualization.spectrogram_window_sec)
        nover = int(fs * self.config.visualization.spectrogram_overlap_sec)
        if win <= 1:
            win = min(256, seg_data.shape[1])
        nover = min(nover, max(0, win - 1))

        f, t, Sxx = spectrogram(seg_data[ch_idx], fs=fs, nperseg=win, noverlap=nover)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="viridis")
        plt.ylim(0, 45)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        plt.title(f"Espectrograma Ictal #{idx + 1} — {recording.file_id}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{recording.file_id}_ictal{idx}_spectrogram.png", dpi=self.config.visualization.figure_dpi)
        plt.close()

    def _generate_ictal_topomaps(self, seg_data, recording, idx, output_dir):
        fs = recording.sampling_rate
        nperseg = min(int(fs * 2), seg_data.shape[1])
        if nperseg < 8:
            return

        freqs, psd = welch(seg_data, fs=fs, nperseg=nperseg, axis=1)

        for band_name, (lo, hi) in self.config.bands.to_dict().items():
            if lo is None or hi is None or hi <= lo:
                continue

            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                continue

            band_power = np.trapz(psd[:, mask], freqs[mask], axis=1)
            topomap_values = np.log10(band_power + 1e-12)

            self.topomap_generator.generate(
                data_vec=topomap_values,
                ch_names=recording.channels,
                output_path=output_dir / f"{recording.file_id}_ictal{idx}_topomap_{band_name}.png",
                title=f"Topomap {band_name.upper()} — Ictal #{idx + 1}",
                dpi=self.config.visualization.figure_dpi,
            )

    def _compute_ictal_connectivity(self, seg_data, fs, recording, idx, output_dir):
        coh_analyzer = CoherenceAnalyzer(band=self.config.connectivity.coherence_band)
        coh_matrix = coh_analyzer.compute_matrix(seg_data, fs)
        self._plot_connectivity_heatmap(
            matrix=coh_matrix,
            ch_names=recording.channels,
            title=f"Coherencia — Ictal #{idx + 1} — {recording.file_id}",
            output_path=output_dir / f"{recording.file_id}_ictal{idx}_coherence_heatmap.png",
        )

        if getattr(self.config.connectivity, "compute_pli", False):
            for band_name, band_range in self.config.connectivity.pli_bands.items():
                pli_analyzer = PLIAnalyzer(band=band_range)
                pli_matrix = pli_analyzer.compute_matrix(seg_data, fs)
                self._plot_connectivity_heatmap(
                    matrix=pli_matrix,
                    ch_names=recording.channels,
                    title=f"PLI {band_name.upper()} — Ictal #{idx + 1} — {recording.file_id}",
                    output_path=output_dir / f"{recording.file_id}_ictal{idx}_pli_{band_name}_heatmap.png",
                )

    def _plot_connectivity_heatmap(self, matrix, ch_names, title, output_path):
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, vmin=0, vmax=1, interpolation="nearest", cmap="viridis")
        plt.colorbar()
        if ch_names and len(ch_names) <= 20:
            ticks = np.arange(len(ch_names))
            plt.xticks(ticks, ch_names, rotation=90, fontsize=7)
            plt.yticks(ticks, ch_names, fontsize=7)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.visualization.figure_dpi)
        plt.close()