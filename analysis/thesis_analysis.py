# analysis/thesis_analysis.py
"""
Análisis Estadístico y Visualización para Tesis
Estudio de características EEG en población pediátrica con epilepsia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para publicación
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 100

# Colores consistentes
COLORS = {
    'ictal': '#e74c3c',
    'interictal': '#3498db',
    'Delta': '#9b59b6',
    'Theta': '#2ecc71',
    'Alpha': '#f1c40f',
    'Beta': '#e74c3c',
    'Gamma': '#1abc9c'
}

BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']


class ThesisAnalyzer:
    """Analizador de datos para la tesis de epilepsia pediátrica"""
    
    def __init__(self, results_dir: str, output_dir: str = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "thesis_figures"
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.subjects_meta = self._load_metadata()
        
    def _load_metadata(self):
        """Carga metadatos de sujetos CHB-MIT"""
        return {
            "chb01": {"age": 11, "gender": "F"},
            "chb02": {"age": 11, "gender": "M"},
            "chb03": {"age": 14, "gender": "F"},
            "chb05": {"age": 7,  "gender": "F"},
            "chb06": {"age": 1,  "gender": "F"},
            "chb07": {"age": 14, "gender": "F"},
            "chb08": {"age": 3,  "gender": "M"},
            "chb09": {"age": 10, "gender": "F"},
            "chb10": {"age": 3,  "gender": "M"},
            "chb11": {"age": 12, "gender": "F"},
            "chb12": {"age": 2,  "gender": "F"},
            "chb13": {"age": 3,  "gender": "F"},
            "chb14": {"age": 9,  "gender": "F"},
            "chb15": {"age": 16, "gender": "M"},
            "chb16": {"age": 7,  "gender": "F"},
            "chb17": {"age": 12, "gender": "F"},
            "chb20": {"age": 6,  "gender": "F"},
            "chb21": {"age": 13, "gender": "F"},
            "chb22": {"age": 9,  "gender": "F"},
            "chb23": {"age": 6,  "gender": "F"},
            "chb24": {"age": 16, "gender": "N/A"},
        }
    
    def load_all_data(self):
        """Consolida todos los CSV de features en un DataFrame"""
        all_data = []
        
        for subject_dir in sorted(self.results_dir.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue
                
            subject_id = subject_dir.name
            
            # Buscar CSV de features
            reports_dir = subject_dir / "reports"
            if not reports_dir.exists():
                continue
                
            csv_files = list(reports_dir.glob(f"{subject_id}_features.csv"))
            if not csv_files:
                csv_files = list(reports_dir.glob("*_features.csv"))
            
            for csv_path in csv_files:
                try:
                    df_subject = pd.read_csv(csv_path)
                    df_subject['subject_id'] = subject_id
                    
                    # Agregar metadatos demográficos
                    meta = self.subjects_meta.get(subject_id, {"age": None, "gender": None})
                    df_subject['age'] = meta['age']
                    df_subject['gender'] = meta['gender']
                    
                    all_data.append(df_subject)
                    print(f"✅ {subject_id}: {len(df_subject)} registros")
                except Exception as e:
                    print(f"⚠️ Error cargando {csv_path}: {e}")
        
        if all_data:
            self.df = pd.concat(all_data, ignore_index=True)
            print(f"\n{'='*50}")
            print(f"📊 DATOS CARGADOS:")
            print(f"   - Registros totales: {len(self.df)}")
            print(f"   - Sujetos únicos: {self.df['subject_id'].nunique()}")
            print(f"   - Archivos con crisis: {self.df['has_seizures'].sum()}")
            print(f"   - Archivos sin crisis: {(~self.df['has_seizures']).sum()}")
            print(f"   - Columnas: {list(self.df.columns)}")
            print(f"{'='*50}\n")
        else:
            print("❌ No se encontraron datos")
            
        return self.df
    
    def descriptive_statistics(self):
        """Genera estadísticas descriptivas completas"""
        if self.df is None:
            self.load_all_data()
        
        print("\n" + "="*60)
        print("📈 ESTADÍSTICAS DESCRIPTIVAS - POBLACIÓN PEDIÁTRICA")
        print("="*60)
        
        # 1. Resumen de la muestra
        print("\n┌─────────────────────────────────────────────────┐")
        print("│ 1. RESUMEN DE LA MUESTRA                        │")
        print("└─────────────────────────────────────────────────┘")
        print(f"   Total de archivos analizados: {len(self.df)}")
        print(f"   Sujetos únicos: {self.df['subject_id'].nunique()}")
        print(f"   Archivos CON crisis: {self.df['has_seizures'].sum()} ({100*self.df['has_seizures'].mean():.1f}%)")
        print(f"   Archivos SIN crisis: {(~self.df['has_seizures']).sum()}")
        print(f"   Total de crisis detectadas: {self.df['n_seizures'].sum()}")
        
        # 2. Demografía
        print("\n┌─────────────────────────────────────────────────┐")
        print("│ 2. DEMOGRAFÍA                                   │")
        print("└─────────────────────────────────────────────────┘")
        df_demo = self.df.groupby('subject_id').agg({'age': 'first', 'gender': 'first'})
        print(f"   Edad media: {df_demo['age'].mean():.1f} ± {df_demo['age'].std():.1f} años")
        print(f"   Rango de edad: {df_demo['age'].min():.0f} - {df_demo['age'].max():.0f} años")
        print(f"   Mediana de edad: {df_demo['age'].median():.0f} años")
        print(f"\n   Distribución por género:")
        for g, c in df_demo['gender'].value_counts().items():
            pct = 100 * c / len(df_demo)
            print(f"     • {g}: {c} sujetos ({pct:.1f}%)")
        
        # 3. Características temporales
        print("\n┌─────────────────────────────────────────────────┐")
        print("│ 3. CARACTERÍSTICAS TEMPORALES                   │")
        print("└─────────────────────────────────────────────────┘")
        print(f"   RMS medio: {self.df['rms_mean'].mean():.4f} ± {self.df['rms_mean'].std():.4f}")
        print(f"   Curtosis media: {self.df['kurtosis_mean'].mean():.4f} ± {self.df['kurtosis_mean'].std():.4f}")
        print(f"   Asimetría media: {self.df['skewness_mean'].mean():.4f} ± {self.df['skewness_mean'].std():.4f}")
        print(f"   Duración media: {self.df['duration'].mean():.1f} ± {self.df['duration'].std():.1f} seg")
        
        # 4. Potencia por banda
        print("\n┌─────────────────────────────────────────────────┐")
        print("│ 4. POTENCIA ESPECTRAL POR BANDA                 │")
        print("└─────────────────────────────────────────────────┘")
        print("\n   Potencia Relativa (proporción del total):")
        for band in BANDS:
            col = f'rel_power_{band}'
            if col in self.df.columns:
                mean_val = self.df[col].mean() * 100  # Convertir a porcentaje
                std_val = self.df[col].std() * 100
                print(f"     • {band:6s}: {mean_val:5.1f}% ± {std_val:4.1f}%")
        
        # 5. Guardar tabla completa
        stats_df = self.df.describe()
        stats_path = self.output_dir / "tabla1_descriptive_statistics.csv"
        stats_df.to_csv(stats_path)
        
        # Tabla resumen para tesis
        summary_data = []
        for band in BANDS:
            col = f'rel_power_{band}'
            if col in self.df.columns:
                summary_data.append({
                    'Banda': band,
                    'Frecuencia (Hz)': f"{{'Delta': '0.5-4', 'Theta': '4-8', 'Alpha': '8-13', 'Beta': '13-30', 'Gamma': '30-40'}[band]}",
                    'Media (%)': f"{self.df[col].mean()*100:.2f}",
                    'DE (%)': f"{self.df[col].std()*100:.2f}",
                    'Min (%)': f"{self.df[col].min()*100:.2f}",
                    'Max (%)': f"{self.df[col].max()*100:.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "tabla2_band_power_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n💾 Tablas guardadas:")
        print(f"   - {stats_path}")
        print(f"   - {summary_path}")
        
        return stats_df
    
    def fig1_band_power_distribution(self):
        """Figura 1: Distribución de potencia relativa por banda"""
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        
        for idx, band in enumerate(BANDS):
            ax = axes[idx]
            col = f'rel_power_{band}'
            
            if col not in self.df.columns:
                ax.text(0.5, 0.5, f'No data for {band}', ha='center', va='center')
                continue
            
            data = self.df[col] * 100  # Convertir a porcentaje
            
            # Histograma con KDE
            ax.hist(data, bins=30, color=COLORS[band], alpha=0.7, 
                   edgecolor='white', density=True)
            
            # KDE superpuesto
            if len(data) > 10:
                data.plot.kde(ax=ax, color='black', linewidth=2)
            
            # Líneas de media y mediana
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Media: {mean_val:.1f}%')
            ax.axvline(median_val, color='blue', linestyle=':', linewidth=2,
                      label=f'Mediana: {median_val:.1f}%')
            
            ax.set_xlabel('Potencia Relativa (%)')
            ax.set_ylabel('Densidad')
            ax.set_title(f'{band} ({self._get_freq_range(band)})', fontweight='bold')
            ax.legend(fontsize=9)
        
        # Ocultar el sexto eje
        axes[5].set_visible(False)
        
        plt.suptitle('Figura 1: Distribución de Potencia Espectral por Banda\n'
                    'Población Pediátrica con Epilepsia (n={} archivos, {} sujetos)'.format(
                        len(self.df), self.df['subject_id'].nunique()),
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure('fig1_band_power_distribution')
        plt.show()
    
    def fig2_band_power_comparison(self):
        """Figura 2: Boxplot comparativo de bandas"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Preparar datos en formato largo
        data_long = []
        for band in BANDS:
            col = f'rel_power_{band}'
            if col in self.df.columns:
                for val in self.df[col] * 100:
                    data_long.append({'Banda': band, 'Potencia Relativa (%)': val})
        
        df_long = pd.DataFrame(data_long)
        
        # Crear boxplot
        palette = [COLORS[b] for b in BANDS]
        box = sns.boxplot(data=df_long, x='Banda', y='Potencia Relativa (%)',
                         palette=palette, ax=ax, width=0.6)
        
        # Agregar puntos individuales
        sns.stripplot(data=df_long, x='Banda', y='Potencia Relativa (%)',
                     color='black', alpha=0.2, size=3, ax=ax, jitter=True)
        
        # Agregar valores de media
        for i, band in enumerate(BANDS):
            col = f'rel_power_{band}'
            if col in self.df.columns:
                mean_val = self.df[col].mean() * 100
                ax.text(i, ax.get_ylim()[1] * 0.95, f'{mean_val:.1f}%', 
                       ha='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Banda de Frecuencia', fontsize=12)
        ax.set_ylabel('Potencia Relativa (%)', fontsize=12)
        ax.set_title('Figura 2: Comparación de Potencia Espectral entre Bandas\n'
                    'Población Pediátrica con Epilepsia',
                    fontsize=14, fontweight='bold')
        
        # Agregar rangos de frecuencia
        freq_labels = ['0.5-4 Hz', '4-8 Hz', '8-13 Hz', '13-30 Hz', '30-40 Hz']
        ax.set_xticklabels([f'{b}\n({f})' for b, f in zip(BANDS, freq_labels)])
        
        plt.tight_layout()
        self._save_figure('fig2_band_power_comparison')
        plt.show()
    
    def fig3_ictal_vs_interictal(self):
        """Figura 3: Comparación ictal vs interictal"""
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        
        results = []
        
        for idx, band in enumerate(BANDS):
            ax = axes[idx]
            col = f'rel_power_{band}'
            
            if col not in self.df.columns:
                continue
            
            # Separar grupos
            ictal = self.df[self.df['has_seizures'] == True][col] * 100
            interictal = self.df[self.df['has_seizures'] == False][col] * 100
            
            # Box plots
            bp = ax.boxplot([interictal.dropna(), ictal.dropna()],
                           labels=['Interictal\n(sin crisis)', 'Ictal\n(con crisis)'],
                           patch_artist=True, widths=0.6)
            
            bp['boxes'][0].set_facecolor(COLORS['interictal'])
            bp['boxes'][1].set_facecolor(COLORS['ictal'])
            
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            # Test estadístico (Mann-Whitney U)
            if len(ictal) > 0 and len(interictal) > 0:
                stat, pval = stats.mannwhitneyu(interictal.dropna(), ictal.dropna(), 
                                                alternative='two-sided')
                
                # Calcular tamaño del efecto (Cohen's d aproximado)
                cohens_d = (ictal.mean() - interictal.mean()) / np.sqrt(
                    (ictal.std()**2 + interictal.std()**2) / 2)
                
                # Significancia
                if pval < 0.001:
                    sig = '***'
                elif pval < 0.01:
                    sig = '**'
                elif pval < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                ax.text(0.5, 0.95, f'p = {pval:.4f} {sig}\nd = {cohens_d:.2f}', 
                       transform=ax.transAxes, ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                results.append({
                    'Banda': band,
                    'Media Interictal (%)': f"{interictal.mean():.2f}",
                    'Media Ictal (%)': f"{ictal.mean():.2f}",
                    'Diferencia (%)': f"{ictal.mean() - interictal.mean():.2f}",
                    'p-valor': f"{pval:.4f}",
                    'Cohen d': f"{cohens_d:.2f}",
                    'Significativo': 'Sí' if pval < 0.05 else 'No'
                })
            
            ax.set_ylabel('Potencia Relativa (%)')
            ax.set_title(f'{band}', fontweight='bold')
        
        # Ocultar el sexto eje
        axes[5].set_visible(False)
        
        # Leyenda en el espacio vacío
        axes[5].set_visible(True)
        axes[5].axis('off')
        axes[5].text(0.5, 0.7, 'Significancia:', fontweight='bold', ha='center', 
                    transform=axes[5].transAxes, fontsize=11)
        axes[5].text(0.5, 0.5, '*** p < 0.001\n** p < 0.01\n* p < 0.05\nns: no significativo',
                    ha='center', transform=axes[5].transAxes, fontsize=10)
        axes[5].text(0.5, 0.2, f'Interictal: n={len(interictal)}\nIctal: n={len(ictal)}',
                    ha='center', transform=axes[5].transAxes, fontsize=10)
        
        plt.suptitle('Figura 3: Comparación de Potencia Espectral\n'
                    'Archivos con Crisis (Ictal) vs Sin Crisis (Interictal)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Guardar tabla de resultados
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.output_dir / 'tabla3_ictal_vs_interictal.csv', index=False)
            print(f"💾 Tabla de comparación guardada")
        
        self._save_figure('fig3_ictal_vs_interictal')
        plt.show()
        
        return results
    
    def fig4_age_correlation(self):
        """Figura 4: Correlación edad vs características espectrales"""
        # Calcular medias por sujeto
        df_subj = self.df.groupby('subject_id').agg({
            'age': 'first',
            'gender': 'first',
            **{f'rel_power_{b}': 'mean' for b in BANDS}
        }).reset_index()
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        
        results = []
        
        for idx, band in enumerate(BANDS):
            ax = axes[idx]
            col = f'rel_power_{band}'
            
            x = df_subj['age']
            y = df_subj[col] * 100
            
            # Scatter plot con colores por género
            for gender, marker, color in [('F', 'o', '#e91e63'), ('M', 's', '#2196f3'), ('N/A', '^', '#9e9e9e')]:
                mask = df_subj['gender'] == gender
                ax.scatter(x[mask], y[mask], c=color, marker=marker, s=80, 
                          alpha=0.7, label=gender, edgecolor='white')
            
            # Línea de regresión
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.8)
            
            # Correlación de Pearson
            r, pval = stats.pearsonr(x, y)
            
            # Añadir texto
            sig = '*' if pval < 0.05 else ''
            ax.text(0.05, 0.95, f'r = {r:.3f}{sig}\np = {pval:.3f}', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Edad (años)')
            ax.set_ylabel('Potencia Relativa (%)')
            ax.set_title(f'{band}', fontweight='bold')
            
            if idx == 0:
                ax.legend(title='Género', loc='upper right')
            
            results.append({
                'Banda': band,
                'Correlación (r)': f"{r:.3f}",
                'p-valor': f"{pval:.3f}",
                'Significativo': 'Sí' if pval < 0.05 else 'No',
                'Tendencia': 'Aumenta' if r > 0 else 'Disminuye'
            })
        
        # Usar el sexto panel para información
        axes[5].axis('off')
        axes[5].text(0.5, 0.6, 'Desarrollo Cerebral y EEG:', fontweight='bold', 
                    ha='center', transform=axes[5].transAxes, fontsize=11)
        axes[5].text(0.5, 0.3, '• Delta/Theta ↓ con edad\n• Alpha ↑ con maduración\n• Línea: regresión lineal\n• *p < 0.05',
                    ha='center', transform=axes[5].transAxes, fontsize=10)
        
        plt.suptitle('Figura 4: Correlación entre Edad y Potencia Espectral\n'
                    f'Población Pediátrica (n={len(df_subj)} sujetos, {df_subj["age"].min():.0f}-{df_subj["age"].max():.0f} años)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Guardar resultados
        pd.DataFrame(results).to_csv(self.output_dir / 'tabla4_age_correlation.csv', index=False)
        
        self._save_figure('fig4_age_correlation')
        plt.show()
        
        return results
    
    def fig5_correlation_heatmap(self):
        """Figura 5: Matriz de correlación entre características"""
        # Seleccionar columnas relevantes
        cols_to_include = ['rms_mean', 'kurtosis_mean', 'skewness_mean', 
                          'age', 'n_seizures', 'duration']
        cols_to_include += [f'rel_power_{b}' for b in BANDS]
        
        cols_available = [c for c in cols_to_include if c in self.df.columns]
        
        # Renombrar para mejor visualización
        rename_dict = {
            'rms_mean': 'RMS',
            'kurtosis_mean': 'Curtosis',
            'skewness_mean': 'Asimetría',
            'age': 'Edad',
            'n_seizures': 'N° Crisis',
            'duration': 'Duración',
            'rel_power_Delta': 'Delta',
            'rel_power_Theta': 'Theta',
            'rel_power_Alpha': 'Alpha',
            'rel_power_Beta': 'Beta',
            'rel_power_Gamma': 'Gamma'
        }
        
        df_corr = self.df[cols_available].copy()
        df_corr = df_corr.rename(columns=rename_dict)
        
        # Matriz de correlación
        corr_matrix = df_corr.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Máscara triangular superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Coeficiente de Correlación (r)'},
                   annot_kws={'size': 9})
        
        ax.set_title('Figura 5: Matriz de Correlación de Características EEG\n'
                    'Población Pediátrica con Epilepsia',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Guardar matriz
        corr_matrix.to_csv(self.output_dir / 'tabla5_correlation_matrix.csv')
        
        self._save_figure('fig5_correlation_heatmap')
        plt.show()
    
    def fig6_subject_profiles(self):
        """Figura 6: Perfil espectral por sujeto"""
        # Media por sujeto
        df_subj = self.df.groupby('subject_id').agg({
            'age': 'first',
            **{f'rel_power_{b}': 'mean' for b in BANDS}
        }).reset_index()
        
        # Ordenar por edad
        df_subj = df_subj.sort_values('age')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df_subj))
        width = 0.15
        
        for i, band in enumerate(BANDS):
            col = f'rel_power_{band}'
            offset = (i - 2) * width
            bars = ax.bar(x + offset, df_subj[col] * 100, width,
                         label=band, color=COLORS[band], alpha=0.8)
        
        ax.set_xlabel('Sujeto (ordenado por edad)', fontsize=12)
        ax.set_ylabel('Potencia Relativa (%)', fontsize=12)
        ax.set_title('Figura 6: Perfil Espectral Individual por Sujeto\n'
                    '(ordenado por edad ascendente)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n({a:.0f}a)" for s, a in 
                          zip(df_subj['subject_id'], df_subj['age'])],
                         rotation=45, ha='right', fontsize=9)
        ax.legend(title='Banda', loc='upper right')
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        self._save_figure('fig6_subject_profiles')
        plt.show()
    
    def fig7_seizure_burden(self):
        """Figura 7: Carga de crisis por sujeto"""
        # Agregar por sujeto
        df_subj = self.df.groupby('subject_id').agg({
            'age': 'first',
            'gender': 'first',
            'n_seizures': 'sum',
            'has_seizures': 'sum',  # Número de archivos con crisis
            'file_id': 'count'      # Total de archivos
        }).reset_index()
        
        df_subj = df_subj.rename(columns={'file_id': 'n_files'})
        df_subj['seizure_rate'] = df_subj['has_seizures'] / df_subj['n_files'] * 100
        df_subj = df_subj.sort_values('n_seizures', ascending=False)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Total de crisis por sujeto
        ax1 = axes[0]
        colors = ['#e74c3c' if g == 'F' else '#2196f3' if g == 'M' else '#9e9e9e' 
                 for g in df_subj['gender']]
        bars = ax1.bar(range(len(df_subj)), df_subj['n_seizures'], color=colors, alpha=0.8)
        ax1.set_xticks(range(len(df_subj)))
        ax1.set_xticklabels(df_subj['subject_id'], rotation=45, ha='right')
        ax1.set_xlabel('Sujeto')
        ax1.set_ylabel('Número Total de Crisis')
        ax1.set_title('A) Carga Total de Crisis', fontweight='bold')
        
        # Panel 2: Crisis vs Edad
        ax2 = axes[1]
        ax2.scatter(df_subj['age'], df_subj['n_seizures'], s=100, c='#e74c3c', 
                   alpha=0.7, edgecolor='white')
        
        # Correlación
        r, p = stats.pearsonr(df_subj['age'], df_subj['n_seizures'])
        ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', transform=ax2.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Edad (años)')
        ax2.set_ylabel('Número Total de Crisis')
        ax2.set_title('B) Relación Edad-Crisis', fontweight='bold')
        
        # Panel 3: Distribución de carga
        ax3 = axes[2]
        ax3.hist(df_subj['n_seizures'], bins=10, color='#e74c3c', alpha=0.7, edgecolor='white')
        ax3.axvline(df_subj['n_seizures'].mean(), color='black', linestyle='--', 
                   label=f'Media: {df_subj["n_seizures"].mean():.1f}')
        ax3.axvline(df_subj['n_seizures'].median(), color='blue', linestyle=':', 
                   label=f'Mediana: {df_subj["n_seizures"].median():.1f}')
        ax3.set_xlabel('Número de Crisis')
        ax3.set_ylabel('Número de Sujetos')
        ax3.set_title('C) Distribución de Carga', fontweight='bold')
        ax3.legend()
        
        plt.suptitle('Figura 7: Análisis de Carga de Crisis Epilépticas',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Guardar tabla
        df_subj.to_csv(self.output_dir / 'tabla7_seizure_burden.csv', index=False)
        
        self._save_figure('fig7_seizure_burden')
        plt.show()
    
    def _get_freq_range(self, band):
        """Retorna el rango de frecuencia de una banda"""
        ranges = {
            'Delta': '0.5-4 Hz',
            'Theta': '4-8 Hz', 
            'Alpha': '8-13 Hz',
            'Beta': '13-30 Hz',
            'Gamma': '30-40 Hz'
        }
        return ranges.get(band, '')
    
    def _save_figure(self, name):
        """Guarda figura en PNG y PDF"""
        png_path = self.output_dir / f'{name}.png'
        pdf_path = self.output_dir / f'{name}.pdf'
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        
        print(f"📊 Guardado: {png_path.name} y {pdf_path.name}")
    
    def generate_full_report(self):
        """Genera el análisis completo para la tesis"""
        print("\n" + "="*70)
        print("🎓 GENERANDO ANÁLISIS COMPLETO PARA TESIS")
        print("   Características EEG en Población Pediátrica con Epilepsia")
        print("="*70)
        
        # Cargar datos
        self.load_all_data()
        
        if self.df is None or len(self.df) == 0:
            print("❌ No hay datos para analizar")
            return
        
        # Generar todos los análisis
        print("\n" + "-"*50)
        print("📈 Sección 1: Estadísticas Descriptivas")
        print("-"*50)
        self.descriptive_statistics()
        
        print("\n" + "-"*50)
        print("📊 Sección 2: Figuras para Tesis")
        print("-"*50)
        
        print("\n▶ Figura 1: Distribución de potencia por banda...")
        self.fig1_band_power_distribution()
        
        print("\n▶ Figura 2: Boxplot comparativo de bandas...")
        self.fig2_band_power_comparison()
        
        print("\n▶ Figura 3: Comparación ictal vs interictal...")
        self.fig3_ictal_vs_interictal()
        
        print("\n▶ Figura 4: Correlación edad vs potencia espectral...")
        self.fig4_age_correlation()
        
        print("\n▶ Figura 5: Matriz de correlación...")
        self.fig5_correlation_heatmap()
        
        print("\n▶ Figura 6: Perfiles espectrales individuales...")
        self.fig6_subject_profiles()
        
        print("\n▶ Figura 7: Análisis de carga de crisis...")
        self.fig7_seizure_burden()
        
        # Resumen final
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETO GENERADO")
        print("="*70)
        print(f"\n📁 Todos los archivos guardados en: {self.output_dir}")
        
        print("\n📄 Archivos generados:")
        for f in sorted(self.output_dir.glob("*")):
            size = f.stat().st_size / 1024
            print(f"   • {f.name} ({size:.1f} KB)")
        
        print("\n🎓 ¡Listo para incluir en tu tesis!")
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de datos EEG para tesis')
    parser.add_argument('--results_dir', type=str, default='results_pediatric_study',
                       help='Directorio con resultados del pipeline')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio para figuras (default: results_dir/thesis_figures)')
    
    args = parser.parse_args()
    
    analyzer = ThesisAnalyzer(args.results_dir, args.output_dir)
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()