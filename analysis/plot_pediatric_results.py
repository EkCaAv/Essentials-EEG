import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# CONFIGURACIÓN
RESULTS_DIR = Path("results_pediatric_study")
CSV_PATH = RESULTS_DIR / "MASTER_DATASET.csv"
OUTPUT_DIR = RESULTS_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def load_data():
    if not CSV_PATH.exists():
        print(f"❌ Error: No se encuentra {CSV_PATH}")
        return None
    
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Datos cargados: {len(df)} registros (archivos)")
    print(f"   Columnas disponibles: {list(df.columns)}")
    return df

def find_column(df, *keywords):
    """Busca columna que contenga TODAS las keywords (case-insensitive)"""
    for col in df.columns:
        col_lower = col.lower()
        if all(kw.lower() in col_lower for kw in keywords):
            return col
    return None

def plot_demographics(df):
    """1. Distribución de Edad y Género"""
    df_subjects = df.drop_duplicates(subset=['subject_id'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma de Edad
    sns.histplot(data=df_subjects, x='age', hue='gender', multiple="stack", 
                 bins=10, palette="Set1", ax=axes[0])
    axes[0].set_title("Distribución de Edad por Género")
    axes[0].set_xlabel("Edad (años)")
    axes[0].set_ylabel("Número de Pacientes")
    
    # Pie de Género
    gender_counts = df_subjects['gender'].value_counts()
    axes[1].pie(gender_counts.values, labels=gender_counts.index, 
                autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[1].set_title("Distribución por Género")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_demografia.png", dpi=300)
    plt.close()
    print("   -> Generado: 01_demografia.png")

def plot_spectral_power(df):
    """2. Potencia Relativa de Bandas (Boxplot)"""
    # Buscar columnas de potencia relativa
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    cols_found = {}
    
    for band in bands:
        col = find_column(df, 'rel', band)
        if col:
            cols_found[band] = col
    
    if not cols_found:
        print("⚠️ No se encontraron columnas de potencia relativa.")
        return

    # Preparar datos en formato largo
    data_long = []
    for band, col in cols_found.items():
        for val in df[col].dropna():
            data_long.append({'Banda': band, 'Potencia Relativa': val})
    
    df_melt = pd.DataFrame(data_long)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melt, x='Banda', y='Potencia Relativa', 
                hue='Banda', palette="coolwarm", legend=False)
    plt.title("Distribución de Potencia Espectral por Banda\nPoblación Pediátrica con Epilepsia")
    plt.ylim(0, 1)
    plt.savefig(OUTPUT_DIR / "02_potencia_espectral.png", dpi=300)
    plt.close()
    print("   -> Generado: 02_potencia_espectral.png")

def plot_brain_maturation(df):
    """3. Maduración Cerebral: Ratio Alpha/Delta vs Edad"""
    # Buscar columnas (case-insensitive)
    col_alpha = find_column(df, 'rel', 'alpha')
    col_delta = find_column(df, 'rel', 'delta')
    
    if not col_alpha or not col_delta:
        print(f"⚠️ No se encontraron columnas Alpha/Delta.")
        print(f"   Alpha: {col_alpha}, Delta: {col_delta}")
        return
    
    print(f"   Usando: {col_alpha} / {col_delta}")
    
    # Calcular ratio por archivo
    df_plot = df.copy()
    df_plot['AD_Ratio'] = df_plot[col_alpha] / (df_plot[col_delta] + 1e-10)
    
    # Promediar por sujeto para scatter más limpio
    df_subj = df_plot.groupby('subject_id').agg({
        'age': 'first',
        'gender': 'first',
        'AD_Ratio': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 7))
    
    # Scatter por género
    sns.scatterplot(data=df_subj, x='age', y='AD_Ratio', hue='gender', 
                    style='gender', s=150, palette="Set1")
    
    # Línea de tendencia
    z = np.polyfit(df_subj['age'], df_subj['AD_Ratio'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_subj['age'].min(), df_subj['age'].max(), 100)
    plt.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7, label='Tendencia')
    
    # Correlación
    from scipy import stats
    r, pval = stats.pearsonr(df_subj['age'], df_subj['AD_Ratio'])
    
    plt.title(f"Maduración Cerebral: Ratio Alpha/Delta vs Edad\n(r = {r:.3f}, p = {pval:.3f})")
    plt.xlabel("Edad (años)")
    plt.ylabel("Ratio Alpha/Delta")
    plt.legend(title='Género')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_maduracion_cerebral.png", dpi=300)
    plt.close()
    print("   -> Generado: 03_maduracion_cerebral.png")

def plot_seizures(df):
    """4. Carga de Crisis por Paciente"""
    seizures_per_subject = df.groupby('subject_id')['n_seizures'].sum().reset_index()
    seizures_per_subject = seizures_per_subject[seizures_per_subject['n_seizures'] > 0]
    
    if seizures_per_subject.empty:
        print("⚠️ Ningún sujeto tiene crisis registradas.")
        return

    seizures_per_subject = seizures_per_subject.sort_values('n_seizures', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=seizures_per_subject, x='subject_id', y='n_seizures', 
                hue='subject_id', palette="magma", legend=False)
    plt.title("Número Total de Crisis Epilépticas por Paciente")
    plt.xlabel("Sujeto")
    plt.ylabel("Cantidad de Crisis")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_crisis_por_sujeto.png", dpi=300)
    plt.close()
    print("   -> Generado: 04_crisis_por_sujeto.png")

def plot_power_vs_age(df):
    """5. Evolución de Potencia por Banda vs Edad (CORREGIDO NANs)"""
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    colors = {'Delta': '#9b59b6', 'Theta': '#2ecc71', 'Alpha': '#f1c40f', 
              'Beta': '#e74c3c', 'Gamma': '#1abc9c'}
    
    # Promediar por sujeto
    cols_to_agg = {'age': 'first'}
    for band in bands:
        col = find_column(df, 'rel', band)
        if col:
            cols_to_agg[col] = 'mean'
    
    df_subj = df.groupby('subject_id').agg(cols_to_agg).reset_index()
    df_subj = df_subj.sort_values('age')
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        col = find_column(df, 'rel', band)
        
        if col and col in df_subj.columns:
            # --- LIMPIEZA DE DATOS (NUEVO) ---
            # Extraemos solo los datos válidos para esta banda
            data_clean = df_subj[['age', col]].dropna()
            x = data_clean['age'].values
            y = data_clean[col].values
            
            # Graficar puntos
            ax.scatter(x, y, c=colors[band], s=80, alpha=0.7)
            
            if len(x) > 1: # Solo calculamos si hay al menos 2 puntos
                # Tendencia
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p(x_line), 'k--', linewidth=2)
                
                # Correlación
                from scipy import stats
                r, pval = stats.pearsonr(x, y)
                sig = '*' if pval < 0.05 else ''
                title_text = f'{band}\nr={r:.2f}{sig}'
            else:
                title_text = f'{band}\n(datos insuficientes)'
            
            ax.set_title(title_text, fontweight='bold')
            ax.set_xlabel('Edad (años)')
            ax.set_ylabel('Potencia Relativa')
            ax.grid(True, alpha=0.3)
    
    # Panel 6: Leyenda/Info
    axes[5].axis('off')
    axes[5].text(0.5, 0.5, '* p < 0.05\n\nTendencia esperada:\n• Delta ↓ con edad\n• Alpha ↑ con edad',
                ha='center', va='center', fontsize=12, transform=axes[5].transAxes)
    
    plt.suptitle('Evolución de Potencia Espectral con la Edad\nPoblación Pediátrica', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_potencia_vs_edad.png", dpi=300)
    plt.close()
    print("   -> Generado: 05_potencia_vs_edad.png")

if __name__ == "__main__":
    print("🎨 Iniciando generación de gráficas...\n")
    df = load_data()
    
    if df is not None:
        print("\n📊 Generando visualizaciones:")
        plot_demographics(df)
        plot_spectral_power(df)
        plot_brain_maturation(df)
        plot_seizures(df)
        plot_power_vs_age(df)
        
        print(f"\n✅ ¡Completo! Gráficas en: {OUTPUT_DIR}")
        print("\n📁 Archivos generados:")
        for f in sorted(OUTPUT_DIR.glob("*.png")):
            print(f"   • {f.name}")