"""
Modelado Iterativo Minimalista
- Población: 0-5 años
- Target: archivo con crisis (has_seizures)
- Método: añadir features incrementalmente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# === CONFIGURACIÓN ===
DATA_PATH = Path("results_pediatric_study/MASTER_DATASET.csv")
OUTPUT_DIR = Path("results_pediatric_study/figures")
AGE_MIN, AGE_MAX = 0, 5  # Rango de edad

# Features ordenadas por relevancia teórica (de más a menos importante)
FEATURES_ORDER = [
    'rel_power_Delta',    # 1. Delta - muy alta en niños pequeños
    'rel_power_Theta',    # 2. Theta - actividad lenta
    'rel_power_Alpha',    # 3. Alpha - maduración
    'rel_power_Beta',     # 4. Beta - actividad rápida
    'rel_power_Gamma',    # 5. Gamma - alta frecuencia
    'spectral_entropy',   # 6. Complejidad de la señal
    'hjorth_complexity',  # 7. Complejidad temporal
    'hjorth_mobility',    # 8. Movilidad
    'rms_mean',           # 9. Amplitud RMS
    'kurtosis_mean',      # 10. Forma de distribución
]

TARGET = 'has_seizures'


def load_and_filter(age_min, age_max):
    """Carga datos y filtra por edad"""
    df = pd.read_csv(DATA_PATH)
    df_filtered = df[(df['age'] >= age_min) & (df['age'] <= age_max)].copy()
    
    print(f"📊 Datos cargados:")
    print(f"   Total registros: {len(df)}")
    print(f"   Filtrado {age_min}-{age_max} años: {len(df_filtered)}")
    print(f"   Sujetos únicos: {df_filtered['subject_id'].nunique()}")
    print(f"   Con crisis: {df_filtered[TARGET].sum()} ({df_filtered[TARGET].mean()*100:.1f}%)")
    
    return df_filtered


def get_available_features(df, feature_list):
    """Retorna solo features que existen en el DataFrame"""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    
    if missing:
        print(f"⚠️  Features no disponibles: {missing}")
    
    return available


def iterative_model(df, features, target):
    """
    Entrena modelos añadiendo features incrementalmente.
    Retorna métricas para cada iteración.
    """
    results = []
    
    X_all = df[features]
    y = df[target].values
    groups = df['subject_id'].values

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        print("❌ No hay suficientes sujetos para validación agrupada (mínimo 2).")
        return []

    cv = GroupKFold(n_splits=n_splits)
    
    print(f"\n🔄 Modelado Iterativo (target: {target})")
    print("=" * 55)
    
    for i in range(1, len(features) + 1):
        # Usar solo las primeras i features
        X_subset = X_all.iloc[:, :i].values
        feature_names = features[:i]

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ])
        scores = cross_val_score(pipe, X_subset, y, cv=cv, groups=groups, scoring='roc_auc')
        
        mean_auc = scores.mean()
        std_auc = scores.std()
        
        # Guardar resultado
        results.append({
            'n_features': i,
            'last_feature': features[i-1],
            'features': feature_names.copy(),
            'auc_mean': mean_auc,
            'auc_std': std_auc
        })
        
        # Mostrar progreso
        delta = ""
        if i > 1:
            diff = mean_auc - results[-2]['auc_mean']
            delta = f"({'↑' if diff > 0 else '↓'}{abs(diff):.3f})"
        
        print(f"  {i:2d}. +{features[i-1]:<20} → AUC: {mean_auc:.3f} ± {std_auc:.3f} {delta}")
    
    return results


def plot_iterative_results(results, output_path):
    """Genera gráfica de progreso iterativo"""
    
    n_features = [r['n_features'] for r in results]
    auc_means = [r['auc_mean'] for r in results]
    auc_stds = [r['auc_std'] for r in results]
    labels = [r['last_feature'].replace('rel_power_', '').replace('_', ' ') for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Línea principal con error
    ax.plot(n_features, auc_means, 'o-', color='#2ecc71', linewidth=2, markersize=10, label='AUC-ROC')
    ax.fill_between(n_features, 
                    np.array(auc_means) - np.array(auc_stds),
                    np.array(auc_means) + np.array(auc_stds),
                    alpha=0.2, color='#2ecc71')
    
    # Línea base (chance = 0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Azar (0.5)')
    
    # Etiquetas en cada punto
    for i, (x, y, lbl) in enumerate(zip(n_features, auc_means, labels)):
        ax.annotate(f'+{lbl}', (x, y), textcoords="offset points", 
                   xytext=(0, 15), ha='center', fontsize=8, rotation=45)
    
    # Configuración
    ax.set_xlabel('Número de Features', fontsize=12)
    ax.set_ylabel('AUC-ROC (5-fold CV)', fontsize=12)
    ax.set_title('Progreso del Modelo al Añadir Features\nPoblación Pediátrica 0-5 años', fontsize=14)
    ax.set_xticks(n_features)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Resaltar mejor modelo
    best_idx = np.argmax(auc_means)
    ax.scatter([n_features[best_idx]], [auc_means[best_idx]], 
              s=200, facecolors='none', edgecolors='gold', linewidths=3, 
              label=f'Mejor: {n_features[best_idx]} features')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Gráfica guardada: {output_path}")


def main():
    print("🧠 MODELADO ITERATIVO - Epilepsia Pediátrica (0-5 años)\n")
    
    # 1. Cargar y filtrar datos
    df = load_and_filter(AGE_MIN, AGE_MAX)
    
    if len(df) < 20:
        print("⚠️  Muy pocos datos. Ajustando rango de edad...")
        df = load_and_filter(0, 7)  # Expandir si es necesario
    
    # 2. Verificar features disponibles
    features = get_available_features(df, FEATURES_ORDER)
    
    if len(features) < 3:
        print("❌ Insuficientes features para modelado.")
        return
    
    print(f"\n✅ Features a evaluar ({len(features)}):")
    for i, f in enumerate(features, 1):
        print(f"   {i}. {f}")
    
    # 3. Ejecutar modelado iterativo
    results = iterative_model(df, features, TARGET)
    
    # 4. Generar gráfica
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not results:
        return
    plot_iterative_results(results, OUTPUT_DIR / "06_iterative_model.png")
    
    # 5. Resumen final
    best = max(results, key=lambda x: x['auc_mean'])
    print(f"\n{'='*55}")
    print(f"🏆 MEJOR CONFIGURACIÓN:")
    print(f"   Features: {best['n_features']}")
    print(f"   AUC: {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")
    print(f"   Incluye: {', '.join(best['features'])}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()