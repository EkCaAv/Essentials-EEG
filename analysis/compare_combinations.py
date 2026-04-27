"""
Comparación de Combinaciones de Features
- Prueba múltiples combinaciones
- Rankea de peor a mejor
- Genera visualización comparativa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from pathlib import Path

# === CONFIGURACIÓN ===
DATA_PATH = Path("results_pediatric_study/MASTER_DATASET.csv")
OUTPUT_DIR = Path("results_pediatric_study/figures/combinations")
AGE_MIN, AGE_MAX = 0, 5

# Features base (nombres cortos para visualización)
FEATURES = {
    'rel_power_Delta': 'Δ Delta',
    'rel_power_Theta': 'θ Theta', 
    'rel_power_Alpha': 'α Alpha',
    'rel_power_Beta': 'β Beta',
    'rel_power_Gamma': 'γ Gamma',
}

TARGET = 'has_seizures'
MAX_COMBINATIONS = 31  # 2^5 - 1 combinaciones posibles


def load_data():
    """Carga y filtra datos"""
    df = pd.read_csv(DATA_PATH)
    df = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)].copy()
    
    # Si muy pocos datos, expandir rango
    if len(df) < 30:
        df = pd.read_csv(DATA_PATH)
        df = df[(df['age'] >= 0) & (df['age'] <= 7)].copy()
        print(f"⚠️ Expandido a 0-7 años por datos insuficientes")
    
    print(f"📊 Datos: {len(df)} registros | {df['subject_id'].nunique()} sujetos | {df[TARGET].mean()*100:.1f}% con crisis")
    return df


def evaluate_combination(X, y, groups, feature_names):
    """Evalúa una combinación específica de features"""
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return {
            'features': feature_names,
            'n_features': len(feature_names),
            'auc_mean': np.nan,
            'auc_std': np.nan,
            'scores': np.array([])
        }

    cv = GroupKFold(n_splits=n_splits)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)),
    ])

    scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring='roc_auc')
    
    return {
        'features': feature_names,
        'n_features': len(feature_names),
        'auc_mean': scores.mean(),
        'auc_std': scores.std(),
        'scores': scores
    }


def generate_all_combinations(df, features_dict):
    """Genera y evalúa todas las combinaciones posibles"""
    
    # Filtrar features disponibles
    available = {k: v for k, v in features_dict.items() if k in df.columns}
    feature_cols = list(available.keys())
    feature_labels = list(available.values())
    
    print(f"\n🔬 Evaluando combinaciones de {len(feature_cols)} features...")
    
    results = []
    total_combos = sum(len(list(combinations(range(len(feature_cols)), r))) 
                       for r in range(1, len(feature_cols) + 1))
    
    print(f"   Total combinaciones: {total_combos}\n")
    
    y = df[TARGET].values
    groups = df['subject_id'].values
    combo_count = 0
    
    # Probar todas las combinaciones (1 feature, 2 features, ..., N features)
    for r in range(1, len(feature_cols) + 1):
        for combo_indices in combinations(range(len(feature_cols)), r):
            combo_count += 1
            
            # Seleccionar features de esta combinación
            selected_cols = [feature_cols[i] for i in combo_indices]
            selected_labels = [feature_labels[i] for i in combo_indices]
            
            X = df[selected_cols].fillna(0).values
            
            result = evaluate_combination(X, y, groups, selected_labels)
            result['combo_id'] = combo_count
            results.append(result)
            
            # Progress
            label_str = ' + '.join(selected_labels)
            auc_text = "nan" if np.isnan(result['auc_mean']) else f"{result['auc_mean']:.3f}"
            print(f"   [{combo_count:2d}/{total_combos}] {label_str:<35} → AUC: {auc_text}")
    
    return results


def plot_ranking(results, output_path):
    """Gráfica de barras horizontal: peor a mejor"""
    
    # Ordenar de peor a mejor
    sorted_results = sorted(results, key=lambda x: x['auc_mean'])
    
    labels = [' + '.join(r['features']) for r in sorted_results]
    aucs = [r['auc_mean'] for r in sorted_results]
    stds = [r['auc_std'] for r in sorted_results]
    n_feats = [r['n_features'] for r in sorted_results]
    
    # Colores según número de features
    colors = plt.cm.viridis(np.array(n_feats) / max(n_feats))
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(results) * 0.4)))
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, aucs, xerr=stds, color=colors, edgecolor='black', alpha=0.8)
    
    # Línea de azar
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Azar (0.5)')
    
    # Etiquetas
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Ranking de Combinaciones de Features\n(Ordenado de Peor a Mejor)', fontsize=14)
    ax.set_xlim(0.3, 1.0)
    
    # Valores en las barras
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax.text(auc + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', va='center', fontsize=8)
    
    # Resaltar mejor y peor
    ax.barh(0, aucs[0], color='#e74c3c', edgecolor='black', alpha=0.9)  # Peor
    ax.barh(len(aucs)-1, aucs[-1], color='#2ecc71', edgecolor='black', alpha=0.9)  # Mejor
    
    # Leyenda de colores
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, max(n_feats)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Nº Features')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Ranking guardado: {output_path}")


def plot_top_bottom(results, output_dir):
    """Genera gráficas individuales para TOP 3 y BOTTOM 3"""
    
    sorted_results = sorted(results, key=lambda x: x['auc_mean'])
    
    bottom_3 = sorted_results[:3]   # Peores
    top_3 = sorted_results[-3:][::-1]  # Mejores (invertido para mostrar el mejor primero)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # === BOTTOM 3 (Peores) ===
    for idx, result in enumerate(bottom_3):
        ax = axes[0, idx]
        scores = result['scores']
        
        ax.bar(range(1, 6), scores, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax.axhline(y=result['auc_mean'], color='black', linestyle='--', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel('Fold')
        ax.set_ylabel('AUC')
        ax.set_title(f"#{idx+1} PEOR\n{' + '.join(result['features'])}\nAUC: {result['auc_mean']:.3f}", 
                    fontsize=10, color='#c0392b')
        ax.set_xticks(range(1, 6))
    
    # === TOP 3 (Mejores) ===
    for idx, result in enumerate(top_3):
        ax = axes[1, idx]
        scores = result['scores']
        
        ax.bar(range(1, 6), scores, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax.axhline(y=result['auc_mean'], color='black', linestyle='--', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel('Fold')
        ax.set_ylabel('AUC')
        ax.set_title(f"#{idx+1} MEJOR\n{' + '.join(result['features'])}\nAUC: {result['auc_mean']:.3f}", 
                    fontsize=10, color='#27ae60')
        ax.set_xticks(range(1, 6))
    
    axes[0, 0].set_ylabel('AUC\n(Peores)', fontsize=11, color='#c0392b')
    axes[1, 0].set_ylabel('AUC\n(Mejores)', fontsize=11, color='#27ae60')
    
    plt.suptitle('Comparación: 3 Peores vs 3 Mejores Combinaciones\nPoblación Pediátrica 0-5 años', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_top_vs_bottom.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Top vs Bottom guardado: {output_dir / '07_top_vs_bottom.png'}")


def plot_heatmap_by_nfeatures(results, output_dir):
    """Heatmap mostrando AUC por número de features"""
    
    # Agrupar por número de features
    from collections import defaultdict
    by_n = defaultdict(list)
    
    for r in results:
        by_n[r['n_features']].append(r)
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    
    for n_feat in range(1, 6):
        ax = axes[n_feat - 1]
        
        if n_feat in by_n:
            group = sorted(by_n[n_feat], key=lambda x: x['auc_mean'], reverse=True)
            
            labels = [' + '.join(r['features']) for r in group]
            aucs = [r['auc_mean'] for r in group]
            
            colors = ['#2ecc71' if auc > 0.6 else '#f39c12' if auc > 0.5 else '#e74c3c' for auc in aucs]
            
            y_pos = range(len(labels))
            ax.barh(y_pos, aucs, color=colors, edgecolor='black', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlim(0.4, 0.9)
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('AUC')
        
        ax.set_title(f'{n_feat} Feature{"s" if n_feat > 1 else ""}', fontweight='bold')
    
    plt.suptitle('AUC por Número de Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "08_auc_by_nfeatures.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 AUC por N-features: {output_dir / '08_auc_by_nfeatures.png'}")


def print_summary(results):
    """Imprime resumen de resultados"""
    
    sorted_results = sorted(results, key=lambda x: x['auc_mean'], reverse=True)
    
    print("\n" + "="*60)
    print("📋 RESUMEN DE RESULTADOS")
    print("="*60)
    
    print("\n🏆 TOP 5 MEJORES:")
    for i, r in enumerate(sorted_results[:5], 1):
        features_str = ' + '.join(r['features'])
        print(f"   {i}. AUC={r['auc_mean']:.3f} | {features_str}")
    
    print("\n💀 TOP 5 PEORES:")
    for i, r in enumerate(sorted_results[-5:][::-1], 1):
        features_str = ' + '.join(r['features'])
        print(f"   {i}. AUC={r['auc_mean']:.3f} | {features_str}")
    
    # Mejor por número de features
    print("\n🎯 MEJOR POR NÚMERO DE FEATURES:")
    from collections import defaultdict
    by_n = defaultdict(list)
    for r in results:
        by_n[r['n_features']].append(r)
    
    for n in sorted(by_n.keys()):
        best = max(by_n[n], key=lambda x: x['auc_mean'])
        features_str = ' + '.join(best['features'])
        print(f"   {n} feat: AUC={best['auc_mean']:.3f} | {features_str}")
    
    print("="*60)


def main():
    print("🧠 COMPARACIÓN DE COMBINACIONES - Epilepsia Pediátrica\n")
    
    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos
    df = load_data()
    
    # 2. Generar todas las combinaciones
    results = generate_all_combinations(df, FEATURES)
    
    # 3. Generar visualizaciones
    print("\n📈 Generando visualizaciones...")
    
    plot_ranking(results, OUTPUT_DIR / "06_ranking_combinations.png")
    plot_top_bottom(results, OUTPUT_DIR)
    plot_heatmap_by_nfeatures(results, OUTPUT_DIR)
    
    # 4. Resumen
    print_summary(results)
    
    print(f"\n✅ Gráficas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()