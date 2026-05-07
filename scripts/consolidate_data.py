import argparse
import pandas as pd
from pathlib import Path
from config.subject_metadata import SUBJECTS_DB

def load_metadata_map():
    """Crea un mapa {subject_id: {age: X, gender: Y}} usando el diccionario SUBJECTS_DB"""
    meta_map = {}
    # Iteramos sobre (id, datos) del diccionario
    for subject_id, data in SUBJECTS_DB.items():
        meta_map[subject_id] = {
            'age': data['age'],       # Acceso correcto con ['clave']
            'gender': data['gender']
        }
    return meta_map

def consolidate(results_dir: str):
    root_path = Path(results_dir)
    if not root_path.exists():
        print(f"❌ Error: El directorio '{results_dir}' no existe.")
        return

    print(f"📂 Buscando resultados en: {root_path}")
    
    all_dfs = []
    meta_map = load_metadata_map()
    
    # Buscar recursivamente archivos de features
    feature_files = list(root_path.rglob("*_features.csv"))
    
    print(f"   -> Encontrados {len(feature_files)} archivos de características.")
    
    for csv_path in feature_files:
        try:
            # Obtener subject_id de la carpeta abuelo (ej: chb01/reports/file.csv -> chb01)
            subject_id = csv_path.parent.parent.name 
            
            df = pd.read_csv(csv_path)
            
            # Inyectar metadatos
            if subject_id in meta_map:
                if 'age' not in df.columns:
                    df['age'] = meta_map[subject_id]['age']
                if 'gender' not in df.columns:
                    df['gender'] = meta_map[subject_id]['gender']
                if 'subject_id' not in df.columns:
                    df['subject_id'] = subject_id
            else:
                # Si el sujeto no está en la metadata (ej: chb24), poner ID pero dejar age/gender vacíos
                if 'subject_id' not in df.columns:
                    df['subject_id'] = subject_id
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"❌ Error leyendo {csv_path.name}: {e}")

    if not all_dfs:
        print("⚠️ No se encontraron datos para consolidar.")
        return

    # Unir todo
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Reordenar columnas
    cols = list(master_df.columns)
    first_cols = ['subject_id', 'age', 'gender', 'file']
    # Filtrar columnas que realmente existen
    first_cols = [c for c in first_cols if c in cols]
    remaining_cols = [c for c in cols if c not in first_cols]
    
    master_df = master_df[first_cols + remaining_cols]
    
    # Guardar
    output_csv = root_path / "MASTER_DATASET.csv"
    master_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ CONSOLIDACIÓN COMPLETADA")
    print(f"   Total registros: {len(master_df)}")
    print(f"   Archivo guardado: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolida múltiples CSVs en uno solo.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directorio de resultados")
    
    args = parser.parse_args()
    consolidate(args.results_dir)