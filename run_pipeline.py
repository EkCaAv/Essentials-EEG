#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 Pipeline de Análisis EEG - Arquitectura Vertical Slices + Screaming

Uso:
    python run_pipeline.py                                    # Configuración base
    python run_pipeline.py --config experiments.exp_smoke_test  # Smoke test
    python run_pipeline.py --config experiments.exp_batch_chb01_03  # Batch test
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from datetime import datetime


def load_config(config_name: str):
    """
    Carga la configuración de forma inteligente
    
    Args:
        config_name: Nombre del módulo de configuración
    
    Returns:
        Instancia de configuración
    
    Raises:
        ImportError: Si no se puede cargar la configuración
    """
    try:
        # Importar el módulo de configuración
        config_module = importlib.import_module(f"config.{config_name}")
        
        # Prioridad 1: Buscar variable CONFIG (forma preferida)
        if hasattr(config_module, 'CONFIG'):
            config = config_module.CONFIG
            print(f"   ✓ Cargado desde CONFIG en {config_name}")
            return config
        
        # Prioridad 2: Buscar clase que termine en Config (pero no BaseConfig importado)
        for name in dir(config_module):
            obj = getattr(config_module, name)
            if (isinstance(obj, type) and 
                name.endswith('Config') and 
                name != 'BaseConfig' and
                hasattr(obj, '__module__') and
                obj.__module__ == config_module.__name__):
                config = obj()
                print(f"   ✓ Instanciado desde clase {name}")
                return config
        
        # Prioridad 3: Usar BaseConfig directamente si es base_config
        if config_name == "base_config":
            from config.base_config import BaseConfig
            config = BaseConfig()
            print(f"   ✓ Usando BaseConfig por defecto")
            return config
        
        raise ImportError(f"No se encontró configuración válida en {config_name}")
        
    except ModuleNotFoundError as e:
        print(f"❌ Error: Módulo no encontrado '{config_name}'")
        print(f"   Verifica que el archivo existe en config/{config_name.replace('.', '/')}.py")
        print(f"   Error detallado: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error al cargar configuración '{config_name}'")
        print(f"   {type(e).__name__}: {str(e)}")
        sys.exit(1)


def print_config_summary(config, config_name: str):
    """
    Imprime resumen de la configuración cargada
    
    Args:
        config: Instancia de configuración
        config_name: Nombre del módulo de configuración
    """
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando Pipeline EEG")
    print(f"{'='*60}")
    print(f"📋 Configuración: {config_name}")
    print(f"🧪 Experimento: {config.experiment_name}")
    print(f"👥 Sujetos: {config.subjects}")
    print(f"📂 Datos: {config.base_dir}")
    print(f"📊 Resultados: {config.results_dir}")
    
    # Target files (opcional)
    if hasattr(config, 'target_files') and config.target_files:
        print(f"🎯 Target files: {config.target_files}")
    
    # Bandas de frecuencia
    print(f"🌊 Bandas: {list(config.bands.to_dict().keys())}")
    
    # Preprocesamiento
    print(f"⚙️  Filtro: {config.preprocessing.filter_low}-{config.preprocessing.filter_high} Hz")
    print(f"⚙️  Notch: {config.preprocessing.notch_freq} Hz")
    
    print(f"{'='*60}\n")


def main():
    """Punto de entrada principal del pipeline"""
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(
        description="Pipeline de Análisis EEG con arquitectura modular",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_pipeline.py                                    # Todos los sujetos
  python run_pipeline.py --config experiments.exp_smoke_test  # Test rápido
  python run_pipeline.py --config experiments.exp_no_gamma   # Sin banda gamma
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base_config",
        help="Nombre del módulo de configuración (default: base_config)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    args = parser.parse_args()
    
    # Registrar tiempo de inicio
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n⏱️  Inicio: {start_datetime}")
    print(f"📦 Cargando configuración: {args.config}")
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Mostrar resumen de configuración
    print_config_summary(config, args.config)
    
    # Importar runner aquí para evitar imports circulares
    from orchestration.pipeline_runner import PipelineRunner
    
    # Crear instancia del runner
    runner = PipelineRunner(config)
    
    # Procesar cada sujeto
    total_subjects = len(config.subjects)
    successful = 0
    failed = 0
    
    for idx, subject in enumerate(config.subjects, 1):
        print(f"\n[{idx}/{total_subjects}] ", end="")
        try:
            runner.run_for_subject(subject)
            successful += 1
        except Exception as e:
            print(f"❌ Error procesando {subject}: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
    
    # Calcular tiempo total
    elapsed_time = time.time() - start_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = int(elapsed_time % 60)
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE EJECUCIÓN")
    print(f"{'='*60}")
    print(f"⏱️  Tiempo total: {elapsed_min}m {elapsed_sec}s")
    print(f"✅ Sujetos exitosos: {successful}/{total_subjects}")
    if failed > 0:
        print(f"❌ Sujetos fallidos: {failed}/{total_subjects}")
    print(f"📂 Resultados en: {config.results_dir}")
    print(f"{'='*60}")
    
    if failed == 0:
        print("\n✅ Pipeline completado exitosamente\n")
        return 0
    else:
        print(f"\n⚠️  Pipeline completado con {failed} errores\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())