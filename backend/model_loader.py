"""
Módulo para cargar modelos y preprocesadores desde archivos pickle.
"""

import pickle
import os
from pathlib import Path


def get_model_path(filename):
    """
    Obtiene la ruta completa del modelo.
    
    Parameters:
    -----------
    filename : str
        Nombre del archivo del modelo
    
    Returns:
    --------
    str : Ruta completa al archivo del modelo
    """
    # Obtener la ruta del directorio raíz del proyecto
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / "modelos"
    return str(model_dir / filename)


def load_model(model_filename):
    """
    Carga un modelo desde un archivo pickle.
    
    Parameters:
    -----------
    model_filename : str
        Nombre del archivo del modelo (ej: 'logistic_regression_model.pkl')
    
    Returns:
    --------
    object : Modelo cargado o None si no existe
    """
    model_path = get_model_path(model_filename)
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error al cargar el modelo {model_filename}: {str(e)}")
            return None
    return None


def load_preprocessor(preprocessor_filename):
    """
    Carga un preprocesador desde un archivo pickle.
    
    Parameters:
    -----------
    preprocessor_filename : str
        Nombre del archivo del preprocesador (ej: 'telco_preprocessor.pkl')
    
    Returns:
    --------
    object : Preprocesador cargado o None si no existe
    """
    return load_model(preprocessor_filename)  # Misma lógica


def model_exists(model_filename):
    """
    Verifica si un archivo de modelo existe.
    
    Parameters:
    -----------
    model_filename : str
        Nombre del archivo del modelo
    
    Returns:
    --------
    bool : True si el archivo existe, False en caso contrario
    """
    model_path = get_model_path(model_filename)
    return os.path.exists(model_path)

