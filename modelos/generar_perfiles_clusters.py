"""
Script de ejemplo para generar perfiles de clusters para K-Means.
Este script debe ejecutarse después de entrenar el modelo K-Means
y analizar las características de cada cluster.
"""

import pickle
import pandas as pd
import numpy as np

# Ejemplo de cómo crear y guardar perfiles de clusters
# Debes adaptar esto según tu análisis de los clusters

def generar_perfiles_clusters(model, data, n_clusters):
    """
    Genera descripciones de perfiles para cada cluster.
    
    Parameters:
    -----------
    model : KMeans
        Modelo K-Means entrenado
    data : DataFrame
        Datos originales (después de preprocesamiento)
    n_clusters : int
        Número de clusters
    
    Returns:
    --------
    dict : Diccionario con perfiles de cada cluster
    """
    profiles = {}
    
    # Asignar clusters a los datos
    clusters = model.predict(data)
    
    for i in range(n_clusters):
        cluster_data = data[clusters == i]
        
        # Calcular estadísticas del cluster
        # (Adapta estas métricas según las características de tu dataset)
        profile = f"""
        **Cluster {i}**
        
        **Características principales:**
        - Número de clientes: {len(cluster_data)}
        - Balance promedio: ${cluster_data['BALANCE'].mean():.2f}
        - Compras promedio: ${cluster_data['PURCHASES'].mean():.2f}
        - Límite de crédito promedio: ${cluster_data['CREDIT_LIMIT'].mean():.2f}
        
        **Perfil del cliente:**
        [Describe aquí las características distintivas de este cluster]
        
        **Aplicaciones:**
        [Describe posibles aplicaciones de negocio para este segmento]
        """
        
        profiles[i] = profile
    
    return profiles

# Ejemplo de uso:
# if __name__ == "__main__":
#     # Cargar modelo y datos
#     with open('kmeans_model.pkl', 'rb') as f:
#         kmeans_model = pickle.load(f)
#     
#     # Cargar datos preprocesados
#     # data = pd.read_csv('credit_card_data_preprocessed.csv')
#     
#     # Generar perfiles
#     # profiles = generar_perfiles_clusters(kmeans_model, data, n_clusters=5)
#     
#     # Guardar perfiles
#     # with open('cluster_profiles.pkl', 'wb') as f:
#     #     pickle.dump(profiles, f)
#     
#     print("Perfiles generados y guardados en cluster_profiles.pkl")

