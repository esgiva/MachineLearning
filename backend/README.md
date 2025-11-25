# Backend

Esta carpeta contiene toda la lógica del backend del proyecto.

## Archivos

- **`model_loader.py`**: Funciones para cargar modelos y preprocesadores desde archivos pickle.
  - `load_model(filename)`: Carga un modelo desde un archivo .pkl
  - `load_preprocessor(filename)`: Carga un preprocesador desde un archivo .pkl
  - `model_exists(filename)`: Verifica si un archivo de modelo existe

- **`predictors.py`**: Funciones de predicción para cada modelo.
  - `predict_logistic_regression(input_data)`: Predicción con Regresión Logística
  - `predict_knn(input_data)`: Predicción con KNN
  - `predict_kmeans(input_data)`: Asignación de cluster con K-Means
  - `prepare_telco_input(...)`: Prepara datos de entrada para modelos de Telco
  - `prepare_credit_card_input(...)`: Prepara datos de entrada para K-Means

## Uso

El backend se importa desde el frontend de la siguiente manera:

```python
from backend.predictors import (
    predict_logistic_regression,
    predict_knn,
    predict_kmeans
)
```

