# Proyecto de Machine Learning - Modelos Supervisados y No Supervisados

Este proyecto contiene modelos de Machine Learning entrenados para clasificación (Regresión Logística y KNN) y clustering (K-Means), junto con una aplicación web interactiva para realizar predicciones.

## 📋 Descripción del Proyecto

El proyecto incluye:

1. **Modelos Supervisados (Clasificación)** - Dataset: Telco Customer Churn
   - Regresión Logística
   - K-Nearest Neighbors (KNN)

2. **Modelo No Supervisado (Clustering)** - Dataset: Credit Card Dataset
   - K-Means Clustering

3. **Aplicación Web** - Interfaz interactiva para probar los modelos

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el repositorio**

2. **Crear un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual**
   
   En Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   En Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. **Instalar las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Estructura del Proyecto

```
Parcial3/
│
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
├── .gitignore                  # Archivos a ignorar en Git
│
├── backend/                    # Lógica del backend
│   ├── __init__.py
│   ├── model_loader.py         # Funciones para cargar modelos
│   ├── predictors.py           # Funciones de predicción
│   └── api.py                  # FastAPI con los endpoints /predict
│
├── frontend/                   # Interfaz de usuario
│   └── app.py                  # Aplicación web principal (Streamlit)
│
├── modelos/                    # Carpeta para los modelos entrenados
│   ├── logreg_model.pkl        # Modelo de Regresión Logística (incluye preprocesador)
│   ├── knn_model.pkl           # Modelo KNN (incluye preprocesador)
│   ├── kmeans_model.pkl        # Modelo K-Means
│   ├── credit_scaler.pkl       # Preprocesador para Credit Card
│   ├── cluster_profiles.pkl    # (Opcional) Perfiles de clusters
│   └── generar_perfiles_clusters.py  # Script de ejemplo
│
└── notebooks/                  # Notebooks de análisis y entrenamiento
    ├── 01_Regresion_Logistica.ipynb
    ├── 02_KNN.ipynb
    └── 03_KMeans.ipynb
```

## 🎯 Uso de la Aplicación Web

### 1. Backend (ejecución local opcional)
El frontend consume un backend HTTP desplegado en Render (`https://machinelearning-af44.onrender.com`).  
Si deseas ejecutar el backend localmente:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Luego, en otra terminal:

```bash
set API_BASE_URL=http://localhost:8000   # Windows PowerShell
export API_BASE_URL=http://localhost:8000  # macOS / Linux
```

Si no defines `API_BASE_URL`, el frontend intentará utilizar el backend desplegado en Render.  
En caso de que dicho backend remoto no esté disponible, la aplicación utilizará automáticamente
los modelos locales como fallback, siempre que los archivos `.pkl` estén en la carpeta `modelos/`.

### 2. Restaurar el backend en Railway

1. **Instala el CLI de Railway** (solo la primera vez)
   ```bash
   npm i -g @railway/cli
   railway login
   ```
2. **Crea o vincula un proyecto**
   ```bash
   railway init                      # crea uno nuevo desde este repo
   # o, si ya tienes un proyecto existente:
   railway link <ID_DEL_PROYECTO>
   ```
3. **Despliega**
   ```bash
   railway up
   ```
   El CLI detectará el `Procfile`/`railway.json` añadidos y ejecutará:
   ```
   uvicorn backend.api:app --host 0.0.0.0 --port $PORT
   ```
   Asegúrate de que la carpeta `modelos/` esté incluida en el despliegue (contiene los `.pkl`).

4. **Obtén la nueva URL** generada por Railway y actualiza el frontend:
   ```bash
   set API_BASE_URL=https://<tu-app>.up.railway.app   # Windows PowerShell
   export API_BASE_URL=https://<tu-app>.up.railway.app # macOS / Linux
   ```
   También puedes definir la variable dentro de los “Environment Variables” del proyecto en la consola de Railway para tenerla siempre disponible.

### 3. Frontend (Streamlit)
### 3. Frontend (Streamlit)

2. **Ejecutar la aplicación**
   
   Si `streamlit` está en tu PATH:
   ```bash
   streamlit run frontend/app.py
   ```
   
   Si no está en tu PATH (más común):
   ```bash
   python -m streamlit run frontend/app.py
   ```

3. **Abrir en el navegador**
   - La aplicación se abrirá automáticamente en `http://localhost:8501`
   - Si no se abre automáticamente, copia la URL que aparece en la terminal

4. **Usar la aplicación**
   - Selecciona el modelo que deseas probar desde el menú lateral
   - Completa el formulario con los datos requeridos
   - Haz clic en el botón de predicción
   - Visualiza los resultados

## 📊 Modelos Incluidos

### Regresión Logística
- **Input**: Variables del dataset Telco Customer Churn
- **Output**: 
  - Probabilidad de abandono (0-100%)
  - Clasificación: Sí/No

### K-Nearest Neighbors (KNN)
- **Input**: Variables del dataset Telco Customer Churn
- **Output**: 
  - Clasificación: Sí/No

### K-Means Clustering
- **Input**: Variables numéricas del Credit Card Dataset
- **Output**: 
  - Número del cluster asignado
  - Descripción del perfil del cluster

## 🌐 Arquitectura y Endpoints

El backend expone los modelos mediante FastAPI en el dominio:

```
https://machinelearning-af44.onrender.com
```

Endpoints disponibles:

- `POST /predict/logistic` → Predicción de Regresión Logística (probabilidades y clasificación)
- `POST /predict/knn` → Predicción con KNN (clasificación)
- `POST /predict/kmeans` → Asignación de cluster y perfil para K-Means

Cada endpoint recibe un JSON con los campos del formulario y devuelve las métricas que consume el frontend de Streamlit.

## 📓 Notebooks de Análisis

El proyecto incluye notebooks completos en la carpeta `notebooks/`:

- **01_Regresion_Logistica.ipynb**: Análisis completo del modelo de Regresión Logística
  - Preprocesamiento de datos
  - Entrenamiento del modelo
  - Métricas (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
  - Matriz de confusión
  - Curva ROC

- **02_KNN.ipynb**: Análisis completo del modelo KNN
  - Preprocesamiento de datos
  - Entrenamiento del modelo
  - Métricas completas
  - Visualizaciones

- **03_KMeans.ipynb**: Análisis completo del modelo K-Means
  - Preprocesamiento de datos
  - Método del codo (Elbow Method)
  - Método de Silhouette
  - Entrenamiento del modelo
  - Interpretación de clusters
  - Visualizaciones con PCA
  - Aplicaciones reales

## 📝 Notas Importantes

- Los modelos deben estar entrenados previamente y guardados como archivos `.pkl`
- Los modelos `logreg_model.pkl` y `knn_model.pkl` ya incluyen el preprocesador dentro (son Pipelines)
- Para K-Means, se requiere el archivo `credit_scaler.pkl` para preprocesar los datos
- Se recomienda incluir `cluster_profiles.pkl` con las descripciones de cada cluster para K-Means
- El frontend ya no importa módulos del backend: todas las predicciones se realizan a través de peticiones HTTP al API

## 🔧 Solución de Problemas

### Error: "streamlit no se reconoce como comando"
- Usa `python -m streamlit run frontend/app.py` en lugar de solo `streamlit run frontend/app.py`

### Error: "No se pudo conectar con el backend"
- Verifica que `API_BASE_URL` apunte al dominio correcto o que el backend local esté ejecutándose
- Comprueba tu conexión a internet si estás usando el backend en Render

### Error: "No se encontraron los archivos del modelo"
- Verifica que los archivos `.pkl` estén en la carpeta `modelos/`
- Verifica que los nombres de los archivos coincidan:
  - `logreg_model.pkl` (no `logistic_regression_model.pkl`)
  - `knn_model.pkl`
  - `kmeans_model.pkl`
  - `credit_scaler.pkl` (no `credit_card_preprocessor.pkl`)

### Error al realizar predicción
- Asegúrate de que el preprocesador y el modelo sean compatibles
- Verifica que los datos de entrada tengan el formato correcto
- Asegúrate de usar scikit-learn versión 1.6.1 (especificada en requirements.txt)

## 🗂️ Archivos del Repositorio

### Estructura para GitHub:
- ✅ Carpeta `/modelos` con los archivos `.pkl`
- ✅ Código de la web en `/frontend` y `/backend`
- ✅ README con instrucciones claras
- ✅ `requirements.txt` con todas las dependencias
- ✅ Notebooks limpios en `/notebooks` con:
  - Preprocesamiento
  - Entrenamiento
  - Resultados
  - Métricas
  - Gráficas

## 👥 Autores

[Tu nombre/equipo]

## 📄 Licencia

[Especificar licencia si aplica]

