# Instrucciones para agregar los modelos

## Archivos necesarios

Copia estos 4 archivos `.pkl` a esta carpeta (`modelos/`):

1. ✅ `logreg_model.pkl` - Modelo de Regresión Logística (incluye preprocesador)
2. ✅ `knn_model.pkl` - Modelo KNN (incluye preprocesador)
3. ✅ `kmeans_model.pkl` - Modelo K-Means
4. ✅ `credit_scaler.pkl` - Preprocesador para Credit Card

## ✅ Nota importante

Los modelos `logreg_model.pkl` y `knn_model.pkl` **ya incluyen el preprocesador** dentro de ellos (son Pipelines de sklearn). No necesitas un archivo separado de preprocesador para Telco.

## Cómo copiar los archivos

### Opción 1: Desde el Explorador de Archivos
1. Abre la carpeta donde tienes los archivos `.pkl`
2. Selecciona los 4 archivos (o 5 si incluyes el preprocesador de Telco)
3. Copia (Ctrl+C)
4. Ve a `C:\Users\milaf\Parcial3\modelos\`
5. Pega (Ctrl+V)

### Opción 2: Desde la terminal
```powershell
# Navega a la carpeta donde están tus archivos .pkl
cd "ruta\a\tus\archivos"

# Copia los archivos a la carpeta modelos
copy logreg_model.pkl C:\Users\milaf\Parcial3\modelos\
copy knn_model.pkl C:\Users\milaf\Parcial3\modelos\
copy kmeans_model.pkl C:\Users\milaf\Parcial3\modelos\
copy credit_scaler.pkl C:\Users\milaf\Parcial3\modelos\
```

## Verificar que los archivos estén en su lugar

Después de copiar, deberías ver en la carpeta `modelos/`:
- logreg_model.pkl
- knn_model.pkl
- kmeans_model.pkl
- credit_scaler.pkl
- telco_preprocessor.pkl (o el nombre que tenga)

