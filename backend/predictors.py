"""
Módulo con funciones de predicción para cada modelo.
"""

import pandas as pd
import numpy as np
from backend.model_loader import load_model, load_preprocessor, model_exists


def predict_logistic_regression(input_data):
    """
    Realiza una predicción usando el modelo de Regresión Logística.
    
    Parameters:
    -----------
    input_data : dict
        Diccionario con los datos de entrada del cliente Telco
    
    Returns:
    --------
    dict : Diccionario con predicción y probabilidades
        {
            'prediction': int (0 o 1),
            'probability_churn': float,
            'probability_no_churn': float,
            'classification': str ('Yes' o 'No')
        }
    """
    # Cargar modelo (ya incluye el preprocesador dentro)
    model = load_model("logreg_model.pkl")
    
    if model is None:
        raise FileNotFoundError("No se encontró el modelo de Regresión Logística")
    
    # Convertir input_data a DataFrame
    df = pd.DataFrame([input_data])
    
    # El modelo ya incluye el preprocesador, solo necesitamos hacer predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    return {
        'prediction': int(prediction),
        'probability_churn': float(probability[1]),
        'probability_no_churn': float(probability[0]),
        'classification': 'Sí' if prediction == 1 else 'No'
    }


def predict_knn(input_data):
    """
    Realiza una predicción usando el modelo KNN.
    
    Parameters:
    -----------
    input_data : dict
        Diccionario con los datos de entrada del cliente Telco
    
    Returns:
    --------
    dict : Diccionario con predicción
        {
            'prediction': int (0 o 1),
            'classification': str ('Yes' o 'No')
        }
    """
    # Cargar modelo (ya incluye el preprocesador dentro)
    model = load_model("knn_model.pkl")
    
    if model is None:
        raise FileNotFoundError("No se encontró el modelo de KNN")
    
    # Convertir input_data a DataFrame
    df = pd.DataFrame([input_data])
    
    # El modelo ya incluye el preprocesador, solo necesitamos hacer predict
    prediction = model.predict(df)[0]
    
    return {
        'prediction': int(prediction),
        'classification': 'Sí' if prediction == 1 else 'No'
    }


def predict_kmeans(input_data):
    """
    Asigna un cluster usando el modelo K-Means.
    
    Parameters:
    -----------
    input_data : dict
        Diccionario con los datos de entrada de la tarjeta de crédito
    
    Returns:
    --------
    dict : Diccionario con cluster asignado y distancia
        {
            'cluster': int,
            'distance_to_centroid': float,
            'profile': str (descripción del cluster si está disponible)
        }
    """
    # Cargar modelo y preprocesador
    model = load_model("kmeans_model.pkl")
    preprocessor = load_preprocessor("credit_scaler.pkl")
    
    if model is None or preprocessor is None:
        raise FileNotFoundError("No se encontraron el modelo o el preprocesador de K-Means")
    
    # Convertir input_data a DataFrame
    df = pd.DataFrame([input_data])
    
    # Preprocesar datos
    processed_data = preprocessor.transform(df)
    
    # Realizar predicción
    cluster = model.predict(processed_data)[0]
    
    # Calcular distancia al centroide
    distances = model.transform(processed_data)
    distance_to_centroid = float(distances[0][cluster])
    
    # Cargar perfiles de clusters si existen
    profile = None
    if model_exists("cluster_profiles.pkl"):
        cluster_profiles = load_model("cluster_profiles.pkl")
        if cluster_profiles and cluster in cluster_profiles:
            profile = cluster_profiles[cluster]
    
    return {
        'cluster': int(cluster),
        'distance_to_centroid': distance_to_centroid,
        'profile': profile
    }


def prepare_telco_input(gender, senior_citizen, partner, dependents, tenure,
                        phone_service, multiple_lines, internet_service,
                        online_security, online_backup, device_protection,
                        tech_support, streaming_tv, streaming_movies, contract,
                        paperless_billing, payment_method, monthly_charges,
                        total_charges):
    """
    Prepara los datos de entrada para los modelos de Telco.
    
    Returns:
    --------
    dict : Diccionario con los datos formateados
    """
    # Convertir valores en español a inglés para el modelo
    gender_map = {"Masculino": "Male", "Femenino": "Female"}
    yes_no_map = {"Sí": "Yes", "No": "No"}
    contract_map = {"Mensual": "Month-to-month", "Un año": "One year", "Dos años": "Two year"}
    payment_map = {
        "Cheque electrónico": "Electronic check",
        "Cheque por correo": "Mailed check",
        "Transferencia bancaria (automática)": "Bank transfer (automatic)",
        "Tarjeta de crédito (automática)": "Credit card (automatic)"
    }
    internet_map = {"Fibra óptica": "Fiber optic", "DSL": "DSL", "No": "No"}
    no_service_map = {"Sin servicio telefónico": "No phone service", "Sin servicio de internet": "No internet service"}
    
    return {
        'gender': gender_map.get(gender, gender),
        'SeniorCitizen': 1 if senior_citizen == "Sí" else 0,
        'Partner': yes_no_map.get(partner, partner),
        'Dependents': yes_no_map.get(dependents, dependents),
        'tenure': tenure,
        'PhoneService': yes_no_map.get(phone_service, phone_service),
        'MultipleLines': no_service_map.get(multiple_lines, yes_no_map.get(multiple_lines, multiple_lines)),
        'InternetService': internet_map.get(internet_service, internet_service),
        'OnlineSecurity': no_service_map.get(online_security, yes_no_map.get(online_security, online_security)),
        'OnlineBackup': no_service_map.get(online_backup, yes_no_map.get(online_backup, online_backup)),
        'DeviceProtection': no_service_map.get(device_protection, yes_no_map.get(device_protection, device_protection)),
        'TechSupport': no_service_map.get(tech_support, yes_no_map.get(tech_support, tech_support)),
        'StreamingTV': no_service_map.get(streaming_tv, yes_no_map.get(streaming_tv, streaming_tv)),
        'StreamingMovies': no_service_map.get(streaming_movies, yes_no_map.get(streaming_movies, streaming_movies)),
        'Contract': contract_map.get(contract, contract),
        'PaperlessBilling': yes_no_map.get(paperless_billing, paperless_billing),
        'PaymentMethod': payment_map.get(payment_method, payment_method),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }


def prepare_credit_card_input(balance, balance_frequency, purchases, oneoff_purchases,
                              installments_purchases, cash_advance, purchases_frequency,
                              oneoff_purchases_frequency, purchases_installments_frequency,
                              cash_advance_frequency, cash_advance_trx, purchases_trx,
                              credit_limit, payments, minimum_payments, prc_full_payment,
                              tenure):
    """
    Prepara los datos de entrada para el modelo K-Means.
    
    Returns:
    --------
    dict : Diccionario con los datos formateados
    """
    return {
        'BALANCE': balance,
        'BALANCE_FREQUENCY': balance_frequency,
        'PURCHASES': purchases,
        'ONEOFF_PURCHASES': oneoff_purchases,
        'INSTALLMENTS_PURCHASES': installments_purchases,
        'CASH_ADVANCE': cash_advance,
        'PURCHASES_FREQUENCY': purchases_frequency,
        'ONEOFF_PURCHASES_FREQUENCY': oneoff_purchases_frequency,
        'PURCHASES_INSTALLMENTS_FREQUENCY': purchases_installments_frequency,
        'CASH_ADVANCE_FREQUENCY': cash_advance_frequency,
        'CASH_ADVANCE_TRX': cash_advance_trx,
        'PURCHASES_TRX': purchases_trx,
        'CREDIT_LIMIT': credit_limit,
        'PAYMENTS': payments,
        'MINIMUM_PAYMENTS': minimum_payments,
        'PRC_FULL_PAYMENT': prc_full_payment,
        'TENURE': tenure
    }

