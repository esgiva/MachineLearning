"""
Aplicación web frontend con Streamlit.
Interfaz de usuario moderna y atractiva para probar los modelos de Machine Learning.
"""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from backend.predictors import (
    predict_logistic_regression,
    predict_knn,
    predict_kmeans,
    prepare_telco_input,
    prepare_credit_card_input,
)

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="ML Models Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  # Deshabilitar el menú completamente
)

# Inyectar CSS y JavaScript INMEDIATAMENTE para ocultar menús
st.markdown("""
<style>
    /* Ocultar TODO el header de acciones - ENFOQUE RADICAL */
    /* Ocultar el contenedor completo del header derecho */
    div[data-testid="stHeader"] > div:last-child,
    div[data-testid="stHeader"] > div:last-of-type,
    div[data-testid="stHeader"] > div:nth-child(2),
    div[data-testid="stHeader"] > div[style*="flex"]:last-child {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        left: -9999px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Eliminar margin del header completo */
    div[data-testid="stHeader"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stHeader"] > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stHeaderActionElements"],
    button[kind="header"],
    div[data-testid="stHeader"] button,
    div[data-testid="stHeader"] svg,
    div[data-testid="stHeader"] button svg {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Ocultar menús desplegables SOLO DEL HEADER - NO los de selectbox */
    div[data-testid="stHeader"] div[role="menu"],
    div[data-testid="stHeader"] ul[role="menu"],
    div[data-testid="stHeader"] div[data-baseweb="popover"],
    div[data-testid="stHeader"] div[data-baseweb="menu"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Ocultar totalmente el menú de tres puntos y cualquier toolbar residual */
    div[data-testid="stToolbar"],
    div[data-testid="stToolbar"] *,
    button[title="Main menu"],
    button[title="View fullscreen"],
    button[title="Settings"],
    button[title="Rerun"],
    button[data-testid="baseButton-header"],
    button[data-testid="baseButton-toolbar"],
    div[data-testid="stHeader"] button[title],
    div[data-testid="stHeader"] [role="menu"],
    div[data-testid="stHeader"] [data-baseweb="popover"],
    div[data-testid="stHeader"] [data-baseweb="menu"],
    .stAppHeader .stToolbar {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
        opacity: 0 !important;
    }
    
    /* Asegurar que los menús de selectbox sean visibles */
    [data-baseweb="select"] ~ [data-baseweb="popover"],
    [data-baseweb="select"] ~ [data-baseweb="menu"],
    .stSelectbox ~ [data-baseweb="popover"],
    .stSelectbox ~ [data-baseweb="menu"],
    [data-baseweb="popover"]:has([role="option"]),
    [data-baseweb="menu"]:has([role="option"]) {
        display: block !important;
        visibility: visible !important;
        z-index: 99999 !important;
    }
</style>

<script>
(function() {
    function hideEverything() {
        // Ocultar TODOS los botones del header (incluyendo los tres puntos)
        document.querySelectorAll('button[kind="header"]').forEach(b => {
            b.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important; width: 0 !important; height: 0 !important; pointer-events: none !important;';
        });
        
        // Ocultar contenedor de acciones completo
        const actions = document.querySelector('[data-testid="stHeaderActionElements"]');
        if (actions) {
            actions.style.cssText = 'display: none !important; visibility: hidden !important; width: 0 !important; height: 0 !important;';
        }
        
        // Ocultar último div del header (donde están los botones) - MÁS AGRESIVO
        const header = document.querySelector('div[data-testid="stHeader"]');
        if (header) {
            // Ocultar TODOS los divs hijos
            const allDivs = header.querySelectorAll('div');
            allDivs.forEach((div, index) => {
                // Ocultar especialmente el último div y el segundo
                if (index === allDivs.length - 1 || index === 1) {
                    div.style.cssText = 'display: none !important; visibility: hidden !important; width: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important;';
                }
                // Si tiene botones, ocultarlo
                if (div.querySelectorAll('button').length > 0) {
                    div.style.cssText = 'display: none !important; visibility: hidden !important; width: 0 !important; height: 0 !important; overflow: hidden !important;';
                }
            });
            
            // Ocultar específicamente el último div
            const lastDiv = header.lastElementChild;
            if (lastDiv && lastDiv.tagName === 'DIV') {
                lastDiv.style.cssText = 'display: none !important; visibility: hidden !important; width: 0 !important; height: 0 !important; overflow: hidden !important; position: absolute !important; left: -9999px !important; margin: 0 !important; padding: 0 !important;';
            }
            
            // Eliminar margin del header
            header.style.margin = '0';
            header.style.padding = '0';
            
            // Eliminar margin de todos los divs hijos
            header.querySelectorAll('div').forEach(div => {
                div.style.margin = '0';
                div.style.padding = '0';
            });
        }
        
        // Ocultar cualquier botón que tenga tres puntos o iconos de menú
        document.querySelectorAll('button').forEach(btn => {
            const title = btn.getAttribute('title') || '';
            const ariaLabel = btn.getAttribute('aria-label') || '';
            const svg = btn.querySelector('svg');
            // Si tiene SVG (iconos) y está en el header, ocultarlo
            if (svg && btn.closest('[data-testid="stHeader"]')) {
                btn.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important; width: 0 !important; height: 0 !important;';
            }
        });
        
        // Ocultar menús SOLO DEL HEADER (no los de los selectbox) - MÁS ESPECÍFICO
        document.querySelectorAll('div[role="menu"], ul[role="menu"]').forEach(m => {
            // Solo ocultar si está en el header, NO si es un menú de selectbox
            const isInHeader = m.closest('[data-testid="stHeader"]') !== null;
            const isSelectboxMenu = m.closest('[data-baseweb="select"]') !== null || 
                                   m.closest('.stSelectbox') !== null ||
                                   m.querySelector('[role="option"]') !== null ||
                                   m.getAttribute('data-baseweb') === 'menu';
            
            if (isInHeader && !isSelectboxMenu) {
                m.style.cssText = 'display: none !important; visibility: hidden !important;';
            } else if (!isInHeader && isSelectboxMenu) {
                // Asegurar que los menús de selectbox sean visibles
                m.style.cssText = 'display: block !important; visibility: visible !important; z-index: 99999 !important; pointer-events: auto !important;';
            }
        });
    }
    
    // Ejecutar inmediatamente
    hideEverything();
    
    // Ejecutar múltiples veces
    for (let i = 0; i < 10; i++) {
        setTimeout(hideEverything, i * 200);
    }
    
    // Observar cambios - PERO NO BLOQUEAR SELECTBOX
    new MutationObserver(function(mutations) {
        // Solo ocultar elementos del header, NO tocar selectbox
        const header = document.querySelector('div[data-testid="stHeader"]');
        if (header) {
            hideEverything();
        }
        // Después de ocultar header, asegurar que los selectbox funcionen
        setTimeout(function() {
            document.querySelectorAll('[data-baseweb="select"]').forEach(select => {
                if (select.closest('[data-testid="stHeader"]') === null) {
                    select.style.pointerEvents = 'auto';
                    select.style.cursor = 'pointer';
                }
            });
        }, 100);
    }).observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true
    });
})();

// Script para asegurar que los selectbox sean visibles Y FUNCIONALES
(function() {
    function enableSelectboxes() {
        // Forzar visibilidad y funcionalidad de todos los selectbox e inputs - FONDO BLANCO, TEXTO NEGRO, ESQUINAS RECTAS
        document.querySelectorAll('[data-baseweb="select"], [data-baseweb="input"], .stSelectbox, .stNumberInput, .stTextInput').forEach(el => {
            if (el.closest('[data-testid="stHeader"]') === null) { // No afectar elementos del header
                el.style.cssText = 'opacity: 1 !important; visibility: visible !important; display: block !important; pointer-events: auto !important; background-color: white !important; background: white !important; color: #000000 !important; border: 1px solid #000000 !important; border-radius: 0 !important; cursor: pointer !important;';
            }
        });
        
        // Forzar visibilidad y funcionalidad de elementos dentro de selectbox - FONDO BLANCO, TEXTO NEGRO, ESQUINAS RECTAS
        document.querySelectorAll('div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, [data-baseweb="select"] div, [data-baseweb="select"] > div > div').forEach(el => {
            if (el.closest('[data-testid="stHeader"]') === null) { // No afectar elementos del header
                el.style.cssText = 'opacity: 1 !important; visibility: visible !important; pointer-events: auto !important; background-color: white !important; background: white !important; color: #000000 !important; border: 1px solid #000000 !important; border-radius: 0 !important; cursor: pointer !important;';
            }
        });
        
        // Forzar que los botones del dropdown sean clickeables - NO BLOQUEAR EVENTOS
        document.querySelectorAll('[data-baseweb="select"] button, [data-baseweb="select"] > div > button, [data-baseweb="select"] > div > div > button, [data-baseweb="select"]').forEach(btn => {
            if (btn.closest('[data-testid="stHeader"]') === null) {
                btn.style.cssText = 'opacity: 1 !important; visibility: visible !important; pointer-events: auto !important; background-color: white !important; background: white !important; color: #000000 !important; border: 1px solid #000000 !important; border-radius: 0 !important; cursor: pointer !important;';
                // NO bloquear eventos - dejar que Streamlit maneje los clicks
                btn.style.pointerEvents = 'auto';
            }
        });
        
        // Asegurar que TODO el contenedor del selectbox sea clickeable
        document.querySelectorAll('[data-baseweb="select"]').forEach(select => {
            if (select.closest('[data-testid="stHeader"]') === null) {
                select.style.pointerEvents = 'auto';
                select.style.cursor = 'pointer';
                // Remover cualquier bloqueo de eventos
                select.onclick = null;
            }
        });
        
        // Forzar que TODO el texto dentro de selectbox sea negro
        document.querySelectorAll('[data-baseweb="select"] span, [data-baseweb="select"] *').forEach(el => {
            if (el.closest('[data-testid="stHeader"]') === null) {
                el.style.cssText = 'color: #000000 !important; background-color: white !important; background: white !important;';
            }
        });
        
        // Asegurar que los popovers y menús funcionen (excepto los del header) - FONDO BLANCO, TEXTO NEGRO, ESQUINAS RECTAS
        document.querySelectorAll('[data-baseweb="popover"], [data-baseweb="menu"], div[role="listbox"], ul[role="listbox"]').forEach(el => {
            // Solo habilitar si NO está en el header Y es parte de un selectbox
            const isInHeader = el.closest('[data-testid="stHeader"]') !== null;
            const isSelectboxMenu = el.closest('[data-baseweb="select"]') !== null || 
                                   el.closest('.stSelectbox') !== null ||
                                   el.getAttribute('role') === 'listbox';
            
            if (!isInHeader && (isSelectboxMenu || el.closest('section[data-testid="stMain"]') !== null)) {
                el.style.cssText = 'opacity: 1 !important; visibility: visible !important; pointer-events: auto !important; display: block !important; z-index: 9999 !important; background-color: white !important; background: white !important; color: #000000 !important; border: 1px solid #000000 !important; border-radius: 0 !important;';
            }
        });
        
        // Forzar que las opciones del menú tengan fondo blanco, texto negro y sean clickeables - ESQUINAS RECTAS
        document.querySelectorAll('[data-baseweb="menu"] li, [data-baseweb="menu"] > div, [role="option"], [data-baseweb="menu"] *, [data-baseweb="menu"] li > div').forEach(el => {
            if (el.closest('[data-testid="stHeader"]') === null) {
                el.style.cssText = 'background-color: white !important; background: white !important; color: #000000 !important; pointer-events: auto !important; cursor: pointer !important; border-radius: 0 !important;';
                // NO bloquear eventos - dejar que Streamlit maneje los clicks
                el.style.pointerEvents = 'auto';
            }
        });
    }
    
    // Ejecutar inmediatamente
    enableSelectboxes();
    
    // Ejecutar múltiples veces con delay - MÁS FRECUENTE
    for (let i = 0; i < 10; i++) {
        setTimeout(enableSelectboxes, i * 200);
    }
    
    // Observar cambios para habilitar selectbox (solo fuera del header) - MÁS AGRESIVO
    new MutationObserver(function(mutations) {
        enableSelectboxes(); // Ejecutar siempre que haya cambios
    }).observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
    
    // También ejecutar periódicamente para asegurar que funcionen
    setInterval(enableSelectboxes, 1000);
})();

// Script ADICIONAL para forzar que los selectbox funcionen - ENFOQUE DIRECTO Y AGRESIVO
(function() {
    function forceSelectboxClickable() {
        // Encontrar TODOS los selectbox y hacerlos clickeables
        document.querySelectorAll('.stSelectbox, [data-baseweb="select"]').forEach(selectbox => {
            if (selectbox.closest('[data-testid="stHeader"]') === null) {
                // Asegurar que el contenedor completo sea clickeable
                selectbox.style.pointerEvents = 'auto';
                selectbox.style.cursor = 'pointer';
                selectbox.style.opacity = '1';
                selectbox.style.visibility = 'visible';
                
                // Encontrar y habilitar el botón del dropdown
                const buttons = selectbox.querySelectorAll('button');
                buttons.forEach(btn => {
                    btn.style.pointerEvents = 'auto';
                    btn.style.cursor = 'pointer';
                    btn.style.opacity = '1';
                    btn.style.visibility = 'visible';
                    btn.style.display = '';
                    // Remover cualquier atributo que bloquee
                    btn.removeAttribute('disabled');
                    btn.removeAttribute('aria-disabled');
                });
                
                // Asegurar que el input/select interno sea clickeable
                const inputs = selectbox.querySelectorAll('input, select, [role="combobox"], [data-baseweb="input"]');
                inputs.forEach(input => {
                    input.style.pointerEvents = 'auto';
                    input.style.cursor = 'pointer';
                    input.removeAttribute('readonly');
                    input.removeAttribute('disabled');
                    input.removeAttribute('aria-disabled');
                });
                
                // Asegurar que TODOS los divs dentro sean clickeables
                selectbox.querySelectorAll('div').forEach(div => {
                    div.style.pointerEvents = 'auto';
                });
            }
        });
        
        // Asegurar que los menús desplegables aparezcan cuando se abren - MÁS AGRESIVO
        document.querySelectorAll('[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], ul[role="listbox"]').forEach(menu => {
            if (menu.closest('[data-testid="stHeader"]') === null) {
                // Si el menú tiene opciones, asegurar que sea visible
                if (menu.querySelector('[role="option"]') || menu.getAttribute('data-baseweb') === 'menu') {
                    menu.style.pointerEvents = 'auto';
                    menu.style.zIndex = '99999';
                    menu.style.display = 'block';
                    menu.style.visibility = 'visible';
                    menu.style.opacity = '1';
                    
                    // Asegurar que las opciones sean clickeables
                    menu.querySelectorAll('[role="option"], li').forEach(option => {
                        option.style.pointerEvents = 'auto';
                        option.style.cursor = 'pointer';
                    });
                }
            }
        });
    }
    
    // Ejecutar inmediatamente y frecuentemente
    forceSelectboxClickable();
    setTimeout(forceSelectboxClickable, 50);
    setTimeout(forceSelectboxClickable, 100);
    setTimeout(forceSelectboxClickable, 200);
    setTimeout(forceSelectboxClickable, 500);
    setTimeout(forceSelectboxClickable, 1000);
    
    // Ejecutar cada 300ms para asegurar que funcionen
    setInterval(forceSelectboxClickable, 300);
    
    // Observar cambios - MÁS AGRESIVO
    new MutationObserver(function() {
        forceSelectboxClickable();
    }).observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class', 'data-baseweb']
    });
})();
</script>
""", unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", "https://machinelearning-af44.onrender.com")


def call_backend(
    endpoint: str,
    payload: dict,
    fallback=None,
    fallback_label: str = "modo local",
    show_warning: bool = False,
):
    """Helper para invocar el backend vía HTTP y usar un fallback local si falla."""
    base_url = (API_BASE_URL.rstrip("/") if API_BASE_URL else None)
    if base_url:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            if fallback:
                if show_warning:
                    st.warning(
                        f"⚠️ No se pudo usar el backend remoto ({exc}). "
                        f"Intentando {fallback_label}…"
                    )
            else:
                st.error(f"❌ Error al comunicarse con el backend: {exc}")
                return None

    if fallback:
        try:
            return fallback()
        except FileNotFoundError as exc:
            st.error(f"❌ {fallback_label.capitalize()}: {exc}")
        except Exception as exc:  # pragma: no cover
            st.error(f"❌ Error inesperado en {fallback_label}: {exc}")
    else:
        st.error("❌ No hay backend configurado ni fallback disponible para procesar la solicitud.")

    return None


def _format_telco_payload(form_data: dict):
    """Convierte el payload del formulario en el formato requerido por los modelos Telco."""
    return prepare_telco_input(
        form_data["gender"],
        form_data["senior_citizen"],
        form_data["partner"],
        form_data["dependents"],
        form_data["tenure"],
        form_data["phone_service"],
        form_data["multiple_lines"],
        form_data["internet_service"],
        form_data["online_security"],
        form_data["online_backup"],
        form_data["device_protection"],
        form_data["tech_support"],
        form_data["streaming_tv"],
        form_data["streaming_movies"],
        form_data["contract"],
        form_data["paperless_billing"],
        form_data["payment_method"],
        form_data["monthly_charges"],
        form_data["total_charges"],
    )


def _predict_logistic_locally(form_data: dict):
    formatted = _format_telco_payload(form_data)
    return predict_logistic_regression(formatted)


def _predict_knn_locally(form_data: dict):
    formatted = _format_telco_payload(form_data)
    return predict_knn(formatted)


def _predict_kmeans_locally(form_data: dict):
    formatted = prepare_credit_card_input(
        form_data["BALANCE"],
        form_data["BALANCE_FREQUENCY"],
        form_data["PURCHASES"],
        form_data["ONEOFF_PURCHASES"],
        form_data["INSTALLMENTS_PURCHASES"],
        form_data["CASH_ADVANCE"],
        form_data["PURCHASES_FREQUENCY"],
        form_data["ONEOFF_PURCHASES_FREQUENCY"],
        form_data["PURCHASES_INSTALLMENTS_FREQUENCY"],
        form_data["CASH_ADVANCE_FREQUENCY"],
        form_data["CASH_ADVANCE_TRX"],
        form_data["PURCHASES_TRX"],
        form_data["CREDIT_LIMIT"],
        form_data["PAYMENTS"],
        form_data["MINIMUM_PAYMENTS"],
        form_data["PRC_FULL_PAYMENT"],
        form_data["TENURE"],
    )
    return predict_kmeans(formatted)


# ============================================
# CSS PERSONALIZADO
# ============================================
st.markdown("""
<style>
    /* Estilos generales */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        color: #155724;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        color: #856404;
        margin: 1rem 0;
    }
    
    .info-section {
        background: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    /* Fondo principal de la página - Claro y atractivo */
    .main .block-container,
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 50%, #fef3c7 100%) !important;
        background-attachment: fixed;
    }
    
    /* Fondo del contenido principal */
    section[data-testid="stMain"] {
        background: transparent !important;
    }
    
    /* Asegurar que los textos se vean bien */
    .stMarkdown,
    .stText,
    p, span, div, label {
        color: #1f2937 !important;
    }
    
    /* SELECTBOX - FONDO BLANCO, SIN BORDE O BORDE BLANCO, TEXTO NEGRO, ESQUINAS RECTAS */
    .stSelectbox [data-baseweb="select"],
    [data-baseweb="select"],
    div[data-baseweb="select"] {
        background-color: white !important;
        background: white !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0 !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        pointer-events: auto !important;
        cursor: pointer !important;
        box-shadow: none !important;
    }
    
    /* Contenedor interno del selectbox - TODOS LOS DIVS - SIN BORDE */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div,
    [data-baseweb="select"] div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] > div > div {
        background-color: white !important;
        background: white !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0 !important;
        pointer-events: auto !important;
        opacity: 1 !important;
        visibility: visible !important;
        cursor: pointer !important;
        box-shadow: none !important;
    }
    
    /* Botón del dropdown - SIN BORDE */
    [data-baseweb="select"] button,
    [data-baseweb="select"] > div > button,
    [data-baseweb="select"] > div > div > button,
    button[data-baseweb="button"] {
        background-color: white !important;
        background: white !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
        opacity: 1 !important;
        visibility: visible !important;
        box-shadow: none !important;
    }
    
    /* Ocultar cursor y elementos visuales al lado del texto */
    [data-baseweb="select"] input,
    [data-baseweb="select"] input::after,
    [data-baseweb="select"] input::before,
    [data-baseweb="select"] span::after,
    [data-baseweb="select"] span::before,
    [data-baseweb="select"] *::after,
    [data-baseweb="select"] *::before {
        caret-color: transparent !important;
        outline: none !important;
        border: none !important;
    }
    
    /* Ocultar el cursor de texto completamente */
    [data-baseweb="select"] input:focus,
    [data-baseweb="select"] input {
        caret-color: transparent !important;
    }
    
    /* Ocultar cualquier elemento visual adicional al lado */
    [data-baseweb="select"] svg[data-baseweb="icon"],
    [data-baseweb="select"] svg {
        opacity: 0.3 !important;
    }
    
    /* Ocultar cualquier línea o separador visual */
    [data-baseweb="select"] hr,
    [data-baseweb="select"] .separator,
    [data-baseweb="select"] [class*="separator"],
    [data-baseweb="select"] [class*="divider"] {
        display: none !important;
    }
    
    /* Texto dentro del selectbox - TODOS LOS ELEMENTOS - TEXTO NEGRO */
    [data-baseweb="select"] span,
    [data-baseweb="select"] div span,
    [data-baseweb="select"] *,
    [data-baseweb="select"] p,
    [data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] *,
    [data-baseweb="select"] button span {
        color: #000000 !important;
        background-color: white !important;
        background: white !important;
    }
    
    /* Botón del dropdown - CLICKEABLE */
    [data-baseweb="select"] button,
    [data-baseweb="select"] > div > button,
    [data-baseweb="select"] > div > div > button {
        background-color: white !important;
        background: white !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Inputs numéricos y de texto */
    .stNumberInput,
    .stTextInput {
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        pointer-events: auto !important;
    }
    
    [data-baseweb="input"] {
        background-color: white !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        pointer-events: auto !important;
    }
    
    /* Inputs y selects específicos - ESQUINAS RECTAS */
    select,
    input[type="number"],
    input[type="text"],
    input {
        background-color: white !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }
    
    [data-baseweb="input"] {
        border-radius: 0 !important;
    }
    
    /* Menús desplegables de selectbox - FONDO BLANCO, TEXTO NEGRO, ESQUINAS RECTAS */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    div[role="listbox"],
    ul[role="listbox"] {
        background-color: white !important;
        background: white !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        pointer-events: auto !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        z-index: 9999 !important;
    }
    
    /* Opciones dentro del menú - ESQUINAS RECTAS, CLICKEABLES */
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] > div,
    [data-baseweb="menu"] > div > div,
    [role="option"],
    [data-baseweb="menu"] li > div {
        background-color: white !important;
        background: white !important;
        color: #000000 !important;
        border-bottom: 1px solid #e0e0e0 !important;
        border-radius: 0 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
    }
    
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover,
    [data-baseweb="menu"] li > div:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    
    /* Tabs visibles */
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #1f2937 !important;
    }
    
    /* Sidebar con el mismo gradiente que el header principal */
    aside[aria-label="sidebar"],
    aside[aria-label="sidebar"] * {
        background-image: none !important;
    }
    
    aside[aria-label="sidebar"] {
        background: linear-gradient(120deg, #5f7fe9 0%, #7a48ce 100%) !important;
    }
    
    aside[aria-label="sidebar"] > div,
    aside[aria-label="sidebar"] > div > div,
    aside[aria-label="sidebar"] [data-testid="stSidebarContent"],
    aside[aria-label="sidebar"] [data-testid="stVerticalBlock"],
    aside[aria-label="sidebar"] [data-testid="stVerticalBlock"] > div,
    aside[aria-label="sidebar"] [data-testid="block-container"] {
        background: transparent !important;
    }
    
    /* Header del sidebar - Claro y atractivo - MÁS ESPECÍFICO */
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarHeader"] > div,
    [data-testid="stSidebarHeader"] > div > div,
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(135deg, #a78bfa 0%, #c084fc 100%) !important;
        color: white !important;
    }
    
    [data-testid="stSidebarHeader"] p,
    [data-testid="stSidebarHeader"] h1,
    [data-testid="stSidebarHeader"] h2,
    [data-testid="stSidebarHeader"] h3,
    [data-testid="stSidebarHeader"] span,
    section[data-testid="stSidebar"] > div:first-child p,
    section[data-testid="stSidebar"] > div:first-child h1,
    section[data-testid="stSidebar"] > div:first-child h2,
    section[data-testid="stSidebar"] > div:first-child h3 {
        color: white !important;
    }
    
    /* CSS adicional para ocultar elementos del header */
    [data-testid="stHeaderActionElements"],
    div[data-testid="stHeader"] > div:last-child,
    button[kind="header"],
    div[role="menu"],
    ul[role="menu"],
    div[data-baseweb="popover"] {
        display: none !important;
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER PRINCIPAL
# ============================================
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🤖 Predictor de Machine Learning</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Predicciones inteligentes con modelos supervisados y no supervisados
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - SELECCIÓN DE MODELO
# ============================================
st.sidebar.markdown("""
<div style="
    text-align: center;
    padding: 1.25rem;
    border-radius: 18px;
    margin-bottom: 2rem;
    background: linear-gradient(120deg, #6c7fe2 0%, #7c4fc3 100%);
    box-shadow: 0 10px 25px rgba(103, 126, 234, 0.35);
">
    <h2 style="color: white; margin: 0;">📊 Modelos</h2>
</div>
""", unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Selecciona el modelo a probar:",
    ["Regresión Logística", "K-Nearest Neighbors (KNN)", "K-Means Clustering"],
    label_visibility="collapsed"
)

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="
    padding: 1.25rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #6c7fe2 0%, #7c4fc3 100%);
    box-shadow: 0 10px 25px rgba(103, 126, 234, 0.35);
">
    <h4 style="color: white; margin-top: 0;">ℹ️ Información</h4>
    <p style="color: white; font-size: 0.9rem; margin: 0;">
        Esta aplicación permite probar modelos de Machine Learning entrenados previamente.
        Completa el formulario y obtén predicciones instantáneas.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# MODELOS SUPERVISADOS (TELCO CHURN/ABANDONO)
# ============================================
if model_choice in ["Regresión Logística", "K-Nearest Neighbors (KNN)"]:
    
    # Header del modelo
    if model_choice == "Regresión Logística":
        icon = "📈"
        color = "#667eea"
        description = "Predice la probabilidad de que un cliente abandone el servicio"
    else:
        icon = "🔍"
        color = "#764ba2"
        description = "Clasifica clientes usando el algoritmo de vecinos más cercanos"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: white;">{icon} {model_choice}</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Dataset: Churn/Abandono de Clientes Telco</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; opacity: 0.8;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulario de entrada con mejor diseño
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="margin-top: 0; color: #2c3e50;">📝 Datos del Cliente</h3>
        <p style="color: #6c757d; margin-bottom: 0;">Completa todos los campos para obtener la predicción</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Organizar formulario en tabs o secciones
    tab1, tab2, tab3 = st.tabs(["👤 Información Personal", "📱 Servicios", "💳 Facturación"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Género", ["Masculino", "Femenino"], key="gender")
            senior_citizen = st.selectbox("Adulto Mayor", ["No", "Sí"], key="senior")
            partner = st.selectbox("Pareja", ["No", "Sí"], key="partner")
            dependents = st.selectbox("Dependientes", ["No", "Sí"], key="dependents")
        with col2:
            tenure = st.number_input("Tiempo (meses)", min_value=0, max_value=100, value=12, key="tenure")
            st.caption("Tiempo que el cliente ha estado con la compañía")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            phone_service = st.selectbox("Servicio Telefónico", ["No", "Sí"], key="phone")
            multiple_lines = st.selectbox("Múltiples Líneas", ["No", "Sí", "Sin servicio telefónico"], key="multiple")
            internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fibra óptica", "No"], key="internet")
            online_security = st.selectbox("Seguridad en Línea", ["No", "Sí", "Sin servicio de internet"], key="security")
        with col2:
            online_backup = st.selectbox("Respaldo en Línea", ["No", "Sí", "Sin servicio de internet"], key="backup")
            device_protection = st.selectbox("Protección de Dispositivos", ["No", "Sí", "Sin servicio de internet"], key="device")
            tech_support = st.selectbox("Soporte Técnico", ["No", "Sí", "Sin servicio de internet"], key="tech")
            streaming_tv = st.selectbox("TV por Streaming", ["No", "Sí", "Sin servicio de internet"], key="tv")
        
        col3, col4 = st.columns(2)
        with col3:
            streaming_movies = st.selectbox("Películas por Streaming", ["No", "Sí", "Sin servicio de internet"], key="movies")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox("Contrato", ["Mensual", "Un año", "Dos años"], key="contract")
            paperless_billing = st.selectbox("Facturación Sin Papel", ["No", "Sí"], key="paperless")
            payment_method = st.selectbox("Método de Pago", [
                "Cheque electrónico", "Cheque por correo", "Transferencia bancaria (automática)", "Tarjeta de crédito (automática)"
            ], key="payment")
        with col2:
            monthly_charges = st.number_input("Cargos Mensuales ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1, key="monthly")
            total_charges = st.number_input("Cargos Totales ($)", min_value=0.0, max_value=10000.0, value=500.0, step=0.1, key="total")
            st.caption("Cargos totales acumulados")
    
    # Botón de predicción
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Realizar Predicción", type="primary", use_container_width=True)
    
    if predict_button:
        payload = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
        }
        
        endpoint = "/predict/logistic" if model_choice == "Regresión Logística" else "/predict/knn"
        fallback_fn = _predict_logistic_locally if model_choice == "Regresión Logística" else _predict_knn_locally
        
        with st.spinner("Enviando datos al backend..."):
            result = call_backend(
                endpoint,
                payload,
                fallback=lambda: fallback_fn(payload),
                fallback_label="predicción local",
            )
        
        if not result:
            st.stop()
        
        if model_choice == "Regresión Logística":
            # Mostrar resultados con diseño mejorado
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <h2 style="color: #2c3e50;">📊 Resultados de la Predicción</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas principales
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="margin: 0; color: #667eea; font-size: 1.5rem;">Clasificación</h3>
                    <h2 style="margin: 0.5rem 0; color: {'#e74c3c' if result['prediction'] == 1 else '#27ae60'};">
                        {result['classification']}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                prob_churn = result['probability_churn'] * 100
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="margin: 0; color: #667eea; font-size: 1.5rem;">Probabilidad de Churn/Abandono</h3>
                    <h2 style="margin: 0.5rem 0; color: #e74c3c;">{prob_churn:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res3:
                prob_no_churn = result['probability_no_churn'] * 100
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="margin: 0; color: #667eea; font-size: 1.5rem;">Probabilidad de No Churn/Abandono</h3>
                    <h2 style="margin: 0.5rem 0; color: #27ae60;">{prob_no_churn:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Gráfico de barras de probabilidades
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Churn/Abandono', 'Sin Churn/Abandono'],
                y=[prob_churn, prob_no_churn],
                marker_color=['#e74c3c', '#27ae60'],
                text=[f'{prob_churn:.2f}%', f'{prob_no_churn:.2f}%'],
                textposition='auto',
            ))
            fig.update_layout(
                title="Distribución de Probabilidades",
                yaxis_title="Probabilidad (%)",
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Barra de progreso visual
            st.markdown("### Probabilidad de Churn/Abandono")
            st.progress(result['probability_churn'])
            
            # Interpretación
            if result['prediction'] == 1:
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0;">⚠️ Alerta: Cliente en Riesgo</h4>
                    <p style="margin: 0.5rem 0 0 0;">
                        El cliente tiene alta probabilidad de abandonar el servicio (Churn/Abandono).
                        Se recomienda tomar acciones preventivas.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4 style="margin: 0;">✅ Cliente Estable</h4>
                    <p style="margin: 0.5rem 0 0 0;">
                        El cliente tiene baja probabilidad de abandonar el servicio.
                        Cliente con buen perfil de retención.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # KNN
            # Mostrar resultados
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <h2 style="color: #2c3e50;">📊 Resultados de la Predicción (KNN)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="margin: 0; color: #764ba2; font-size: 1.5rem;">Clasificación KNN</h3>
                    <h2 style="margin: 0.5rem 0; color: {'#e74c3c' if result['prediction'] == 1 else '#27ae60'}; font-size: 2.5rem;">
                        {result['classification']}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown("""
                <div class="info-section">
                    <h4 style="margin-top: 0;">ℹ️ Sobre KNN</h4>
                    <p style="margin: 0;">
                        K-Nearest Neighbors clasifica basándose en los clientes más similares
                        en el dataset de entrenamiento.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretación
            if result['prediction'] == 1:
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0;">⚠️ Predicción: Churn/Abandono</h4>
                    <p style="margin: 0.5rem 0 0 0;">
                        Según el algoritmo KNN, este cliente tiene probabilidad de Churn/Abandono
                        basándose en clientes similares del dataset.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4 style="margin: 0;">✅ Predicción: No Churn/Abandono</h4>
                    <p style="margin: 0.5rem 0 0 0;">
                        Según el algoritmo KNN, este cliente NO tiene probabilidad de Churn/Abandono.
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ============================================
# MODELO NO SUPERVISADO (K-MEANS)
# ============================================
else:  # K-Means
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: white;">🔍 K-Means Clustering</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Dataset: Tarjetas de Crédito para Clustering</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; opacity: 0.8;">
            Agrupa clientes en clusters basándose en patrones de comportamiento
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Las asignaciones de clusters se realizan en el backend desplegado en Railway.")
    
    # Formulario de entrada mejorado
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="margin-top: 0; color: #2c3e50;">💳 Datos de la Tarjeta de Crédito</h3>
        <p style="color: #6c757d; margin-bottom: 0;">Ingresa los valores numéricos para asignar el cluster</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Organizar en tabs
    tab1, tab2, tab3 = st.tabs(["💰 Balance y Compras", "📊 Frecuencias", "💵 Pagos y Límites"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            balance = st.number_input("Saldo", min_value=0.0, value=0.0, step=0.01, key="balance")
            balance_frequency = st.number_input("Frecuencia de Saldo", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="bal_freq")
            purchases = st.number_input("Compras", min_value=0.0, value=0.0, step=0.01, key="purchases")
            oneoff_purchases = st.number_input("Compras Únicas", min_value=0.0, value=0.0, step=0.01, key="oneoff")
        with col2:
            installments_purchases = st.number_input("Compras a Plazos", min_value=0.0, value=0.0, step=0.01, key="install")
            cash_advance = st.number_input("Adelanto en Efectivo", min_value=0.0, value=0.0, step=0.01, key="cash")
            st.caption("💡 Consejo: La frecuencia de saldo es un valor entre 0 y 1")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            purchases_frequency = st.number_input("Frecuencia de Compras", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="pur_freq")
            oneoff_purchases_frequency = st.number_input("Frecuencia de Compras Únicas", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="oneoff_freq")
            purchases_installments_frequency = st.number_input("Frecuencia de Compras a Plazos", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="inst_freq")
        with col2:
            cash_advance_frequency = st.number_input("Frecuencia de Adelantos", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="cash_freq")
            cash_advance_trx = st.number_input("Transacciones de Adelanto", min_value=0, value=0, step=1, key="cash_trx")
            purchases_trx = st.number_input("Transacciones de Compras", min_value=0, value=0, step=1, key="pur_trx")
        st.caption("💡 Consejo: Las frecuencias son valores entre 0 y 1 (0 = nunca, 1 = siempre)")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            credit_limit = st.number_input("Límite de Crédito", min_value=0.0, value=1000.0, step=0.01, key="limit")
            payments = st.number_input("Pagos", min_value=0.0, value=0.0, step=0.01, key="payments")
            minimum_payments = st.number_input("Pagos Mínimos", min_value=0.0, value=0.0, step=0.01, key="min_pay")
        with col2:
            prc_full_payment = st.number_input("Porcentaje de Pago Completo", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="full_pay")
            tenure = st.number_input("Tiempo (meses)", min_value=0, max_value=20, value=12, step=1, key="tenure_cc")
        st.caption("💡 Consejo: El porcentaje de pago completo es un valor entre 0 y 1")
    
    # Botón de predicción
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        cluster_button = st.button("🔮 Asignar Cluster", type="primary", use_container_width=True)
    
    if cluster_button:
        payload = {
            "BALANCE": balance,
            "BALANCE_FREQUENCY": balance_frequency,
            "PURCHASES": purchases,
            "ONEOFF_PURCHASES": oneoff_purchases,
            "INSTALLMENTS_PURCHASES": installments_purchases,
            "CASH_ADVANCE": cash_advance,
            "PURCHASES_FREQUENCY": purchases_frequency,
            "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
            "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
            "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
            "CASH_ADVANCE_TRX": cash_advance_trx,
            "PURCHASES_TRX": purchases_trx,
            "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments,
            "MINIMUM_PAYMENTS": minimum_payments,
            "PRC_FULL_PAYMENT": prc_full_payment,
            "TENURE": tenure,
        }
        
        with st.spinner("Calculando cluster en el backend..."):
            result = call_backend(
                "/predict/kmeans",
                payload,
                fallback=lambda: _predict_kmeans_locally(payload),
                fallback_label="asignación local",
            )
        
        if not result:
            st.stop()
        
        # Mostrar resultados
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #2c3e50;">📊 Resultados del Clustering</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas principales
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            cluster_num = result['cluster']
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
            color = colors[cluster_num % len(colors)]
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; border-left-color: {color};">
                <h3 style="margin: 0; color: {color}; font-size: 1.5rem;">Cluster Asignado</h3>
                <h1 style="margin: 0.5rem 0; color: {color}; font-size: 4rem;">
                    {cluster_num}
                </h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin: 0; color: #667eea; font-size: 1.5rem;">Distancia al Centroide</h3>
                <h2 style="margin: 0.5rem 0; color: #2c3e50;">{result['distance_to_centroid']:.4f}</h2>
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                    Menor distancia = mayor similitud con el cluster
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Descripción del perfil del cluster
        st.markdown("### 📋 Perfil del Cluster")
        
        if result['profile']:
            st.markdown(f"""
            <div class="info-section">
                <h4 style="margin-top: 0; color: {color}; font-size: 1.3rem;">
                    Cluster {cluster_num} - Características
                </h4>
                <div style="color: #2c3e50; line-height: 1.8;">
                    {result['profile']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-section">
                <h4 style="margin-top: 0;">Cluster {cluster_num}</h4>
                <p style="margin: 0; color: #2c3e50;">
                    Este cluster representa un grupo de clientes con características similares.
                    Para obtener una descripción detallada del perfil, asegúrate de incluir
                    el archivo 'cluster_profiles.pkl' con las descripciones precalculadas.
                </p>
            </div>
            """, unsafe_allow_html=True)

