"""
API para exponer los modelos mediante HTTP.
Se utiliza FastAPI para permitir que el frontend (Streamlit)
consuma los modelos a través de peticiones POST.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.predictors import (
    predict_logistic_regression,
    predict_knn,
    predict_kmeans,
    prepare_telco_input,
    prepare_credit_card_input,
)


app = FastAPI(
    title="Machine Learning API",
    description="API para exponer los modelos de Regresión Logística, KNN y K-Means",
    version="1.0.0",
)


class TelcoRequest(BaseModel):
    gender: str
    senior_citizen: str = Field(..., alias="senior_citizen")
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


class CreditCardRequest(BaseModel):
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: int
    PURCHASES_TRX: int
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: int


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Machine Learning API operativa"}


@app.post("/predict/logistic")
def predict_logistic(request: TelcoRequest):
    try:
        formatted = prepare_telco_input(
            request.gender,
            request.senior_citizen,
            request.partner,
            request.dependents,
            request.tenure,
            request.phone_service,
            request.multiple_lines,
            request.internet_service,
            request.online_security,
            request.online_backup,
            request.device_protection,
            request.tech_support,
            request.streaming_tv,
            request.streaming_movies,
            request.contract,
            request.paperless_billing,
            request.payment_method,
            request.monthly_charges,
            request.total_charges,
        )
        result = predict_logistic_regression(formatted)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Error interno en el modelo") from exc


@app.post("/predict/knn")
def predict_knn_endpoint(request: TelcoRequest):
    try:
        formatted = prepare_telco_input(
            request.gender,
            request.senior_citizen,
            request.partner,
            request.dependents,
            request.tenure,
            request.phone_service,
            request.multiple_lines,
            request.internet_service,
            request.online_security,
            request.online_backup,
            request.device_protection,
            request.tech_support,
            request.streaming_tv,
            request.streaming_movies,
            request.contract,
            request.paperless_billing,
            request.payment_method,
            request.monthly_charges,
            request.total_charges,
        )
        result = predict_knn(formatted)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Error interno en el modelo") from exc


@app.post("/predict/kmeans")
def predict_kmeans_endpoint(request: CreditCardRequest):
    try:
        formatted = prepare_credit_card_input(
            request.BALANCE,
            request.BALANCE_FREQUENCY,
            request.PURCHASES,
            request.ONEOFF_PURCHASES,
            request.INSTALLMENTS_PURCHASES,
            request.CASH_ADVANCE,
            request.PURCHASES_FREQUENCY,
            request.ONEOFF_PURCHASES_FREQUENCY,
            request.PURCHASES_INSTALLMENTS_FREQUENCY,
            request.CASH_ADVANCE_FREQUENCY,
            request.CASH_ADVANCE_TRX,
            request.PURCHASES_TRX,
            request.CREDIT_LIMIT,
            request.PAYMENTS,
            request.MINIMUM_PAYMENTS,
            request.PRC_FULL_PAYMENT,
            request.TENURE,
        )
        result = predict_kmeans(formatted)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Error interno en el modelo") from exc


# Permite ejecutar con: uvicorn backend.api:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)



