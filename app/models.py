"""
Modelos Pydantic para request/response de la API
"""
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class AnalyzeRequest(BaseModel):
    """
    Modelo para el request de análisis de sentimiento
    
    Pydantic valida automáticamente:
    - text no vacío
    - longitud entre 1-5000 caracteres
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Texto a analizar (inglés)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing."
            }
        }


class AnalyzeResponse(BaseModel):
    """
    Modelo para la respuesta del análisis
    
    Incluye todo el contexto necesario para MLOps:
    - Resultado del modelo
    - Metadata de ejecución
    - Métricas de performance
    - Información de costos
    """
    text: str = Field(..., description="Texto original analizado")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Sentimiento detectado"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confianza del modelo (0-1)"
    )
    processing_time_ms: float = Field(
        ...,
        description="Tiempo de procesamiento en milisegundos"
    )
    model_version: str = Field(
        ...,
        description="Versión del modelo usado"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp UTC de la respuesta"
    )
    cost_estimate_usd: float = Field(
        ...,
        description="Costo estimado de esta inferencia en USD"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "sentiment": "positive",
                "confidence": 0.9998,
                "processing_time_ms": 125.3,
                "model_version": "distilbert-base-uncased-finetuned-sst-2-english",
                "timestamp": "2025-12-16T10:30:00.123456Z",
                "cost_estimate_usd": 0.0001
            }
        }


class HealthResponse(BaseModel):
    """
    Modelo para el health check endpoint
    """
    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la app")
    model_loaded: bool = Field(..., description="Si el modelo ML está cargado")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True
            }
        }