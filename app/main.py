"""
FastAPI Application - Sentiment Analysis API con MLOps completo

Este módulo implementa:
- API REST con FastAPI
- Logging estructurado (structlog)
- Métricas Prometheus
- Error handling
- Health checks
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from datetime import datetime
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.models import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.sentiment import get_analyzer
from app.config import settings


# ============================================================================
# CONFIGURACIÓN DE LOGGING ESTRUCTURADO
# ============================================================================

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),  # Timestamp ISO8601
        structlog.processors.add_log_level,  # Añade nivel (INFO, ERROR)
        structlog.processors.JSONRenderer()  # Output en JSON
    ]
)

logger = structlog.get_logger()


# ============================================================================
# MÉTRICAS PROMETHEUS
# ============================================================================

# Counter: cuenta eventos (siempre sube, nunca baja)
requests_total = Counter(
    'sentiment_api_requests_total',
    'Total de requests recibidas',
    ['endpoint', 'status']  # Labels para filtrar
)

# Histogram: distribución de valores (para latencia)
request_duration = Histogram(
    'sentiment_api_request_duration_seconds',
    'Duración de requests en segundos',
    ['endpoint']
)

# Counter: predicciones por tipo de sentimiento
predictions_total = Counter(
    'sentiment_api_predictions_total',
    'Total de predicciones por sentimiento',
    ['sentiment']
)


# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API de análisis de sentimiento con stack MLOps production-grade",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
)


# ============================================================================
# STARTUP EVENT - Carga el modelo al iniciar
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Se ejecuta UNA vez al iniciar la aplicación
    
    Carga el modelo ML en memoria para que esté listo.
    Si esto falla, la app no arranca (fail-fast).
    """
    logger.info(
        "application_starting",
        version=settings.app_version,
        model=settings.model_name
    )
    
    try:
        # Esto inicializa el singleton y carga el modelo
        analyzer = get_analyzer()
        logger.info(
            "application_ready",
            model_loaded=True,
            model_version=analyzer.model_version
        )
    except Exception as e:
        logger.error(
            "application_startup_failed",
            error=str(e)
        )
        # Re-raise para que la app no arranque
        raise


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get(
    "/",
    tags=["Root"],
    summary="Root endpoint"
)
async def root():
    """
    Endpoint raíz - info básica de la API
    """
    return {
        "message": "Sentiment Analysis API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check"
)
async def health_check():
    """
    Health check endpoint
    
    Verifica que:
    - La app está corriendo
    - El modelo está cargado
    
    Usado por:
    - Docker HEALTHCHECK
    - Kubernetes liveness probe
    - Monitoreo externo
    """
    try:
        analyzer = get_analyzer()
        model_loaded = analyzer is not None
        
        # Métrica: request exitosa
        requests_total.labels(
            endpoint="/health",
            status="success"
        ).inc()
        
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            model_loaded=model_loaded
        )
        
    except Exception as e:
        # Métrica: request fallida
        requests_total.labels(
            endpoint="/health",
            status="error"
        ).inc()
        
        logger.error(
            "health_check_failed",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analiza sentimiento de texto"
)
async def analyze_sentiment(request: AnalyzeRequest):
    """
    Analiza el sentimiento de un texto en inglés
    
    Args:
        request: AnalyzeRequest con campo 'text'
        
    Returns:
        AnalyzeResponse con:
        - sentiment: positive/negative/neutral
        - confidence: 0-1
        - processing_time_ms: latencia
        - model_version: modelo usado
        - timestamp: momento del análisis
        - cost_estimate_usd: costo estimado
        
    Raises:
        HTTPException 422: Si validación falla (texto vacío, muy largo, etc)
        HTTPException 500: Si análisis falla
        
    Example:
        >>> POST /analyze
        >>> {"text": "I love this product!"}
        >>>
        >>> Response:
        >>> {
        >>>   "sentiment": "positive",
        >>>   "confidence": 0.9998,
        >>>   "processing_time_ms": 120.5,
        >>>   ...
        >>> }
    """
    try:
        # Medir latencia total del endpoint
        with request_duration.labels(endpoint="/analyze").time():
            
            # 1. Obtener analyzer
            analyzer = get_analyzer()
            
            # 2. Analizar sentimiento
            result = analyzer.analyze(request.text)
            
            # 3. Construir response
            response = AnalyzeResponse(
                text=request.text,
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                processing_time_ms=result["processing_time_ms"],
                model_version=analyzer.model_version,
                timestamp=datetime.utcnow(),
                cost_estimate_usd=settings.cost_per_inference_usd
            )
            
            # 4. Logging estructurado
            logger.info(
                "sentiment_analyzed",
                sentiment=response.sentiment,
                confidence=response.confidence,
                processing_time_ms=response.processing_time_ms,
                text_length=len(request.text),
                cost_usd=response.cost_estimate_usd
            )
            
            # 5. Métricas
            requests_total.labels(
                endpoint="/analyze",
                status="success"
            ).inc()
            
            predictions_total.labels(
                sentiment=response.sentiment
            ).inc()
            
            return response
            
    except Exception as e:
        # Métrica de error
        requests_total.labels(
            endpoint="/analyze",
            status="error"
        ).inc()
        
        # Log estructurado del error
        logger.error(
            "analysis_failed",
            error=str(e),
            error_type=type(e).__name__,
            text_length=len(request.text) if request.text else 0
        )
        
        # HTTP 500
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get(
    "/metrics",
    tags=["Metrics"],
    summary="Prometheus metrics"
)
async def metrics():
    """
    Endpoint de métricas Prometheus
    
    Expone métricas en formato Prometheus:
    - Counters (requests_total, predictions_total)
    - Histograms (request_duration)
    
    Prometheus puede hacer scraping de este endpoint.
    También útil para debugging manual.
    
    Returns:
        Plain text en formato Prometheus
        
    Example:
        >>> GET /metrics
        >>> 
        >>> # HELP sentiment_api_requests_total Total de requests
        >>> # TYPE sentiment_api_requests_total counter
        >>> sentiment_api_requests_total{endpoint="/analyze",status="success"} 42.0
        >>> ...
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# PUNTO DE ENTRADA (para testing local)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Esto permite ejecutar: python -m app.main
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )