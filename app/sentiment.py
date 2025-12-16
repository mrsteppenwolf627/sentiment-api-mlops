"""
Lógica de análisis de sentimiento usando HuggingFace Transformers
"""
from transformers import pipeline
from typing import Dict
import time
import logging

from app.config import settings


# Logger para este módulo
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analizador de sentimiento usando modelo pre-entrenado de HuggingFace
    
    El modelo se carga UNA VEZ al inicializar (es pesado, ~250MB en memoria)
    y se reutiliza para todas las inferencias.
    
    Thread-safe: Transformers pipeline puede usarse concurrentemente.
    """
    
    def __init__(self):
        """
        Inicializa el modelo de sentiment analysis
        
        Nota: Primera vez descarga ~250MB (modelo DistilBERT)
        Subsecuentes usos lo carga de cache local.
        """
        logger.info(f"Loading sentiment model: {settings.model_name}")
        
        try:
            self.model = pipeline(
                task="sentiment-analysis",
                model=settings.model_name,
                max_length=settings.max_length,
                truncation=True,
                device=-1  # -1 = CPU, 0 = GPU (si tienes)
            )
            self.model_version = settings.model_name
            logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze(self, text: str) -> Dict:
        """
        Analiza el sentimiento de un texto
        
        Args:
            text: Texto en inglés a analizar
            
        Returns:
            Dict con:
            - sentiment: "positive", "negative" o "neutral"
            - confidence: float entre 0-1
            - processing_time_ms: tiempo en milisegundos
            
        Raises:
            Exception si el análisis falla
        """
        start_time = time.time()
        
        try:
            # Inference del modelo
            result = self.model(text)[0]
            
            # Mapear labels del modelo HuggingFace
            # DistilBERT-SST2 devuelve "POSITIVE" o "NEGATIVE"
            label_map = {
                "POSITIVE": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral"
            }
            
            sentiment = label_map.get(result["label"].upper(), "neutral")
            confidence = float(result["score"])
            
            # Calcular tiempo de procesamiento
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                f"Analysis complete: sentiment={sentiment}, "
                f"confidence={confidence:.4f}, "
                f"time={processing_time_ms:.2f}ms"
            )
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "processing_time_ms": processing_time_ms
            }
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Analysis failed after {processing_time_ms:.2f}ms: {e}"
            )
            raise


# Singleton global del analyzer
# Se carga UNA vez al importar este módulo
_analyzer: SentimentAnalyzer | None = None


def get_analyzer() -> SentimentAnalyzer:
    """
    Obtiene o inicializa el analyzer (Singleton pattern)
    
    El analyzer se crea solo UNA vez y se reutiliza.
    Esto evita cargar el modelo múltiples veces (costoso).
    
    Returns:
        Instancia única de SentimentAnalyzer
    """
    global _analyzer
    
    if _analyzer is None:
        logger.info("Initializing analyzer (first time)")
        _analyzer = SentimentAnalyzer()
    
    return _analyzer