"""
Configuración centralizada de la aplicación
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Configuración de la aplicación usando Pydantic Settings
    
    Permite override via variables de entorno o archivo .env
    """
    
    # Información de la app
    app_name: str = "Sentiment API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Configuración del modelo ML
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_length: int = 512  # Máximo de tokens a procesar
    
    # Costos estimados (puedes ajustar según tu análisis)
    cost_per_inference_usd: float = 0.0001
    
    # Configuración de servidor
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"  # Permite cargar desde archivo .env
        case_sensitive = False


# Singleton de configuración
settings = Settings()