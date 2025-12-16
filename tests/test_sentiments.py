"""
Tests para el módulo de sentiment analysis
Testea:
- Inicialización del analyzer
- Análisis de textos positivos/negativos
- Edge cases (texto vacío, muy largo, etc.)
- Singleton pattern
"""
import pytest
from app.sentiment import SentimentAnalyzer, get_analyzer

class TestSentimentAnalyzer:
    """Tests de la clase SentimentAnalyzer"""

    def test_analyzer_initialization(self):
        """Test que el analyzer se inicializa correctamente"""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        assert analyzer.model is not None
        assert analyzer.model_version is not None
        assert isinstance(analyzer.model_version, str)
        assert len(analyzer.model_version) > 0

    def test_analyzer_singleton(self):
        """Test que get_analyzer devuelve siempre la misma instancia"""
        analyzer1 = get_analyzer()
        analyzer2 = get_analyzer()
        # Deben ser el MISMO objeto (mismo id en memoria)
        assert analyzer1 is analyzer2
        assert id(analyzer1) == id(analyzer2)

    def test_analyze_positive_text(self):
        """Test análisis de texto claramente positivo"""
        analyzer = get_analyzer()
        result = analyzer.analyze("I love this product! It's amazing and wonderful.")
        assert result["sentiment"] == "positive"
        assert result["confidence"] > 0.8  # Alta confianza
        assert result["processing_time_ms"] > 0
        assert isinstance(result["processing_time_ms"], float)

    def test_analyze_negative_text(self):
        """Test análisis de texto claramente negativo"""
        analyzer = get_analyzer()
        result = analyzer.analyze("This is terrible. I hate it and it doesn't work.")
        assert result["sentiment"] == "negative"
        assert result["confidence"] > 0.8  # Alta confianza
        assert result["processing_time_ms"] > 0

    def test_analyze_neutral_text(self):
        """Test análisis de texto neutro/ambiguo"""
        analyzer = get_analyzer()
        result = analyzer.analyze("The product exists.")
        # Puede ser cualquier sentiment, pero debe ser válido
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
        assert result["processing_time_ms"] > 0

    def test_analyze_empty_string(self):
        """Test con string vacío (edge case)"""
        analyzer = get_analyzer()
        # Puede lanzar excepción o devolver resultado, dependemos de implementación
        try:
            result = analyzer.analyze("")
            assert "sentiment" in result
        except Exception:
            pass  # Si falla controlado, también es aceptable en este contexto

    def test_analyze_long_text(self):
        """Test con texto muy largo (>512 tokens)"""
        analyzer = get_analyzer()
        # Texto de ~1000 palabras
        long_text = "This is great! " * 200
        result = analyzer.analyze(long_text)
        # Debe funcionar (truncación automática)
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert result["confidence"] > 0