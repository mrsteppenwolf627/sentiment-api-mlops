# 1. Usar Python 3.11 ligero como base
FROM python:3.11-slim as builder

WORKDIR /app

# 2. Instalar dependencias del sistema y limpiar basura
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalar librerías de Python
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Segunda etapa (para que ocupe menos espacio) ---
FROM python:3.11-slim
WORKDIR /app

# 4. Copiar las librerías instaladas en la etapa anterior
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/

# 5. Configurar variables de entorno
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 6. Exponer el puerto 8000
EXPOSE 8000

# 7. Chequeo de salud (para que Docker sepa si está vivo)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# 8. Comando para arrancar la API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]