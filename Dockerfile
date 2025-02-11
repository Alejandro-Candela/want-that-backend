# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Configuración para que Python no almacene en buffer la salida
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Comando de inicio: Se apunta al archivo src/app.py
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"] 