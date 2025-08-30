# Dockerfile para Sistema de Manutenção Preditiva
# Bootcamp CDIA - Projeto Final

FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretório para logs
RUN mkdir -p /app/logs

# Expor portas
EXPOSE 8000 8501

# Variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Script de inicialização
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Comando padrão (API)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Para rodar o dashboard, use:
# docker run -p 8501:8501 <image> streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
