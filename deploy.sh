#!/bin/bash
# Script de Deploy para Sistema de Manutenção Preditiva
# Bootcamp CDIA - Projeto Final

set -e

echo "🚀 Iniciando deploy do Sistema de Manutenção Preditiva..."

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    error "Docker não está instalado. Instale o Docker primeiro."
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose não está instalado. Instale o Docker Compose primeiro."
fi

# Verificar se os modelos existem
if [ ! -f "modelo_manutencao_preditiva.pkl" ]; then
    warning "Modelo não encontrado. Executando treinamento..."
    python train_model.py
fi

if [ ! -f "scaler_features.pkl" ]; then
    error "Scaler não encontrado. Execute o treinamento primeiro."
fi

# Verificar se os dados existem
if [ ! -f "predicoes_manutencao_preditiva.csv" ]; then
    warning "Arquivo de predições não encontrado. Executando predições..."
    python train_model.py
fi

# Criar diretórios necessários
log "Criando diretórios..."
mkdir -p logs
mkdir -p models
mkdir -p data

# Copiar modelos para diretório apropriado
log "Organizando modelos..."
cp modelo_manutencao_preditiva.pkl models/
cp scaler_features.pkl models/
cp predicoes_manutencao_preditiva.csv data/

# Construir imagens Docker
log "Construindo imagens Docker..."
docker-compose build

# Parar containers existentes
log "Parando containers existentes..."
docker-compose down

# Iniciar serviços
log "Iniciando serviços..."
docker-compose up -d

# Aguardar serviços ficarem prontos
log "Aguardando serviços ficarem prontos..."
sleep 10

# Verificar status dos serviços
log "Verificando status dos serviços..."

# Verificar API
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    log "✅ API está funcionando (http://localhost:5000)"
else
    error "❌ API não está respondendo"
fi

# Verificar Dashboard
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    log "✅ Dashboard está funcionando (http://localhost:8501)"
else
    warning "⚠️  Dashboard pode estar iniciando ainda..."
fi

# Mostrar logs dos serviços
log "Últimos logs dos serviços:"
echo ""
echo "=== API LOGS ==="
docker-compose logs --tail=10 api
echo ""
echo "=== DASHBOARD LOGS ==="
docker-compose logs --tail=10 dashboard

echo ""
log "🎉 Deploy concluído com sucesso!"
echo ""
echo "📊 Serviços disponíveis:"
echo "   • API: http://localhost:5000"
echo "   • Dashboard: http://localhost:8501"
echo "   • Documentação da API: http://localhost:5000/docs"
echo ""
echo "🔧 Comandos úteis:"
echo "   • Ver logs: docker-compose logs -f"
echo "   • Parar serviços: docker-compose down"
echo "   • Reiniciar: docker-compose restart"
echo "   • Reconstruir: docker-compose up --build -d"
echo ""
echo "📁 Arquivos importantes:"
echo "   • Modelos: ./models/"
echo "   • Dados: ./data/"
echo "   • Logs: ./logs/"
