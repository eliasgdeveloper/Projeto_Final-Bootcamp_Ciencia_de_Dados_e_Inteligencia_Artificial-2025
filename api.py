"""
API REST para Sistema de Manutenção Preditiva
Bootcamp CDIA - Projeto Final
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar aplicação FastAPI
app = FastAPI(
    title="Sistema de Manutenção Preditiva",
    description="API para predição de falhas em máquinas industriais usando IoT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de dados Pydantic
class MachineInput(BaseModel):
    """Dados de entrada para predição de uma máquina"""
    tipo: str = Field(..., description="Tipo da máquina", pattern="^[LMH]$")
    temperatura_ar: float = Field(..., description="Temperatura do ar em Kelvin", ge=250, le=320)
    temperatura_processo: float = Field(..., description="Temperatura do processo em Kelvin", ge=250, le=320)
    umidade_relativa: float = Field(..., description="Umidade relativa em %", ge=0, le=100)
    velocidade_rotacional: float = Field(..., description="Velocidade rotacional em RPM", ge=0, le=5000)
    torque: float = Field(..., description="Torque em Nm", ge=0, le=100)
    desgaste_da_ferramenta: float = Field(..., description="Desgaste da ferramenta em minutos", ge=0, le=500)

    class Config:
        schema_extra = {
            "example": {
                "tipo": "L",
                "temperatura_ar": 298.3,
                "temperatura_processo": 309.1,
                "umidade_relativa": 90.0,
                "velocidade_rotacional": 1616.0,
                "torque": 31.1,
                "desgaste_da_ferramenta": 195.0
            }
        }

class BatchInput(BaseModel):
    """Dados para predição em lote"""
    machines: List[MachineInput] = Field(..., description="Lista de máquinas para predição")

class PredictionOutput(BaseModel):
    """Resultado da predição"""
    machine_id: Optional[str] = Field(None, description="ID da máquina (se fornecido)")
    falha_prevista: int = Field(..., description="Predição binária: 0=sem falha, 1=com falha")
    probabilidade_falha: float = Field(..., description="Probabilidade de falha (0-1)")
    nivel_risco: str = Field(..., description="Nível de risco: BAIXO, MÉDIO, ALTO")
    recomendacao: str = Field(..., description="Recomendação baseada no risco")
    timestamp: str = Field(..., description="Timestamp da predição")

class BatchOutput(BaseModel):
    """Resultado de predições em lote"""
    total_machines: int = Field(..., description="Total de máquinas analisadas")
    predictions: List[PredictionOutput] = Field(..., description="Lista de predições")
    summary: Dict = Field(..., description="Resumo estatístico")

class HealthCheck(BaseModel):
    """Status da API"""
    status: str = Field(..., description="Status da API")
    timestamp: str = Field(..., description="Timestamp da verificação")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    version: str = Field(..., description="Versão da API")

# Variáveis globais para modelo
model = None
scaler = None
feature_names = None

def load_models():
    """Carrega modelo e scaler"""
    global model, scaler, feature_names
    try:
        model = joblib.load('modelo_manutencao_preditiva.pkl')
        scaler = joblib.load('scaler_features.pkl')
        
        # Definir nomes das features na ordem correta
        feature_names = [
            'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta',
            'temp_diff', 'potencia_estimada', 'eficiencia_termica',
            'desgaste_alto', 'stress_operacional', 'tipo_encoded'
        ]
        
        logger.info("Modelo e scaler carregados com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        return False

def prepare_features(machine: MachineInput) -> np.ndarray:
    """Prepara features para predição"""
    # Codificar tipo de máquina
    tipo_encoded = {'L': 0, 'M': 1, 'H': 2}[machine.tipo]
    
    # Calcular features derivadas
    temp_diff = machine.temperatura_processo - machine.temperatura_ar
    potencia_estimada = machine.torque * machine.velocidade_rotacional
    eficiencia_termica = machine.temperatura_processo / machine.temperatura_ar
    desgaste_alto = 1 if machine.desgaste_da_ferramenta > 155 else 0
    
    # Stress operacional (simplificado - em produção seria calculado com mais precisão)
    stress_vars = [
        machine.temperatura_processo / 313.8,  # normalizado pelo máximo
        machine.velocidade_rotacional / 2886.0,
        machine.torque / 76.6,
        machine.desgaste_da_ferramenta / 253.0
    ]
    stress_operacional = np.mean(stress_vars)
    
    # Criar array de features
    features = np.array([
        machine.temperatura_ar,
        machine.temperatura_processo,
        machine.umidade_relativa,
        machine.velocidade_rotacional,
        machine.torque,
        machine.desgaste_da_ferramenta,
        temp_diff,
        potencia_estimada,
        eficiencia_termica,
        desgaste_alto,
        stress_operacional,
        tipo_encoded
    ]).reshape(1, -1)
    
    return features

def get_risk_level(probability: float) -> str:
    """Determina nível de risco baseado na probabilidade"""
    if probability < 0.3:
        return "BAIXO"
    elif probability < 0.7:
        return "MÉDIO"
    else:
        return "ALTO"

def get_recommendation(probability: float) -> str:
    """Gera recomendação baseada no risco"""
    if probability > 0.7:
        return "AÇÃO IMEDIATA: Parar máquina para inspeção. Verificar sistema de resfriamento e desgaste da ferramenta."
    elif probability > 0.3:
        return "MONITORAMENTO INTENSIVO: Aumentar frequência de inspeções e preparar peças de reposição."
    else:
        return "OPERAÇÃO NORMAL: Manter monitoramento de rotina e cronograma de manutenção preventiva."

# Carregar modelos na inicialização
@app.on_event("startup")
async def startup_event():
    """Inicialização da API"""
    logger.info("Iniciando API de Manutenção Preditiva...")
    success = load_models()
    if not success:
        logger.error("Falha ao carregar modelos na inicialização")

# Endpoints da API

@app.get("/", response_model=HealthCheck)
async def root():
    """Endpoint raiz - verificação de saúde"""
    return HealthCheck(
        status="online",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None and scaler is not None,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Verificação de saúde da API"""
    return HealthCheck(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None and scaler is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict_single(machine: MachineInput):
    """Predição para uma única máquina"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # Preparar features
        features = prepare_features(machine)
        
        # Escalonar features
        features_scaled = scaler.transform(features)
        
        # Fazer predição
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        
        # Determinar nível de risco e recomendação
        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(probability)
        
        return PredictionOutput(
            falha_prevista=int(prediction),
            probabilidade_falha=float(probability),
            nivel_risco=risk_level,
            recomendacao=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    """Predição em lote para múltiplas máquinas"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        predictions = []
        
        for i, machine in enumerate(batch.machines):
            # Preparar features
            features = prepare_features(machine)
            
            # Escalonar features
            features_scaled = scaler.transform(features)
            
            # Fazer predição
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0, 1]
            
            # Determinar nível de risco e recomendação
            risk_level = get_risk_level(probability)
            recommendation = get_recommendation(probability)
            
            predictions.append(PredictionOutput(
                machine_id=f"machine_{i+1}",
                falha_prevista=int(prediction),
                probabilidade_falha=float(probability),
                nivel_risco=risk_level,
                recomendacao=recommendation,
                timestamp=datetime.now().isoformat()
            ))
        
        # Calcular resumo estatístico
        total_failures = sum(p.falha_prevista for p in predictions)
        avg_probability = np.mean([p.probabilidade_falha for p in predictions])
        high_risk_count = sum(1 for p in predictions if p.nivel_risco == "ALTO")
        
        summary = {
            "total_failures_predicted": total_failures,
            "failure_rate": total_failures / len(predictions),
            "average_probability": float(avg_probability),
            "high_risk_machines": high_risk_count,
            "risk_distribution": {
                "BAIXO": sum(1 for p in predictions if p.nivel_risco == "BAIXO"),
                "MÉDIO": sum(1 for p in predictions if p.nivel_risco == "MÉDIO"),
                "ALTO": high_risk_count
            }
        }
        
        return BatchOutput(
            total_machines=len(batch.machines),
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Informações sobre o modelo"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        info = {
            "model_type": type(model).__name__,
            "features": feature_names,
            "feature_count": len(feature_names),
            "scaler_type": type(scaler).__name__ if scaler else None,
            "model_loaded": True
        }
        
        # Tentar obter informações adicionais do modelo
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            info["feature_importance"] = {k: float(v) for k, v in feature_importance.items()}
        
        return info
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Recarrega o modelo"""
    try:
        success = load_models()
        if success:
            return {"message": "Modelo recarregado com sucesso", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=500, detail="Falha ao recarregar modelo")
    except Exception as e:
        logger.error(f"Erro ao recarregar modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# Exemplo de endpoint para estatísticas
@app.get("/stats")
async def get_statistics():
    """Estatísticas simuladas do sistema"""
    try:
        # Em produção, essas estatísticas viriam de um banco de dados
        stats = {
            "total_predictions_today": 1247,
            "total_machines_monitored": 7173,
            "current_alerts": 109,
            "uptime_hours": 24.5,
            "last_maintenance_check": "2025-08-26T12:00:00",
            "system_health": "optimal"
        }
        return stats
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
