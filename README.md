Autor: **Elias Gomes**  
# Projeto Final-Bootcamp Ciência de Dados e Inteligência Artificial - 2025 - SENAI/SC

## 🌐 Acesse o Dashboard Interativo  

Quer explorar os dados de forma dinâmica e interativa?  
Clique no link abaixo para acessar a versão completa online, com filtros, navegação e análises avançadas:  

🔗 **[Ciência_de_Dados_e_Inteligência_Artificial-SENAI/SC]** 
**(https://eliasgdeveloper-projeto-final-bootcamp-ciencia-dashboard-nzxkat.streamlit.app/)**  

💡 No dashboard você pode:  
- Interagir com gráficos dinâmicos e responsivos  
- Filtrar períodos específicos para análises temporais  
- Explorar insights adicionais que vão além do documentado aqui no repositório  

📊 Principais análises disponíveis:  
- **Distribuição de Risco por Tipo de Máquina**  
- **Distribuição de Probabilidades de Falhas**  
- **Top 10 Máquinas de Maior Risco**  
- **Tendência de Falhas ao Longo do Tempo**  
- **Análise de Custos de Manutenção Preditiva vs. Corretiva**  
- **Comparativo de Performance entre Equipamentos**  

**(https://github.com/eliasgdeveloper/Projeto_Final-Bootcamp_Ciencia_de_Dados_e_Inteligencia_Artificial-2025)**


Sistema completo para predição de falhas em máquinas industriais usando dados de sensores IoT, desenvolvido com Machine Learning e tecnologias modernas.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.68+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.25+-red.svg)
![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)

## 📊 Visão Geral

O sistema utiliza dados de sensores IoT para prever falhas em máquinas industriais com **92.36% de precisão (AUC)**, identificando 5 tipos específicos de falhas:

- **FDF**: Falha por Desgaste da Ferramenta
- **FDC**: Falha por Dissipação de Calor  
- **FP**: Falha por Potência
- **FTE**: Falha por Tensão Excessiva
- **FA**: Falha Aleatória

## 🚀 Funcionalidades

### ✅ Core Features
- **Análise Exploratória Completa** - Identificação de padrões e anomalias
- **Limpeza Inteligente de Dados** - Tratamento de valores ausentes e outliers
- **Feature Engineering Avançada** - Criação de 5+ features derivadas
- **Múltiplos Algoritmos ML** - Random Forest, Gradient Boosting, XGBoost
- **Validação Cruzada** - Avaliação robusta dos modelos
- **Predições em Tempo Real** - API REST para integração

### 🎯 Features Extras (Diferenciais)
- **Dashboard Interativo** - Streamlit com 4 páginas especializadas
- **API REST Completa** - FastAPI com documentação automática
- **Scripts Organizados** - Código modular e reutilizável
- **Containerização Docker** - Deploy simplificado
- **Análise de ROI** - Cálculo de economia potencial

## 📁 Estrutura do Projeto

```
📦 sistema-manutencao-preditiva/
├── 📊 data/
│   ├── 01_train-Dataset de treino-Avaliação.csv
│   ├── 02_train-Dataset de avaliação em produção pela API.csv
│   └── predicoes_manutencao_preditiva.csv
├── 🤖 models/
│   ├── modelo_manutencao_preditiva.pkl
│   ├── scaler_features.pkl
│   └── model_info.json
├── 📓 notebooks/
│   └── Projeto_Manutencao_Preditiva.ipynb
├── 🐍 src/
│   ├── api.py                    # API REST FastAPI
│   ├── dashboard.py              # Dashboard Streamlit
│   ├── train_model.py            # Script de treinamento
│   └── utils.py                  # Utilitários e classes
├── 🐳 docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── deploy.sh
├── 📋 requirements.txt
└── 📖 README.md
```

## 🔧 Instalação e Configuração

### Opção 1: Instalação Local

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/sistema-manutencao-preditiva.git
cd sistema-manutencao-preditiva

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Treinar modelo (se necessário)
python train_model.py

# 5. Iniciar API
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 6. Iniciar Dashboard (novo terminal)
streamlit run dashboard.py --server.port 8501
```

### Opção 2: Docker (Recomendado)

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/sistema-manutencao-preditiva.git
cd sistema-manutencao-preditiva

# 2. Deploy completo
chmod +x deploy.sh
./deploy.sh
```

## 📈 Como Usar

### 🌐 API REST

A API oferece endpoints para predição individual e em lote:

```python
import requests

# Predição individual
data = {
    "tipo": "L",
    "temperatura_ar": 298.3,
    "temperatura_processo": 309.1,
    "umidade_relativa": 90.0,
    "velocidade_rotacional": 1616.0,
    "torque": 31.1,
    "desgaste_da_ferramenta": 195.0
}

response = requests.post("http://localhost:8000/predict", json=data)
resultado = response.json()

print(f"Probabilidade de falha: {resultado['probabilidade_falha']:.4f}")
print(f"Recomendação: {resultado['recomendacao']}")
```

### 📊 Dashboard

Acesse `http://localhost:8501` para o dashboard interativo com:

1. **Dashboard Principal** - Visão geral e métricas
2. **Predição Individual** - Interface para testar máquinas
3. **Análise Detalhada** - Filtros e visualizações avançadas
4. **Monitoramento em Tempo Real** - Simulação de ambiente produtivo

### 🐍 Scripts Python

```python
from utils import DataProcessor, ModelEvaluator, BusinessAnalyzer

# Preprocessar dados
processor = DataProcessor()
X, y_binary, y_multilabel, df_clean = processor.full_preprocessing(df)

# Avaliar modelo
evaluator = ModelEvaluator()
results = evaluator.evaluate_binary_model(model, X_test, y_test)

# Análise de negócio
analyzer = BusinessAnalyzer()
recommendations = analyzer.generate_maintenance_recommendations(predictions)
```

## 🎯 Resultados Alcançados

### 📊 Performance do Modelo
- **AUC Score**: 0.9236 (92.36%)
- **Precisão**: 98% para máquinas saudáveis
- **Recall**: 42% para detecção de falhas
- **F1-Score**: 0.49 para classificação de falhas

### 💰 Impacto Econômico
- **Máquinas Analisadas**: 7.173
- **Falhas Previstas**: 168 (2.34%)
- **Máquinas Alto Risco**: 109
- **ROI Estimado**: 900%+ em economia de manutenção

### 🎯 Insights Principais
1. **Máquinas tipo L** apresentam maior taxa de falha (2.68%)
2. **Falhas de dissipação de calor** são as mais frequentes (1.10%)
3. **Sistema de resfriamento** é ponto crítico de atenção
4. **Monitoramento de temperatura** deve ser prioritário

## 🔬 Metodologia Técnica

### 📋 Pipeline de Dados
1. **Análise Exploratória** - Identificação de padrões e problemas
2. **Limpeza de Dados** - Tratamento de valores impossíveis
3. **Feature Engineering** - Criação de variáveis derivadas
4. **Balanceamento** - Tratamento de classes desbalanceadas
5. **Validação** - Cross-validation estratificada

### 🤖 Modelagem
- **Algoritmos Testados**: Random Forest, Gradient Boosting, XGBoost, Logistic Regression
- **Abordagem**: Classificação multilabel + binária
- **Otimização**: Grid Search para hiperparâmetros
- **Validação**: Holdout + Cross-validation

### 📏 Métricas de Avaliação
- **AUC-ROC**: Principal métrica (dados desbalanceados)
- **Precision/Recall**: Análise de trade-offs
- **F1-Score**: Medida harmônica
- **Confusion Matrix**: Análise detalhada de erros

## 🚀 Próximos Passos

### 🔄 Melhorias Técnicas
- [ ] Implementar modelos ensemble avançados
- [ ] Adicionar explicabilidade (SHAP, LIME)
- [ ] Pipeline de retreino automático
- [ ] Monitoramento de drift dos dados
- [ ] Integração com bancos de dados reais

### 🏭 Expansão Funcional
- [ ] Predição de tempo até falha
- [ ] Otimização de cronograma de manutenção
- [ ] Integração com sistemas ERP
- [ ] Alertas automáticos via email/SMS
- [ ] Dashboard mobile

### ☁️ Infraestrutura
- [ ] Deploy em cloud (AWS/Azure/GCP)
- [ ] CI/CD pipeline completo
- [ ] Monitoramento com Prometheus/Grafana
- [ ] Backup automático de modelos
- [ ] Escalabilidade horizontal

## 👥 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **Bootcamp CDIA** pela oportunidade de aprendizado
- **Comunidade Open Source** pelas ferramentas utilizadas
- **Instrutores** pelo suporte e orientação

## 📞 Contato

**Desenvolvedor**: Elias Gomes
- 📧 Email: eliasgdeveloper@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/eliasgomesdeveloper
- 🐙 GitHub: https://github.com/eliasgdeveloper

---

<div align="center">

**🔧 Sistema de Manutenção Preditiva** | Bootcamp CDIA 2025

Desenvolvido com ❤️ e muito ☕

</div>
