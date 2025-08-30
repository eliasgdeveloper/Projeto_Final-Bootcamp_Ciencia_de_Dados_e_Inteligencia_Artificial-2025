"""
Dashboard Streamlit para Sistema de Manutenção Preditiva
Bootcamp CDIA - Projeto Final
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sistema de Manutenção Preditiva",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔧 Sistema Inteligente de Manutenção Preditiva")
st.markdown("---")

@st.cache_data
def load_data():
    """Carrega os dados de predições"""
    try:
        df = pd.read_csv('predicoes_manutencao_preditiva.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo de predições não encontrado!")
        return None

@st.cache_resource
def load_models():
    """Carrega modelo e scaler"""
    try:
        model = joblib.load('modelo_manutencao_preditiva.pkl')
        scaler = joblib.load('scaler_features.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Modelos não encontrados!")
        return None, None

def create_risk_gauge(probability):
    """Cria um gauge de risco"""
    if probability < 0.3:
        color = "green"
        risk_level = "BAIXO"
    elif probability < 0.7:
        color = "yellow"
        risk_level = "MÉDIO"
    else:
        color = "red"
        risk_level = "ALTO"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Risco de Falha: {risk_level}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def predict_single_machine(model, scaler, features):
    """Faz predição para uma única máquina"""
    if model is None or scaler is None:
        return None, None
    
    # Preparar features
    features_scaled = scaler.transform([features])
    
    # Predição
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0, 1]
    
    return prediction, probability

# Sidebar
st.sidebar.header("📊 Navegação")
page = st.sidebar.selectbox(
    "Escolha uma página:",
    ["Dashboard Principal", "Predição Individual", "Análise Detalhada", "Monitoramento em Tempo Real"]
)

# Carregar dados
df = load_data()
model, scaler = load_models()

if df is not None:
    
    if page == "Dashboard Principal":
        st.header("📈 Visão Geral do Sistema")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_machines = len(df)
            st.metric("Total de Máquinas", total_machines)
        
        with col2:
            failures_predicted = df['falha_predita'].sum()
            st.metric("Falhas Previstas", int(failures_predicted))
        
        with col3:
            failure_rate = (failures_predicted / total_machines) * 100
            st.metric("Taxa de Falha", f"{failure_rate:.2f}%")
        
        with col4:
            high_risk = (df['probabilidade_falha'] > 0.7).sum()
            st.metric("Máquinas Alto Risco", int(high_risk))
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição de Risco por Tipo de Máquina")
            risk_by_type = df.groupby('tipo')['probabilidade_falha'].mean().reset_index()
            fig1 = px.bar(
                risk_by_type, 
                x='tipo', 
                y='probabilidade_falha',
                title="Probabilidade Média de Falha por Tipo",
                color='probabilidade_falha',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Distribuição de Probabilidades")
            fig2 = px.histogram(
                df, 
                x='probabilidade_falha',
                nbins=50,
                title="Distribuição das Probabilidades de Falha"
            )
            fig2.add_vline(x=0.7, line_dash="dash", line_color="red", 
                          annotation_text="Limite Alto Risco")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Top máquinas de risco
        st.subheader("🚨 Top 10 Máquinas de Alto Risco")
        top_risk = df.nlargest(10, 'probabilidade_falha')[
            ['id', 'id_produto', 'tipo', 'probabilidade_falha']
        ]
        top_risk['Risco (%)'] = (top_risk['probabilidade_falha'] * 100).round(2)
        st.dataframe(top_risk, use_container_width=True)
    
    elif page == "Predição Individual":
        st.header("🔍 Predição para Máquina Individual")
        
        if model is not None and scaler is not None:
            st.subheader("Insira os Parâmetros da Máquina")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tipo = st.selectbox("Tipo de Máquina", ['L', 'M', 'H'])
                temp_ar = st.number_input("Temperatura do Ar (K)", 
                                         min_value=250.0, max_value=320.0, value=300.0)
                temp_processo = st.number_input("Temperatura do Processo (K)", 
                                               min_value=250.0, max_value=320.0, value=310.0)
                umidade = st.number_input("Umidade Relativa (%)", 
                                         min_value=80.0, max_value=100.0, value=90.0)
            
            with col2:
                velocidade = st.number_input("Velocidade Rotacional (RPM)", 
                                           min_value=1000, max_value=3000, value=1500)
                torque = st.number_input("Torque (Nm)", 
                                        min_value=3.0, max_value=80.0, value=40.0)
                desgaste = st.number_input("Desgaste da Ferramenta (min)", 
                                          min_value=0, max_value=300, value=100)
            
            if st.button("🔮 Fazer Predição", type="primary"):
                # Preparar features (mesma ordem do treinamento)
                tipo_encoded = {'L': 0, 'M': 1, 'H': 2}[tipo]
                
                # Features derivadas
                temp_diff = temp_processo - temp_ar
                potencia_estimada = torque * velocidade
                eficiencia_termica = temp_processo / temp_ar
                desgaste_alto = 1 if desgaste > 155 else 0  # threshold do treino
                
                # Stress operacional (simplificado)
                stress_operacional = 0.5  # valor médio
                
                features = [
                    temp_ar, temp_processo, umidade, velocidade, torque, desgaste,
                    temp_diff, potencia_estimada, eficiencia_termica, 
                    desgaste_alto, stress_operacional, tipo_encoded
                ]
                
                prediction, probability = predict_single_machine(model, scaler, features)
                
                if prediction is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Resultado da Predição")
                        if prediction == 1:
                            st.error("⚠️ FALHA PREVISTA")
                        else:
                            st.success("✅ MÁQUINA SAUDÁVEL")
                        
                        st.metric("Probabilidade de Falha", f"{probability:.4f}")
                        
                    with col2:
                        st.subheader("🎯 Gauge de Risco")
                        gauge_fig = create_risk_gauge(probability)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Recomendações
                    st.subheader("💡 Recomendações")
                    if probability > 0.7:
                        st.error("""
                        🚨 **AÇÃO IMEDIATA NECESSÁRIA**
                        - Parar a máquina para inspeção
                        - Verificar sistema de resfriamento
                        - Examinar desgaste da ferramenta
                        - Agendar manutenção preventiva
                        """)
                    elif probability > 0.3:
                        st.warning("""
                        ⚠️ **MONITORAMENTO INTENSIVO**
                        - Aumentar frequência de inspeções
                        - Verificar parâmetros operacionais
                        - Preparar peças de reposição
                        """)
                    else:
                        st.success("""
                        ✅ **OPERAÇÃO NORMAL**
                        - Manter monitoramento de rotina
                        - Seguir cronograma de manutenção preventiva
                        """)
        else:
            st.error("Modelo não carregado. Verifique os arquivos.")
    
    elif page == "Análise Detalhada":
        st.header("📊 Análise Detalhada dos Dados")
        
        # Filtros
        st.subheader("🔍 Filtros")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tipos_selecionados = st.multiselect(
                "Tipos de Máquina", 
                df['tipo'].unique(), 
                default=df['tipo'].unique()
            )
        
        with col2:
            min_prob = st.slider("Probabilidade Mínima", 0.0, 1.0, 0.0)
        
        with col3:
            max_prob = st.slider("Probabilidade Máxima", 0.0, 1.0, 1.0)
        
        # Filtrar dados
        df_filtered = df[
            (df['tipo'].isin(tipos_selecionados)) &
            (df['probabilidade_falha'] >= min_prob) &
            (df['probabilidade_falha'] <= max_prob)
        ]
        
        st.subheader(f"📈 Análise de {len(df_filtered)} Máquinas")
        
        # Visualizações avançadas
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot por tipo
            fig3 = px.box(
                df_filtered, 
                x='tipo', 
                y='probabilidade_falha',
                title="Distribuição de Risco por Tipo de Máquina"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Scatter plot
            fig4 = px.scatter(
                df_filtered, 
                x='id', 
                y='probabilidade_falha',
                color='tipo',
                title="Probabilidade de Falha por ID da Máquina"
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Tabela de dados filtrados
        st.subheader("📋 Dados Filtrados")
        st.dataframe(df_filtered, use_container_width=True)
        
        # Download dos dados filtrados
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="dados_filtrados.csv",
            mime="text/csv"
        )
    
    elif page == "Monitoramento em Tempo Real":
        st.header("🔄 Monitoramento em Tempo Real")
        
        # Simulação de dados em tempo real
        if st.button("🔄 Atualizar Dados"):
            st.rerun()
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)")
        if auto_refresh:
            st.rerun()
        
        # Status atual
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Máquinas Online", len(df))
        
        with col2:
            alertas = (df['probabilidade_falha'] > 0.7).sum()
            st.metric("Alertas Ativos", int(alertas), delta=np.random.randint(-2, 3))
        
        with col3:
            media_risco = df['probabilidade_falha'].mean()
            st.metric("Risco Médio", f"{media_risco:.3f}", 
                     delta=f"{np.random.uniform(-0.01, 0.01):.3f}")
        
        # Mapa de calor das máquinas
        st.subheader("🗺️ Mapa de Risco das Máquinas")
        
        # Criar dados simulados para mapa de calor
        n_rows, n_cols = 10, 15
        risk_matrix = np.random.choice(df['probabilidade_falha'].values, size=(n_rows, n_cols))
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Probabilidade de Falha")
        ))
        
        fig_heatmap.update_layout(
            title="Mapa de Calor - Risco de Falhas por Localização",
            xaxis_title="Posição X",
            yaxis_title="Posição Y"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Timeline de eventos
        st.subheader("📅 Timeline de Eventos")
        
        # Simular eventos
        import datetime
        eventos = [
            {"Tempo": "2025-08-26 14:30", "Máquina": "L56434", "Evento": "Alto risco detectado", "Prioridade": "Alta"},
            {"Tempo": "2025-08-26 14:25", "Máquina": "M18727", "Evento": "Manutenção agendada", "Prioridade": "Média"},
            {"Tempo": "2025-08-26 14:20", "Máquina": "H35319", "Evento": "Parâmetros normalizados", "Prioridade": "Baixa"},
        ]
        
        for evento in eventos:
            col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
            with col1:
                st.text(evento["Tempo"])
            with col2:
                st.text(evento["Máquina"])
            with col3:
                st.text(evento["Evento"])
            with col4:
                if evento["Prioridade"] == "Alta":
                    st.error("🔴")
                elif evento["Prioridade"] == "Média":
                    st.warning("🟡")
                else:
                    st.success("🟢")

else:
    st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo 'predicoes_manutencao_preditiva.csv' existe.")

# Footer
st.markdown("---")
st.markdown("**Sistema de Manutenção Preditiva** | Bootcamp CDIA 2025 | Desenvolvido com ❤️ e Streamlit")
