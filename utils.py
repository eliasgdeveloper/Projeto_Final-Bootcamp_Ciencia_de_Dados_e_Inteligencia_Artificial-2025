"""
Utilitários para o Sistema de Manutenção Preditiva
Funções auxiliares para limpeza de dados, feature engineering e análise
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Classe para processamento e limpeza dos dados"""
    
    def __init__(self):
        self.le_tipo = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = [
            'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta',
            'temp_diff', 'potencia_estimada', 'eficiencia_termica',
            'desgaste_alto', 'stress_operacional', 'tipo_encoded'
        ]
    
    def clean_categorical_data(self, df):
        """Limpa e padroniza dados categóricos"""
        df_clean = df.copy()
        
        # Padronizar falha_maquina
        df_clean['falha_maquina'] = df_clean['falha_maquina'].astype(str).str.lower()
        df_clean['falha_maquina'] = df_clean['falha_maquina'].map({
            'sim': 1, 'não': 0, 'nao': 0, 'n': 0, 'false': 0, 'true': 1,
            '1': 1, '0': 0, 'y': 1
        })
        
        # Padronizar colunas de falhas específicas
        failure_columns = [
            'FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)', 
            'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)'
        ]
        
        for col in failure_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower()
                df_clean[col] = df_clean[col].map({
                    'true': 1, 'false': 0, 'sim': 1, 'não': 0, 'nao': 0, 
                    'n': 0, '0': 0, '1': 1, '-': 0
                })
        
        return df_clean
    
    def clean_numerical_data(self, df):
        """Limpa dados numéricos"""
        df_clean = df.copy()
        
        numeric_columns = [
            'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta'
        ]
        
        # Converter para numérico
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Corrigir valores fisicamente impossíveis
        df_clean.loc[df_clean['temperatura_ar'] < 250, 'temperatura_ar'] = np.nan
        df_clean.loc[df_clean['velocidade_rotacional'] < 0, 'velocidade_rotacional'] = np.nan
        df_clean.loc[df_clean['desgaste_da_ferramenta'] < 0, 'desgaste_da_ferramenta'] = np.nan
        df_clean.loc[df_clean['umidade_relativa'] > 100, 'umidade_relativa'] = 100
        
        return df_clean
    
    def impute_missing_values(self, df):
        """Imputa valores ausentes"""
        df_clean = df.copy()
        
        numeric_columns = [
            'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta'
        ]
        
        # Imputar por mediana do tipo de máquina
        for col in numeric_columns:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                for machine_type in df_clean['tipo'].unique():
                    mask = (df_clean['tipo'] == machine_type) & df_clean[col].isnull()
                    median_value = df_clean[df_clean['tipo'] == machine_type][col].median()
                    df_clean.loc[mask, col] = median_value
        
        return df_clean
    
    def create_features(self, df):
        """Cria features derivadas"""
        df_features = df.copy()
        
        # Features derivadas
        df_features['temp_diff'] = df_features['temperatura_processo'] - df_features['temperatura_ar']
        df_features['potencia_estimada'] = df_features['torque'] * df_features['velocidade_rotacional']
        df_features['eficiencia_termica'] = df_features['temperatura_processo'] / df_features['temperatura_ar']
        
        # Indicador de desgaste alto
        desgaste_threshold = df_features['desgaste_da_ferramenta'].quantile(0.75)
        df_features['desgaste_alto'] = (df_features['desgaste_da_ferramenta'] > desgaste_threshold).astype(int)
        
        # Stress operacional
        stress_vars = ['temperatura_processo', 'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
        scaler_stress = StandardScaler()
        stress_normalized = scaler_stress.fit_transform(df_features[stress_vars])
        df_features['stress_operacional'] = np.mean(stress_normalized, axis=1)
        
        return df_features
    
    def prepare_features_and_targets(self, df):
        """Prepara features e targets para modelagem"""
        # Features
        feature_columns = [
            'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta',
            'temp_diff', 'potencia_estimada', 'eficiencia_termica',
            'desgaste_alto', 'stress_operacional'
        ]
        
        X = df[feature_columns].copy()
        
        # Codificar tipo de máquina
        X['tipo_encoded'] = self.le_tipo.fit_transform(df['tipo'])
        
        # Targets
        target_columns = [
            'FDF (Falha Desgaste Ferramenta)', 'FDC (Falha Dissipacao Calor)',
            'FP (Falha Potencia)', 'FTE (Falha Tensao Excessiva)', 'FA (Falha Aleatoria)'
        ]
        
        y_multilabel = df[target_columns]
        y_binary = df['falha_maquina']
        
        # Remover linhas com NaN nos targets
        valid_idx = ~(y_binary.isna() | y_multilabel.isna().any(axis=1))
        X_clean = X[valid_idx]
        y_binary_clean = y_binary[valid_idx]
        y_multilabel_clean = y_multilabel[valid_idx]
        
        return X_clean, y_binary_clean, y_multilabel_clean
    
    def full_preprocessing(self, df):
        """Pipeline completo de preprocessamento"""
        print("Iniciando preprocessamento dos dados...")
        
        # 1. Limpeza categórica
        df = self.clean_categorical_data(df)
        print("✓ Dados categóricos limpos")
        
        # 2. Limpeza numérica
        df = self.clean_numerical_data(df)
        print("✓ Dados numéricos limpos")
        
        # 3. Imputação
        df = self.impute_missing_values(df)
        print("✓ Valores ausentes imputados")
        
        # 4. Feature engineering
        df = self.create_features(df)
        print("✓ Features derivadas criadas")
        
        # 5. Preparar features e targets
        X, y_binary, y_multilabel = self.prepare_features_and_targets(df)
        print("✓ Features e targets preparados")
        
        print(f"Shape final: {X.shape}")
        return X, y_binary, y_multilabel, df

class ModelEvaluator:
    """Classe para avaliação de modelos"""
    
    @staticmethod
    def evaluate_binary_model(model, X_test, y_test, model_name="Modelo"):
        """Avalia modelo de classificação binária"""
        print(f"\n=== AVALIAÇÃO: {model_name} ===")
        
        # Predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name="Modelo"):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=15):
        """Plota importância das features"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance_df.head(top_n), x='importance', y='feature')
            plt.title('Importância das Features')
            plt.xlabel('Importância')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        else:
            print("Modelo não possui feature_importances_")
            return None

class BusinessAnalyzer:
    """Classe para análises de negócio"""
    
    @staticmethod
    def analyze_failure_patterns(df, predictions_df=None):
        """Analisa padrões de falha"""
        print("=== ANÁLISE DE PADRÕES DE FALHA ===")
        
        # Análise por tipo de máquina
        if 'falha_maquina' in df.columns:
            failure_by_type = df.groupby('tipo')['falha_maquina'].agg(['count', 'sum', 'mean'])
            print("\nTaxa de falha por tipo de máquina:")
            print(failure_by_type)
        
        # Se há predições, analisar
        if predictions_df is not None:
            print("\n=== ANÁLISE DAS PREDIÇÕES ===")
            total_machines = len(predictions_df)
            failures_predicted = predictions_df['falha_predita'].sum()
            high_risk = (predictions_df['probabilidade_falha'] > 0.7).sum()
            
            print(f"Total de máquinas analisadas: {total_machines}")
            print(f"Falhas previstas: {failures_predicted} ({failures_predicted/total_machines:.2%})")
            print(f"Máquinas de alto risco: {high_risk}")
    
    @staticmethod
    def generate_maintenance_recommendations(predictions_df):
        """Gera recomendações de manutenção"""
        print("\n=== RECOMENDAÇÕES DE MANUTENÇÃO ===")
        
        # Máquinas de alto risco
        high_risk = predictions_df[predictions_df['probabilidade_falha'] > 0.7]
        medium_risk = predictions_df[
            (predictions_df['probabilidade_falha'] > 0.3) & 
            (predictions_df['probabilidade_falha'] <= 0.7)
        ]
        
        print(f"\n1. PRIORIDADE ALTA ({len(high_risk)} máquinas):")
        if len(high_risk) > 0:
            print("   - Parar máquinas para inspeção imediata")
            print("   - Verificar sistema de resfriamento")
            print("   - Examinar desgaste da ferramenta")
            print("   Top 5 máquinas críticas:")
            top_critical = high_risk.nlargest(5, 'probabilidade_falha')[
                ['id', 'id_produto', 'tipo', 'probabilidade_falha']
            ]
            for _, row in top_critical.iterrows():
                print(f"     • {row['id_produto']} (Tipo {row['tipo']}) - Risco: {row['probabilidade_falha']:.3f}")
        
        print(f"\n2. PRIORIDADE MÉDIA ({len(medium_risk)} máquinas):")
        if len(medium_risk) > 0:
            print("   - Aumentar frequência de inspeções")
            print("   - Preparar peças de reposição")
            print("   - Monitoramento intensivo")
        
        return {
            'high_risk_machines': high_risk,
            'medium_risk_machines': medium_risk,
            'total_alerts': len(high_risk) + len(medium_risk)
        }
    
    @staticmethod
    def calculate_maintenance_cost_savings(predictions_df, cost_per_failure=50000, cost_per_prevention=5000):
        """Calcula economia potencial com manutenção preditiva"""
        failures_predicted = predictions_df['falha_predita'].sum()
        
        # Custo sem manutenção preditiva (todas as falhas acontecem)
        cost_without_prediction = failures_predicted * cost_per_failure
        
        # Custo com manutenção preditiva (prevenir falhas de alto risco)
        high_risk_count = (predictions_df['probabilidade_falha'] > 0.7).sum()
        cost_with_prediction = high_risk_count * cost_per_prevention
        
        savings = cost_without_prediction - cost_with_prediction
        
        print(f"\n=== ANÁLISE ECONÔMICA ===")
        print(f"Falhas previstas: {failures_predicted}")
        print(f"Máquinas de alto risco: {high_risk_count}")
        print(f"Custo sem predição: R$ {cost_without_prediction:,.2f}")
        print(f"Custo com predição: R$ {cost_with_prediction:,.2f}")
        print(f"Economia potencial: R$ {savings:,.2f}")
        print(f"ROI: {(savings/cost_with_prediction)*100:.1f}%")
        
        return {
            'cost_without_prediction': cost_without_prediction,
            'cost_with_prediction': cost_with_prediction,
            'savings': savings,
            'roi': (savings/cost_with_prediction)*100 if cost_with_prediction > 0 else 0
        }

def load_and_prepare_data(train_path, test_path=None):
    """Função principal para carregar e preparar dados"""
    processor = DataProcessor()
    
    # Carregar dados
    print("Carregando dados...")
    df_train = pd.read_csv(train_path)
    
    if test_path:
        df_test = pd.read_csv(test_path)
        print(f"Dados carregados - Treino: {df_train.shape}, Teste: {df_test.shape}")
    else:
        df_test = None
        print(f"Dados carregados - Treino: {df_train.shape}")
    
    # Preprocessar dados de treino
    X, y_binary, y_multilabel, df_processed = processor.full_preprocessing(df_train)
    
    return X, y_binary, y_multilabel, df_processed, df_test, processor

if __name__ == "__main__":
    # Exemplo de uso
    print("Utilitários de Manutenção Preditiva carregados com sucesso!")
    print("Use as classes DataProcessor, ModelEvaluator e BusinessAnalyzer para análises.")
