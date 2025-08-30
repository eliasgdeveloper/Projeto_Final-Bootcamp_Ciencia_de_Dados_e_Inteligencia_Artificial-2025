"""
Script Principal de Treinamento de Modelos
Sistema de Manutenção Preditiva - Bootcamp CDIA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import warnings
from utils import DataProcessor, ModelEvaluator, BusinessAnalyzer
warnings.filterwarnings('ignore')

def train_binary_models(X_train, X_test, y_train, y_test):
    """Treina modelos de classificação binária"""
    print("=== TREINAMENTO - CLASSIFICAÇÃO BINÁRIA ===")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Treinar modelo completo
        model.fit(X_train, y_train)
        
        # Avaliar
        eval_results = evaluator.evaluate_binary_model(model, X_test, y_test, name)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            **eval_results
        }
        
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def train_multilabel_models(X_train, X_test, y_train, y_test):
    """Treina modelos de classificação multilabel"""
    print("\n=== TREINAMENTO - CLASSIFICAÇÃO MULTILABEL ===")
    
    base_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    multilabel_models = {}
    
    for name, base_model in base_models.items():
        print(f"Treinando {name} multilabel...")
        
        # Usar MultiOutputClassifier
        multilabel_model = MultiOutputClassifier(base_model)
        multilabel_model.fit(X_train, y_train)
        
        multilabel_models[name] = multilabel_model
    
    return multilabel_models

def hyperparameter_tuning(X_train, y_train, model_type='gradient_boosting'):
    """Realiza otimização de hiperparâmetros"""
    print(f"\n=== OTIMIZAÇÃO DE HIPERPARÂMETROS - {model_type.upper()} ===")
    
    if model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9]
        }
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    # Grid Search com validação cruzada
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,  # Reduzido para economizar tempo
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Executando Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_models(best_model, scaler, model_info):
    """Salva modelos e informações"""
    print("\n=== SALVANDO MODELOS ===")
    
    # Salvar modelo principal
    joblib.dump(best_model, 'modelo_manutencao_preditiva.pkl')
    print("✓ Modelo principal salvo")
    
    # Salvar scaler
    joblib.dump(scaler, 'scaler_features.pkl')
    print("✓ Scaler salvo")
    
    # Salvar informações do modelo
    import json
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print("✓ Informações do modelo salvas")

def main():
    """Função principal do pipeline de treinamento"""
    print("🚀 INICIANDO PIPELINE DE TREINAMENTO")
    print("=" * 50)
    
    # 1. Carregar e preprocessar dados
    from utils import load_and_prepare_data
    
    X, y_binary, y_multilabel, df_processed, df_test, processor = load_and_prepare_data(
        '01_train-Dataset de treino-Avaliação.csv',
        '02_train-Dataset de avaliação em produção pela API.csv'
    )
    
    # 2. Dividir dados
    print("\n=== DIVISÃO DOS DADOS ===")
    X_train, X_val, y_binary_train, y_binary_val = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train_ml, X_val_ml, y_multilabel_train, y_multilabel_val = train_test_split(
        X, y_multilabel, test_size=0.2, random_state=42
    )
    
    print(f"Treino: {X_train.shape}, Validação: {X_val.shape}")
    
    # 3. Escalonamento
    print("\n=== ESCALONAMENTO ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_ml_scaled = scaler.transform(X_train_ml)
    X_val_ml_scaled = scaler.transform(X_val_ml)
    
    # 4. Treinamento de modelos binários
    binary_results = train_binary_models(
        X_train_scaled, X_val_scaled, y_binary_train, y_binary_val
    )
    
    # 5. Selecionar melhor modelo
    best_model_name = max(binary_results.items(), key=lambda x: x[1]['auc_score'])
    best_model = binary_results[best_model_name[0]]['model']
    
    print(f"\n🏆 MELHOR MODELO: {best_model_name[0]}")
    print(f"AUC Score: {best_model_name[1]['auc_score']:.4f}")
    
    # 6. Otimização de hiperparâmetros do melhor modelo
    if best_model_name[0] == 'Gradient Boosting':
        optimized_model, best_params = hyperparameter_tuning(
            X_train_scaled, y_binary_train, 'gradient_boosting'
        )
    elif best_model_name[0] == 'Random Forest':
        optimized_model, best_params = hyperparameter_tuning(
            X_train_scaled, y_binary_train, 'random_forest'
        )
    else:
        optimized_model = best_model
        best_params = {}
    
    # 7. Avaliação final do modelo otimizado
    print("\n=== AVALIAÇÃO FINAL ===")
    evaluator = ModelEvaluator()
    final_results = evaluator.evaluate_binary_model(
        optimized_model, X_val_scaled, y_binary_val, "Modelo Otimizado"
    )
    
    # 8. Treinamento multilabel
    multilabel_models = train_multilabel_models(
        X_train_ml_scaled, X_val_ml_scaled, y_multilabel_train, y_multilabel_val
    )
    
    # 9. Preparar dados de teste e fazer predições
    if df_test is not None:
        print("\n=== PREDIÇÕES NO DATASET DE TESTE ===")
        
        # Preprocessar dados de teste
        df_test_clean = processor.clean_numerical_data(df_test)
        df_test_clean = processor.impute_missing_values(df_test_clean)
        df_test_clean = processor.create_features(df_test_clean)
        
        # Preparar features de teste
        feature_columns = [
            'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
            'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta',
            'temp_diff', 'potencia_estimada', 'eficiencia_termica',
            'desgaste_alto', 'stress_operacional'
        ]
        
        X_test = df_test_clean[feature_columns].copy()
        X_test['tipo_encoded'] = processor.le_tipo.transform(df_test_clean['tipo'])
        X_test_scaled = scaler.transform(X_test)
        
        # Fazer predições
        pred_binary = optimized_model.predict(X_test_scaled)
        pred_binary_proba = optimized_model.predict_proba(X_test_scaled)[:, 1]
        
        # Criar DataFrame de resultados
        results_df = df_test.copy()
        results_df['falha_predita'] = pred_binary
        results_df['probabilidade_falha'] = pred_binary_proba
        
        # Salvar predições
        results_df.to_csv('predicoes_manutencao_preditiva.csv', index=False)
        print("✓ Predições salvas em 'predicoes_manutencao_preditiva.csv'")
        
        # Análise de negócio
        analyzer = BusinessAnalyzer()
        analyzer.analyze_failure_patterns(df_processed, results_df)
        maintenance_info = analyzer.generate_maintenance_recommendations(results_df)
        cost_analysis = analyzer.calculate_maintenance_cost_savings(results_df)
    
    # 10. Salvar modelos e informações
    model_info = {
        'best_model_name': best_model_name[0],
        'best_params': best_params,
        'auc_score': final_results['auc_score'],
        'feature_names': processor.feature_names,
        'training_date': pd.Timestamp.now().isoformat(),
        'dataset_shape': X.shape
    }
    
    save_models(optimized_model, scaler, model_info)
    
    print("\n🎉 PIPELINE DE TREINAMENTO CONCLUÍDO!")
    print(f"Modelo final: {best_model_name[0]} (AUC: {final_results['auc_score']:.4f})")
    
    return optimized_model, scaler, final_results

if __name__ == "__main__":
    # Executar pipeline completo
    model, scaler, results = main()
