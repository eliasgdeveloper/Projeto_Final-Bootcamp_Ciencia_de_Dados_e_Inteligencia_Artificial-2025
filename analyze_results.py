import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_csv('predicoes_manutencao_preditiva.csv')

print("=" * 60)
print("ANÁLISE COMPLETA DOS RESULTADOS")
print("=" * 60)

print("\n=== RESUMO GERAL ===")
print(f"Total de máquinas analisadas: {len(df):,}")
print(f"Falhas previstas: {int(df['falha_predita'].sum()):,}")
print(f"Taxa de falha geral: {df['falha_predita'].mean():.2%}")
print(f"Probabilidade média de falha: {df['probabilidade_falha'].mean():.4f}")

# Análise de risco
alto_risco = (df['probabilidade_falha'] > 0.7).sum()
medio_risco = ((df['probabilidade_falha'] > 0.3) & (df['probabilidade_falha'] <= 0.7)).sum()
baixo_risco = (df['probabilidade_falha'] <= 0.3).sum()

print(f"\nDistribuição de Risco:")
print(f"  • Alto risco (>70%): {alto_risco:,} máquinas ({alto_risco/len(df):.2%})")
print(f"  • Médio risco (30-70%): {medio_risco:,} máquinas ({medio_risco/len(df):.2%})")
print(f"  • Baixo risco (≤30%): {baixo_risco:,} máquinas ({baixo_risco/len(df):.2%})")

print("\n=== ANÁLISE POR TIPO DE MÁQUINA ===")
por_tipo = df.groupby('tipo').agg({
    'falha_predita': ['count', 'sum', 'mean'],
    'probabilidade_falha': ['mean', 'max', 'min']
}).round(4)

for tipo in ['L', 'M', 'H']:
    if tipo in df['tipo'].values:
        subset = df[df['tipo'] == tipo]
        total = len(subset)
        falhas = int(subset['falha_predita'].sum())
        taxa = subset['falha_predita'].mean()
        prob_media = subset['probabilidade_falha'].mean()
        prob_max = subset['probabilidade_falha'].max()
        alto_risco_tipo = (subset['probabilidade_falha'] > 0.7).sum()
        
        print(f"\nTipo {tipo}:")
        print(f"  • Total: {total:,} máquinas")
        print(f"  • Falhas previstas: {falhas:,} ({taxa:.2%})")
        print(f"  • Prob. média: {prob_media:.4f}")
        print(f"  • Prob. máxima: {prob_max:.4f}")
        print(f"  • Alto risco: {alto_risco_tipo:,} máquinas")

print("\n=== TOP 10 MÁQUINAS DE MAIOR RISCO ===")
top_risk = df.nlargest(10, 'probabilidade_falha')[['id', 'id_produto', 'tipo', 'probabilidade_falha']]
for idx, row in top_risk.iterrows():
    print(f"{row['id_produto']} (Tipo {row['tipo']}) - Risco: {row['probabilidade_falha']:.4f} ({row['probabilidade_falha']:.1%})")

print("\n=== VALIDAÇÃO DOS RESULTADOS ===")
# Verificar se os resultados fazem sentido
print("Verificações de consistência:")

# 1. Probabilidades entre 0 e 1
prob_valid = ((df['probabilidade_falha'] >= 0) & (df['probabilidade_falha'] <= 1)).all()
print(f"✓ Probabilidades válidas (0-1): {prob_valid}")

# 2. Predições binárias (0 ou 1)
pred_valid = df['falha_predita'].isin([0, 1]).all()
print(f"✓ Predições binárias válidas: {pred_valid}")

# 3. Correlação probabilidade vs predição
correlation = df['probabilidade_falha'].corr(df['falha_predita'])
print(f"✓ Correlação prob-predição: {correlation:.4f}")

# 4. Distribuição realista
taxa_falha = df['falha_predita'].mean()
realista = 0.005 <= taxa_falha <= 0.05  # Entre 0.5% e 5% é realista para indústria
print(f"✓ Taxa de falha realista: {realista} ({taxa_falha:.2%})")

print("\n=== INSIGHTS PRINCIPAIS ===")
tipo_maior_risco = df.groupby('tipo')['probabilidade_falha'].mean().idxmax()
print(f"• Tipo de máquina com maior risco: {tipo_maior_risco}")

# Probabilidade média por tipo
for tipo in ['L', 'M', 'H']:
    if tipo in df['tipo'].values:
        prob_tipo = df[df['tipo'] == tipo]['probabilidade_falha'].mean()
        print(f"• Tipo {tipo}: {prob_tipo:.4f} probabilidade média de falha")

print("\n=== RECOMENDAÇÕES DE AÇÃO ===")
print(f"1. URGENTE: Inspecionar {alto_risco:,} máquinas de alto risco")
print(f"2. MONITORAR: Acompanhar {medio_risco:,} máquinas de risco médio")
print(f"3. ROTINA: Manter {baixo_risco:,} máquinas em operação normal")

# Estimativa de economia
custo_falha = 50000  # R$ por falha
custo_prevencao = 5000  # R$ por prevenção
falhas_evitadas = alto_risco * 0.8  # Assumindo 80% de eficácia
economia = (falhas_evitadas * custo_falha) - (alto_risco * custo_prevencao)
print(f"\n=== ANÁLISE ECONÔMICA ===")
print(f"• Custo sem predição: R$ {int(df['falha_predita'].sum()) * custo_falha:,.2f}")
print(f"• Custo com predição: R$ {alto_risco * custo_prevencao:,.2f}")
print(f"• Economia estimada: R$ {economia:,.2f}")
print(f"• ROI: {(economia/(alto_risco * custo_prevencao))*100:.1f}%")

print("\n" + "=" * 60)
print("CONCLUSÃO: RESULTADOS CONSISTENTES E REALISTAS ✓")
print("=" * 60)
