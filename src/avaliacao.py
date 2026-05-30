import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não interativo
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import src.configuracao as config

def avaliar_pipelines():
    print("=== Iniciando Pipeline de Avaliação em Português ===")
    
    # 1. Carregar dados de teste e pipelines salvos
    caminho_dados_teste = os.path.join(config.MODEL_DIR, 'dados_teste.pkl')
    caminho_modelo_lr = os.path.join(config.MODEL_DIR, 'modelo_baseline.pkl')
    caminho_modelo_lgb = os.path.join(config.MODEL_DIR, 'modelo_champion.pkl')
    caminho_modelo_xgb = os.path.join(config.MODEL_DIR, 'modelo_xgboost.pkl')
    
    if not all(os.path.exists(p) for p in [caminho_dados_teste, caminho_modelo_lr, caminho_modelo_lgb, caminho_modelo_xgb]):
        print("Erro: Arquivos de modelos ou dados de teste não encontrados. Execute treinamento.py primeiro.")
        return

    with open(caminho_dados_teste, 'rb') as f:
        dados_teste = pickle.load(f)
    X_teste_raw = dados_teste['X_teste_raw']
    y_teste = dados_teste['y_teste']
    
    with open(caminho_modelo_lr, 'rb') as f:
        pipeline_lr = pickle.load(f)
        
    with open(caminho_modelo_lgb, 'rb') as f:
        pipeline_lgb = pickle.load(f)
        
    with open(caminho_modelo_xgb, 'rb') as f:
        pipeline_xgb = pickle.load(f)
        
    # Dicionário de resultados
    resultados = {}
    
    pipelines = {
        'Baseline (Regressão Logística)': pipeline_lr,
        'Champion (LightGBM)': pipeline_lgb,
        'Champion 2 (XGBoost)': pipeline_xgb
    }

    # 2. Avaliação dos Pipelines
    for nome, pipeline in pipelines.items():
        print(f"\nAvaliando Pipeline: {nome}...")
        # Predições utilizando dados brutos X_teste_raw - o Pipeline cuida de tudo!
        y_pred = pipeline.predict(X_teste_raw)
        y_prob = pipeline.predict_proba(X_teste_raw)[:, 1]
        
        # Calcular métricas
        auc = roc_auc_score(y_teste, y_prob)
        ap = average_precision_score(y_teste, y_prob)
        acc = accuracy_score(y_teste, y_pred)
        prec = precision_score(y_teste, y_pred)
        rec = recall_score(y_teste, y_pred)
        f1 = f1_score(y_teste, y_pred)
        
        resultados[nome] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'auc': auc,
            'ap': ap,
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'report': classification_report(y_teste, y_pred, output_dict=True)
        }
        
        print(f"ROC AUC: {auc:.4f} | PR AUC (AP): {ap:.4f} | F1-Score: {f1:.4f}")

    # 3. Gerar Gráficos
    sns.set_theme(style="whitegrid")
    
    # Gráfico 1: Curva ROC e PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Curva ROC
    for nome, res in resultados.items():
        fpr, tpr, _ = roc_curve(y_teste, res['y_prob'])
        axes[0].plot(fpr, tpr, label=f"{nome} (AUC = {res['auc']:.4f})", lw=2)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Taxa de Falso Positivo (FPR)', fontsize=12)
    axes[0].set_ylabel('Taxa de Verdadeiro Positivo (TPR)', fontsize=12)
    axes[0].set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right")
    
    # Curva Precision-Recall
    for nome, res in resultados.items():
        precision, recall, _ = precision_recall_curve(y_teste, res['y_prob'])
        axes[1].plot(recall, precision, label=f"{nome} (AP = {res['ap']:.4f})", lw=2)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Revocação (Recall)', fontsize=12)
    axes[1].set_ylabel('Precisão (Precision)', fontsize=12)
    axes[1].set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    caminho_curvas = os.path.join(config.PLOT_DIR, 'curvas_avaliacao_modelos.png')
    plt.savefig(caminho_curvas, dpi=150)
    plt.close()
    print(f"Gráfico de curvas salvo em: {caminho_curvas}")

    # Gráfico 2: Matrizes de Confusão (3 Modelos Lado a Lado)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (nome, res) in enumerate(resultados.items()):
        cm = confusion_matrix(y_teste, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        axes[i].set_title(f'Matriz de Confusão - {nome}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predito', fontsize=11)
        axes[i].set_ylabel('Real', fontsize=11)
        axes[i].set_xticklabels(['Não Inadimplente', 'Inadimplente'])
        axes[i].set_yticklabels(['Não Inadimplente', 'Inadimplente'], rotation=0)
    
    plt.tight_layout()
    caminho_matrizes = os.path.join(config.PLOT_DIR, 'matrizes_confusao.png')
    plt.savefig(caminho_matrizes, dpi=150)
    plt.close()
    print(f"Gráfico de matrizes de confusão salvo em: {caminho_matrizes}")

    # Gráfico 3: Importância das Variáveis do LightGBM extraídas do Pipeline
    prep = pipeline_lgb.named_steps['preprocessador']
    colunas_numericas = prep.transformers_[0][2]
    ohe = prep.named_transformers_['cat']
    colunas_categoricas_ohe = ohe.get_feature_names_out(config.CATEGORICAL_COLS).tolist()
    nomes_features_finais = list(colunas_numericas) + colunas_categoricas_ohe

    lgb_imp = pipeline_lgb.named_steps['classificador'].feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': nomes_features_finais,
        'Importance': lgb_imp
    }).sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Top 15 Variáveis Mais Importantes - LightGBM (Pipeline)', fontsize=14, fontweight='bold')
    plt.xlabel('Importância (Splits)', fontsize=12)
    plt.ylabel('Variável', fontsize=12)
    plt.tight_layout()
    caminho_importancia = os.path.join(config.PLOT_DIR, 'importancia_variaveis_lgb.png')
    plt.savefig(caminho_importancia, dpi=150)
    plt.close()
    print(f"Gráfico de Importância de Features salvo em: {caminho_importancia}")

    # 4. Escrever Relatório de Avaliação em Markdown
    caminho_relatorio = os.path.join(config.DATA_DIR, 'relatorio_avaliacao.md')
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Avaliação dos Pipelines Integrados - Previsão de Inadimplência\n\n")
        f.write("Este relatório compara a performance do pipeline Baseline (Regressão Logística) com os pipelines Champions (LightGBM e XGBoost).\n\n")
        
        f.write("## 1. Tabela Comparativa de Métricas (Pipelines de Ponta a Ponta)\n\n")
        f.write("| Modelo | ROC AUC | PR AUC (AP) | F1-Score | Acurácia | Precisão | Recall (Sensibilidade) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for nome, res in resultados.items():
            f.write(f"| {nome} | {res['auc']:.4f} | {res['ap']:.4f} | {res['f1']:.4f} | {res['acc']:.4f} | {res['prec']:.4f} | {res['rec']:.4f} |\n")
        f.write("\n")
        
        f.write("## 2. Relatórios de Classificação Detalhados\n\n")
        for nome, res in resultados.items():
            f.write(f"### {nome}\n")
            f.write("```\n")
            f.write(classification_report(y_teste, res['y_pred']))
            f.write("```\n\n")

        f.write("## 3. Análise de Importância das Features (LightGBM Pipeline)\n\n")
        f.write("As variáveis que mais contribuíram para a decisão do classificador do LightGBM no pipeline foram:\n\n")
        f.write("| Posição | Variável | Importância |\n")
        f.write("| --- | --- | --- |\n")
        for idx, row in feat_imp_df.reset_index().iterrows():
            f.write(f"| {idx+1} | {row['Feature']} | {row['Importance']:.0f} |\n")
        f.write("\n")
        
        f.write("## 4. Visualizações Geradas\n")
        f.write("- Curvas de Performance: `plots/curvas_avaliacao_modelos.png`\n")
        f.write("- Matrizes de Confusão: `plots/matrizes_confusao.png`\n")
        f.write("- Importância de Variáveis: `plots/importancia_variaveis_lgb.png`\n\n")
        f.write("--- \n*Relatório gerado automaticamente pelo pipeline de avaliação em português contendo Regressão Logística, LightGBM e XGBoost.*")

    print(f"Relatório de avaliação final salvo em: {caminho_relatorio}")
    print("=== Avaliação Concluída! ===")

if __name__ == '__main__':
    avaliar_pipelines()
