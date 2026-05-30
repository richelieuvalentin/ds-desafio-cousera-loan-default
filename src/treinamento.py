import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier

import src.configuracao as config
import src.preparacao_dados as preparacao_dados
import src.engenharia_variaveis as engenharia_variaveis

def pipeline_treinamento():
    print("=== Iniciando Pipeline de Treinamento e Reestruturação ===")
    
    # 1. Carregar dados brutos
    print("Carregando dados de treino brutos...")
    df_bruto = preparacao_dados.carregar_dados_brutos(is_treino=True)
    
    # 2. Dividir o dataset em Treino, Validação e Teste Local
    print("Dividindo o dataset (Treino, Validação, Teste)...")
    X_treino, X_val, X_teste, y_treino, y_val, y_teste = preparacao_dados.dividir_treino_val_teste(df_bruto)
    
    # Salvar dados de teste brutos para avaliação posterior
    dados_teste = {
        'X_teste_raw': X_teste,
        'y_teste': y_teste
    }
    caminho_teste = os.path.join(config.MODEL_DIR, 'dados_teste.pkl')
    with open(caminho_teste, 'wb') as f:
        pickle.dump(dados_teste, f)
    print(f"Conjunto de teste bruto salvo em: {caminho_teste}")
    
    # 3. Mapear as colunas numéricas pós engenharia de features
    colunas_numericas_finais = engenharia_variaveis.obter_colunas_pos_engenharia(
        config.NUMERICAL_COLS, 
        config.DROP_COLS_AFTER_ENG
    )
    
    print(f"Total de colunas numéricas finais no modelo: {len(colunas_numericas_finais)}")
    print(f"Total de colunas categóricas no modelo: {len(config.CATEGORICAL_COLS)}")

    # 4. Criar o pré-processador (StandardScaler + OneHotEncoder)
    preprocessador = engenharia_variaveis.obter_preprocessador(
        colunas_numericas_finais, 
        config.CATEGORICAL_COLS
    )

    # 5. CONSTRUIR PIPELINES INTEGRADOS DE PONTA A PONTA
    
    # 5.1. Pipeline Baseline (Regressão Logística)
    print("\n--- Construindo e Treinando Pipeline Baseline: Regressão Logística ---")
    pipeline_lr = Pipeline(steps=[
        ('engenheiro_features', engenharia_variaveis.EngenheiroFeatures(colunas_descarte=config.DROP_COLS_AFTER_ENG)),
        ('preprocessador', preprocessador),
        ('classificador', LogisticRegression(
            max_iter=1000, 
            random_state=config.RANDOM_STATE, 
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Treinar a Regressão Logística passando dados brutos
    pipeline_lr.fit(X_treino, y_treino)
    
    caminho_modelo_lr = os.path.join(config.MODEL_DIR, 'modelo_baseline.pkl')
    with open(caminho_modelo_lr, 'wb') as f:
        pickle.dump(pipeline_lr, f)
    print(f"Pipeline Regressão Logística salvo em: {caminho_modelo_lr}")

    # 5.2. Pipeline Champion (LightGBM)
    print("\n--- Construindo e Treinando Pipeline Champion: LightGBM ---")
    pipeline_lgb = Pipeline(steps=[
        ('engenheiro_features', engenharia_variaveis.EngenheiroFeatures(colunas_descarte=config.DROP_COLS_AFTER_ENG)),
        ('preprocessador', preprocessador),
        ('classificador', LGBMClassifier(**config.LGBM_PARAMS))
    ])

    # Para usar o early stopping do LightGBM com o Pipeline:
    # 1. Ajustamos a parte de preparação (Feature Engineering + Preprocessing) do pipeline nos dados de treino
    pipeline_prep = Pipeline(steps=pipeline_lgb.steps[:-1])
    X_treino_proc = pipeline_prep.fit_transform(X_treino, y_treino)
    
    # 2. Transformamos os dados de validação usando a parte de preparação
    X_val_proc = pipeline_prep.transform(X_val)
    
    # 3. Treinamos o pipeline completo passando o eval_set processado para o classificador LightGBM
    pipeline_lgb.fit(
        X_treino, y_treino,
        classificador__eval_set=[(X_val_proc, y_val)],
        classificador__callbacks=[
            early_stopping(stopping_rounds=50, verbose=True),
            log_evaluation(period=50)
        ]
    )

    caminho_modelo_lgb = os.path.join(config.MODEL_DIR, 'modelo_champion.pkl')
    with open(caminho_modelo_lgb, 'wb') as f:
        pickle.dump(pipeline_lgb, f)
    print(f"Pipeline LightGBM salvo em: {caminho_modelo_lgb}")

    # 5.3. Pipeline Champion 2 (XGBoost)
    print("\n--- Construindo e Treinando Pipeline Champion 2: XGBoost ---")
    pipeline_xgb = Pipeline(steps=[
        ('engenheiro_features', engenharia_variaveis.EngenheiroFeatures(colunas_descarte=config.DROP_COLS_AFTER_ENG)),
        ('preprocessador', preprocessador),
        ('classificador', XGBClassifier(early_stopping_rounds=50, **config.XGB_PARAMS))
    ])

    # Treinar o pipeline XGBoost
    pipeline_xgb.fit(
        X_treino, y_treino,
        classificador__eval_set=[(X_val_proc, y_val)],
        classificador__verbose=100
    )

    caminho_modelo_xgb = os.path.join(config.MODEL_DIR, 'modelo_xgboost.pkl')
    with open(caminho_modelo_xgb, 'wb') as f:
        pickle.dump(pipeline_xgb, f)
    print(f"Pipeline XGBoost salvo em: {caminho_modelo_xgb}")
    
    print("\n=== Treinamento Concluído com Sucesso! Modelos integrados salvos em .pkl ===")

if __name__ == '__main__':
    pipeline_treinamento()
