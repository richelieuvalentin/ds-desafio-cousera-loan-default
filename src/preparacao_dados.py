import os
import pandas as pd
from sklearn.model_selection import train_test_split
import src.configuracao as config

def carregar_dados_brutos(is_treino=True):
    """
    Carrega o dataset bruto de treino ou teste.
    """
    nome_arquivo = 'train.csv' if is_treino else 'test.csv'
    caminho = os.path.join(config.DATA_DIR, nome_arquivo)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho)
    return df

def dividir_treino_val_teste(df):
    """
    Divide o conjunto de dados bruto (sem transformações) em:
    - Treino (70%)
    - Validação (10% - para monitoramento e early stopping do LightGBM)
    - Teste Local (20% - para avaliação final das métricas)
    
    A amostragem é estratificada com base no target 'Default'.
    """
    # Separar X e y. Removemos apenas a coluna de identificação 'LoanID' aqui.
    X = df.drop(columns=[config.TARGET_COL] + config.DROP_COLS, errors='ignore')
    y = df[config.TARGET_COL]

    # Split local test (20%)
    X_treino_val, X_teste, y_treino_val, y_teste = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )

    # Split val (12.5% do treino/val restante = 10% do total)
    proporcao_val = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_treino_val, y_treino_val, 
        test_size=proporcao_val, 
        random_state=config.RANDOM_STATE, 
        stratify=y_treino_val
    )

    return X_treino, X_val, X_teste, y_treino, y_val, y_teste
