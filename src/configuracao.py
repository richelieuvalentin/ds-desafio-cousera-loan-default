import os

# Caminhos do Projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# Criar diretórios se não existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Definições do Dataset
TARGET_COL = 'Default'
DROP_COLS = ['LoanID']

CATEGORICAL_COLS = [
    'Education',
    'EmploymentType',
    'MaritalStatus',
    'HasMortgage',
    'HasDependents',
    'LoanPurpose',
    'HasCoSigner'
]

NUMERICAL_COLS = [
    'Age',
    'Income',
    'LoanAmount',
    'CreditScore',
    'MonthsEmployed',
    'NumCreditLines',
    'InterestRate',
    'LoanTerm',
    'DTIRatio'
]

# Configurações do Pipeline
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Usado apenas para monitoramento (early stopping) do LightGBM no treino

# Hiperparâmetros do LightGBM
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 15,
    'max_depth': 4,
    'min_child_samples': 100,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'random_state': RANDOM_STATE,
    'verbosity': -1,
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced'
}

# Hiperparâmetros do XGBoost
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 4,
    'min_child_weight': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 7.6
}

# Seleção de variáveis: Descartar colunas colineares após engenharia de features
DROP_COLS_AFTER_ENG = [
    'Income',
    'MonthlyIncome',
    'LoanToIncomeByTerm',
    'LogLoanAmount'
]
