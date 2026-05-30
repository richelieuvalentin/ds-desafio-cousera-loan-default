import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import src.configuracao as config

class EngenheiroFeatures(BaseEstimator, TransformerMixin):
    """
    Transformador customizado do Scikit-Learn que realiza a engenharia de features
    e a seleção de variáveis (descarte de colineares) em um único passo no Pipeline.
    """
    def __init__(self, colunas_descarte=None):
        self.colunas_descarte = colunas_descarte if colunas_descarte is not None else []
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df_feat = X.copy()

        # 1. Relação Valor do Empréstimo / Renda (Loan-to-Income Ratio)
        df_feat['LoanToIncomeRatio'] = df_feat['LoanAmount'] / df_feat['Income']

        # 2. Renda Mensal Estimada
        df_feat['MonthlyIncome'] = df_feat['Income'] / 12.0

        # 3. Dívida Mensal Estimada (baseada no DTI Ratio)
        df_feat['MonthlyDebt'] = df_feat['DTIRatio'] * df_feat['MonthlyIncome']

        # 4. Cálculo da Parcela Mensal (Sistema Price)
        r = (df_feat['InterestRate'] / 100.0) / 12.0
        n = df_feat['LoanTerm']
        
        installment = np.where(
            r > 0,
            df_feat['LoanAmount'] * (r * (1 + r)**n) / ((1 + r)**n - 1),
            df_feat['LoanAmount'] / n
        )
        df_feat['EstimatedMonthlyInstallment'] = installment

        # 5. Relação Parcela / Renda (Payment-to-Income Ratio)
        df_feat['PaymentToIncomeRatio'] = df_feat['EstimatedMonthlyInstallment'] / df_feat['MonthlyIncome']

        # 6. Renda Disponível Estimada
        df_feat['DisposableIncome'] = df_feat['MonthlyIncome'] - df_feat['MonthlyDebt'] - df_feat['EstimatedMonthlyInstallment']

        # 7. Multiplicador de Emprego (Months Employed / Age em meses)
        age_in_months = df_feat['Age'] * 12.0
        df_feat['EmploymentToAgeRatio'] = df_feat['MonthsEmployed'] / age_in_months

        # 8. Multiplicador de Risco (Taxa de Juros / Score de Crédito)
        df_feat['RiskScoreMultiplier'] = df_feat['InterestRate'] / (df_feat['CreditScore'] + 1e-5)

        # 9. Relação Empréstimo-Renda Ajustada ao Prazo
        df_feat['LoanToIncomeByTerm'] = df_feat['LoanAmount'] / (df_feat['Income'] * (df_feat['LoanTerm'] / 12.0) + 1e-5)

        # 10. Empréstimo e Renda por Linha de Crédito
        df_feat['LoanAmountPerCreditLine'] = df_feat['LoanAmount'] / (df_feat['NumCreditLines'] + 1e-5)
        df_feat['IncomePerCreditLine'] = df_feat['Income'] / (df_feat['NumCreditLines'] + 1e-5)

        # 11. Sinalizadores de Risco de Negócio (Booleanos em formato numérico 0/1)
        df_feat['IsUnemployedNoCoSigner'] = np.where((df_feat['EmploymentType'] == 'Unemployed') & (df_feat['HasCoSigner'] == 'No'), 1, 0)
        df_feat['IsYoungLowIncome'] = np.where((df_feat['Age'] < 30) & (df_feat['Income'] < 40000), 1, 0)
        df_feat['IsHighDebtRatio'] = np.where(df_feat['DTIRatio'] > 0.7, 1, 0)

        # 12. Transformações Logarítmicas
        df_feat['LogIncome'] = np.log1p(df_feat['Income'])
        df_feat['LogLoanAmount'] = np.log1p(df_feat['LoanAmount'])

        # 13. Interações Cruzadas Importantes
        df_feat['InterestRate_x_DTIRatio'] = df_feat['InterestRate'] * df_feat['DTIRatio']
        df_feat['Age_x_CreditScore'] = df_feat['Age'] * df_feat['CreditScore']

        # --- Features Criativas Adicionais ---
        # 14. Comprometimento Total de Renda (Total Debt Service to Income Ratio)
        df_feat['TotalDebtServiceToIncome'] = (df_feat['MonthlyDebt'] + df_feat['EstimatedMonthlyInstallment']) / (df_feat['MonthlyIncome'] + 1e-5)

        # 15. Cobertura de Parcela por Renda Disponível
        df_feat['DisposableIncomeToInstallment'] = df_feat['DisposableIncome'] / (df_feat['EstimatedMonthlyInstallment'] + 1e-5)

        # 16. Custo Total de Juros do Contrato
        df_feat['TotalInterestCost'] = (df_feat['EstimatedMonthlyInstallment'] * df_feat['LoanTerm']) - df_feat['LoanAmount']

        # 17. Cruzamentos Comportamentais de Risco
        df_feat['IsYoungSingleWithDependents'] = np.where((df_feat['Age'] < 26) & (df_feat['MaritalStatus'] == 'Single') & (df_feat['HasDependents'] == 'Yes'), 1, 0)
        df_feat['IsBusinessNoCoSigner'] = np.where((df_feat['LoanPurpose'] == 'Business') & (df_feat['HasCoSigner'] == 'No'), 1, 0)

        # 18. Alavancagem e Renda por Idade
        df_feat['CreditLinesPerAge'] = df_feat['NumCreditLines'] / df_feat['Age']
        df_feat['IncomePerAge'] = df_feat['Income'] / df_feat['Age']

        # 19. Discretização de Risco de Score (Binning)
        df_feat['Score_Pessimo'] = np.where(df_feat['CreditScore'] < 500, 1, 0)
        df_feat['Score_Ruim'] = np.where((df_feat['CreditScore'] >= 500) & (df_feat['CreditScore'] < 580), 1, 0)
        df_feat['Score_Regular'] = np.where((df_feat['CreditScore'] >= 580) & (df_feat['CreditScore'] < 670), 1, 0)
        df_feat['Score_Bom'] = np.where(df_feat['CreditScore'] >= 670, 1, 0)

        # 20. Discretização de Risco de Juros (Binning)
        df_feat['Juros_Critico'] = np.where(df_feat['InterestRate'] > 18.0, 1, 0)
        df_feat['Juros_Alto'] = np.where((df_feat['InterestRate'] >= 12.0) & (df_feat['InterestRate'] <= 18.0), 1, 0)

        # Descartar colunas colineares selecionadas
        if len(self.colunas_descarte) > 0:
            df_feat = df_feat.drop(columns=self.colunas_descarte, errors='ignore')

        return df_feat

def obter_preprocessador(colunas_numericas, colunas_categoricas):
    """
    Retorna o ColumnTransformer que realiza StandardScaler nas numéricas 
    e OneHotEncoder nas categóricas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), colunas_numericas),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), colunas_categoricas)
        ]
    )
    return preprocessor

def obter_colunas_pos_engenharia(colunas_numericas_originais, colunas_descarte):
    """
    Calcula dinamicamente a lista de colunas numéricas que estarão disponíveis
    após a aplicação do EngenheiroFeatures e remoção de colineares.
    """
    novas_colunas = [
        'LoanToIncomeRatio', 'MonthlyIncome', 'MonthlyDebt', 
        'EstimatedMonthlyInstallment', 'PaymentToIncomeRatio', 
        'DisposableIncome', 'EmploymentToAgeRatio',
        'RiskScoreMultiplier', 'LoanToIncomeByTerm', 'LoanAmountPerCreditLine',
        'IncomePerCreditLine', 'IsUnemployedNoCoSigner', 'IsYoungLowIncome',
        'IsHighDebtRatio', 'LogIncome', 'LogLoanAmount',
        'InterestRate_x_DTIRatio', 'Age_x_CreditScore',
        'TotalDebtServiceToIncome', 'DisposableIncomeToInstallment',
        'TotalInterestCost', 'IsYoungSingleWithDependents',
        'IsBusinessNoCoSigner', 'CreditLinesPerAge', 'IncomePerAge',
        'Score_Pessimo', 'Score_Ruim', 'Score_Regular', 'Score_Bom',
        'Juros_Critico', 'Juros_Alto'
    ]
    
    # Unir originais com as novas criadas
    todas_numericas = list(colunas_numericas_originais) + novas_colunas
    
    # Filtrar removendo as colunas descartadas
    numericas_finais = [c for c in todas_numericas if c not in colunas_descarte]
    
    return numericas_finais
