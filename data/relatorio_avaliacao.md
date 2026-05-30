# Relatório de Avaliação dos Pipelines Integrados - Previsão de Inadimplência

Este relatório compara a performance do pipeline Baseline (Regressão Logística) com os pipelines Champions (LightGBM e XGBoost).

## 1. Tabela Comparativa de Métricas (Pipelines de Ponta a Ponta)

| Modelo | ROC AUC | PR AUC (AP) | F1-Score | Acurácia | Precisão | Recall (Sensibilidade) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (Regressão Logística) | 0.7614 | 0.3363 | 0.3434 | 0.6906 | 0.2279 | 0.6968 |
| Champion (LightGBM) | 0.7586 | 0.3331 | 0.3427 | 0.6939 | 0.2283 | 0.6871 |
| Champion 2 (XGBoost) | 0.7587 | 0.3328 | 0.3424 | 0.6936 | 0.2280 | 0.6871 |

## 2. Relatórios de Classificação Detalhados

### Baseline (Regressão Logística)
```
              precision    recall  f1-score   support

           0       0.95      0.69      0.80     45139
           1       0.23      0.70      0.34      5931

    accuracy                           0.69     51070
   macro avg       0.59      0.69      0.57     51070
weighted avg       0.86      0.69      0.74     51070
```

### Champion (LightGBM)
```
              precision    recall  f1-score   support

           0       0.94      0.69      0.80     45139
           1       0.23      0.69      0.34      5931

    accuracy                           0.69     51070
   macro avg       0.59      0.69      0.57     51070
weighted avg       0.86      0.69      0.75     51070
```

### Champion 2 (XGBoost)
```
              precision    recall  f1-score   support

           0       0.94      0.69      0.80     45139
           1       0.23      0.69      0.34      5931

    accuracy                           0.69     51070
   macro avg       0.59      0.69      0.57     51070
weighted avg       0.86      0.69      0.75     51070
```

## 3. Análise de Importância das Features (LightGBM Pipeline)

As variáveis que mais contribuíram para a decisão do classificador do LightGBM no pipeline foram:

| Posição | Variável | Importância |
| --- | --- | --- |
| 1 | MonthsEmployed | 521 |
| 2 | LoanToIncomeRatio | 462 |
| 3 | InterestRate | 355 |
| 4 | Age | 344 |
| 5 | Age_x_CreditScore | 288 |
| 6 | RiskScoreMultiplier | 275 |
| 7 | CreditLinesPerAge | 253 |
| 8 | TotalInterestCost | 247 |
| 9 | IncomePerCreditLine | 197 |
| 10 | LogIncome | 180 |
| 11 | TotalDebtServiceToIncome | 174 |
| 12 | InterestRate_x_DTIRatio | 162 |
| 13 | MonthlyDebt | 147 |
| 14 | CreditScore | 141 |
| 15 | EmploymentToAgeRatio | 139 |

## 4. Visualizações Geradas
- Curvas de Performance: `plots/curvas_avaliacao_modelos.png`
- Matrizes de Confusão: `plots/matrizes_confusao.png`
- Importância de Variáveis: `plots/importancia_variaveis_lgb.png`

--- 
*Relatório gerado automaticamente pelo pipeline de avaliação em português contendo Regressão Logística, LightGBM e XGBoost.*