# Relatório de Avaliação de Modelos - Previsão de Inadimplência

Este relatório compara a performance do modelo Baseline (Regressão Logística) com o modelo Champion (LightGBM).

## 1. Tabela Comparativa de Métricas

| Modelo | ROC AUC | PR AUC (AP) | F1-Score | Acurácia | Precisão | Recall (Sensibilidade) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (Regressão Logística) | 0.7615 | 0.3364 | 0.3434 | 0.6907 | 0.2279 | 0.6965 |
| Champion (LightGBM) | 0.7588 | 0.3331 | 0.3418 | 0.6939 | 0.2278 | 0.6844 |

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

           0       0.94      0.70      0.80     45139
           1       0.23      0.68      0.34      5931

    accuracy                           0.69     51070
   macro avg       0.59      0.69      0.57     51070
weighted avg       0.86      0.69      0.75     51070
```

## 3. Análise de Importância das Features (LightGBM)

As variáveis que mais contribuíram para a decisão do LightGBM de prever inadimplência foram:

| Posição | Variável | Importância |
| --- | --- | --- |
| 1 | MonthsEmployed | 554 |
| 2 | LoanToIncomeRatio | 505 |
| 3 | InterestRate | 466 |
| 4 | Age | 427 |
| 5 | RiskScoreMultiplier | 291 |
| 6 | Age_x_CreditScore | 285 |
| 7 | IncomePerCreditLine | 273 |
| 8 | LoanAmount | 236 |
| 9 | InterestRate_x_DTIRatio | 218 |
| 10 | LogIncome | 192 |
| 11 | EmploymentToAgeRatio | 150 |
| 12 | MonthlyDebt | 147 |
| 13 | CreditScore | 130 |
| 14 | DisposableIncome | 111 |
| 15 | HasDependents_Yes | 110 |

## 4. Visualizações Geradas
- Curvas ROC e Precision-Recall: `plots/model_evaluation_curves.png`
- Matrizes de Confusão: `plots/confusion_matrices.png`
- Importância de Variáveis: `plots/lgb_feature_importance.png`

--- 
*Relatório gerado automaticamente pelo pipeline.*