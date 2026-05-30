# Análise Exploratória de Dados (EDA) - Resumo

## 1. Dimensões dos Datasets
- **Treino (`train.csv`):** 255347 linhas, 18 colunas
- **Teste (`test.csv`):** 109435 linhas, 17 colunas

## 2. Distribuição da Variável Alvo (`Default`)
| Default | Quantidade | Porcentagem |
| --- | --- | --- |
| 0 | 225694 | 88.39% |
| 1 | 29653 | 11.61% |

## 3. Valores Ausentes (Null/NaN)
### Dataset de Treino:
| Coluna | Nulos | Porcentagem |
| --- | --- | --- |
Nenhum valor ausente encontrado no conjunto de treino.

### Dataset de Teste:
| Coluna | Nulos | Porcentagem |
| --- | --- | --- |
Nenhum valor ausente encontrado no conjunto de teste.

## 4. Tipos de Dados e Exemplos
| Coluna | Tipo | Exemplo Treino | Exemplo Teste |
| --- | --- | --- | --- |
| LoanID | object | I38PQUQS96 | 7RYZGMKJIR |
| Age | int64 | 56 | 32 |
| Income | int64 | 85994 | 131645 |
| LoanAmount | int64 | 50587 | 43797 |
| CreditScore | int64 | 520 | 802 |
| MonthsEmployed | int64 | 80 | 23 |
| NumCreditLines | int64 | 4 | 2 |
| InterestRate | float64 | 15.23 | 6.1 |
| LoanTerm | int64 | 36 | 24 |
| DTIRatio | float64 | 0.44 | 0.13 |
| Education | object | Bachelor's | High School |
| EmploymentType | object | Full-time | Full-time |
| MaritalStatus | object | Divorced | Divorced |
| HasMortgage | object | Yes | Yes |
| HasDependents | object | Yes | No |
| LoanPurpose | object | Other | Other |
| HasCoSigner | object | Yes | No |
| Default | int64 | 0 | N/A |

## 5. Análise de Colunas Categóricas
### Coluna: `Education`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Bachelor's | 64366 | 25.21% |
| High School | 63903 | 25.03% |
| Master's | 63541 | 24.88% |
| PhD | 63537 | 24.88% |

### Coluna: `EmploymentType`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Part-time | 64161 | 25.13% |
| Unemployed | 63824 | 25.00% |
| Self-employed | 63706 | 24.95% |
| Full-time | 63656 | 24.93% |

### Coluna: `MaritalStatus`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Married | 85302 | 33.41% |
| Divorced | 85033 | 33.30% |
| Single | 85012 | 33.29% |

### Coluna: `HasMortgage`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Yes | 127677 | 50.00% |
| No | 127670 | 50.00% |

### Coluna: `HasDependents`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Yes | 127742 | 50.03% |
| No | 127605 | 49.97% |

### Coluna: `LoanPurpose`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Business | 51298 | 20.09% |
| Home | 51286 | 20.08% |
| Education | 51005 | 19.97% |
| Other | 50914 | 19.94% |
| Auto | 50844 | 19.91% |

### Coluna: `HasCoSigner`
| Categoria | Contagem | Porcentagem |
| --- | --- | --- |
| Yes | 127701 | 50.01% |
| No | 127646 | 49.99% |

## 6. Estatísticas Descritivas (Numéricas)
| Coluna | Média | Desvio Padrão | Mínimo | 25% | 50% | 75% | Máximo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 43.50 | 14.99 | 18.00 | 31.00 | 43.00 | 56.00 | 69.00 |
| Income | 82499.30 | 38963.01 | 15000.00 | 48825.50 | 82466.00 | 116219.00 | 149999.00 |
| LoanAmount | 127578.87 | 70840.71 | 5000.00 | 66156.00 | 127556.00 | 188985.00 | 249999.00 |
| CreditScore | 574.26 | 158.90 | 300.00 | 437.00 | 574.00 | 712.00 | 849.00 |
| MonthsEmployed | 59.54 | 34.64 | 0.00 | 30.00 | 60.00 | 90.00 | 119.00 |
| NumCreditLines | 2.50 | 1.12 | 1.00 | 2.00 | 2.00 | 3.00 | 4.00 |
| InterestRate | 13.49 | 6.64 | 2.00 | 7.77 | 13.46 | 19.25 | 25.00 |
| LoanTerm | 36.03 | 16.97 | 12.00 | 24.00 | 36.00 | 48.00 | 60.00 |
| DTIRatio | 0.50 | 0.23 | 0.10 | 0.30 | 0.50 | 0.70 | 0.90 |
| Default | 0.12 | 0.32 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

