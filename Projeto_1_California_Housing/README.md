# Projeto 1: Precificação de Imóveis — California Housing

> Pipeline de regressão supervisionada para prever o valor mediano de imóveis na Califórnia, do carregamento dos dados brutos à serialização para produção.

---

## Objetivo

Prever a coluna `median_house_value` a partir de variáveis demográficas e geográficas, atingindo pelo menos uma das metas abaixo no conjunto de teste:

| Métrica | Meta |
|---------|------|
| RMSE    | < **$70.000** |
| R²      | > **0,70** |

---

## Dataset

- **Fonte:** [California Housing — Aurélien Géron / Hands-On ML2](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
- **Tamanho:** ~20.640 linhas × 10 colunas
- **Problemas tratados:** valores ausentes em `total_bedrooms`; variável categórica `ocean_proximity`

---

## Pipeline

Todo o fluxo é encapsulado em um único `Pipeline` do scikit-learn para evitar data leakage.

```
DataFrame bruto
      │
      ▼
ColumnTransformer
 ├── Colunas numéricas   → SimpleImputer(mediana) → StandardScaler
 └── Colunas categóricas → SimpleImputer(moda)    → OneHotEncoder
      │
      ▼
RandomForestRegressor
      │
      ▼
median_house_value predito
```

---

## Etapas do Notebook

| Passo | Descrição |
|-------|-----------|
| 1 | Carregamento dos dados e divisão treino/teste 80/20 |
| 2 | Construção do `ColumnTransformer` com pipelines numérico e categórico |
| 3 | Comparação de quatro regressores: Regressão Linear, Árvore de Decisão, Random Forest, Gradient Boosting |
| 4 | Otimização com `RandomizedSearchCV` (30 iterações, 5-fold CV) |
| 5 | Avaliação no teste: RMSE, R², MAE |
| 6 | Serialização do pipeline final em `pipeline_california.joblib` |
| 7 | Simulação de produção: carregamento do modelo e predição a partir de payload bruto (com `NaN`) |

---

## Simulação de Produção

```python
payload = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": float("nan"),  # valor ausente tratado pelo pipeline
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}

preco = predizer_imovel(payload)  # → valor predito em USD
```
