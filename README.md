# C318 - Fraud Detection

Projeto da disciplina **C318** voltado à detecção de fraudes em e-commerce com técnicas de Machine Learning.

## Objetivo

Construir um pipeline de classificação capaz de identificar transações fraudulentas, com ênfase em qualidade dos dados, seleção de atributos e avaliação adequada do modelo em cenários desbalanceados.

## Dataset

Fonte: [E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset) (Kaggle).

Dataset sintético com ~300 mil transações de ~6 mil usuários, desbalanceamento natural (~2% de fraude) e sinais comportamentais correlacionados (país, canal, horário, distância de envio, etc.).

Arquivo esperado: `transactions.csv` em `data/raw/`.

```bash
# Com a API do Kaggle configurada (~/.kaggle/kaggle.json):
kaggle datasets download -d umuttuygurr/e-commerce-fraud-detection-dataset -p data/raw --unzip
```

Os dados não são versionados no Git (ver `.gitignore`).

## Estrutura

```text
C318-Fraud-Detection/
├── data/
│   ├── raw/           # Dados originais (não versionados)
│   ├── processed/     # Dados limpos e prontos para modelagem
│   └── external/      # Fontes auxiliares, se houver
├── notebooks/         # Análises e experimentos (adicionados por etapa)
├── src/
│   ├── data/          # Funções de carregamento e limpeza
│   ├── features/      # Funções de engenharia e seleção de features
│   └── models/        # Treino, avaliação e persistência dos modelos
├── models/            # Artefatos de modelo (não versionados)
├── reports/
│   └── figures/       # Gráficos e figuras geradas
├── requirements.txt
└── README.md
```

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Treino do modelo

```bash
python -m src.models.train
```

O script gera `data/processed/transactions_clean.csv` se ainda não existir, treina e compara **Random Forest** e **HistGradientBoosting** e salva:

- `models/fraud_model.joblib` - modelo escolhido + limiar de decisão + lista de features;
- `reports/model_comparison.csv` - métricas de validação e teste dos dois modelos;
- `reports/model_metrics.json` - resumo do modelo escolhido;
- `reports/model_tuning.csv` - histórico da busca de hiperparâmetros dos dois modelos.

Decisões de avaliação:

- **Divisão temporal 70/15/15** por `transaction_time`: o modelo nunca treina com transações posteriores às que avalia.
- **Métrica principal: PR-AUC** (average precision). Com ~2% de fraude, acurácia não informa nada (prever "nunca fraude" acerta ~98%).
- **Hiperparâmetros e limiar escolhidos na validação**; o teste é usado uma única vez.
- **Teste de robustez sem `avg_amount_user`, `total_transactions_user` e `amount_to_avg_ratio`**: esses agregados são calculados pelo dataset sobre o período inteiro e carregam um pouco de informação futura.

Validação cruzada temporal (5 folds, `TimeSeriesSplit`) das configurações escolhidas, sem usar o teste:

```bash
python -m src.models.cross_validation
```

Resultados em `reports/cross_validation_summary.csv` e `reports/cross_validation_folds.csv`. O resumo das decisões de modelagem está em [`reports/decisoes_treinamento.md`](reports/decisoes_treinamento.md).

Para usar o modelo salvo:

```python
from src.data.cleaning import clean_transactions
from src.models.train import load_model, predict_fraud

artifact = load_model()
preds = predict_fraud(artifact, clean_transactions(novas_transacoes))
```
