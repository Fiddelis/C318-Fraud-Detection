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
│   └── features/      # Funções de engenharia e seleção de features
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
