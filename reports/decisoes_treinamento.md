# Decisões de treinamento do modelo

Resumo das escolhas feitas na etapa de modelagem e da justificativa de cada uma. Código em `src/models/`; números em `reports/model_comparison.csv` e `reports/cross_validation_summary.csv`.

## 1. Divisão dos dados: temporal, não aleatória

As 299.695 transações (jan–out/2024) foram ordenadas por `transaction_time` e divididas em blocos consecutivos:

| Conjunto | Período | Transações | Fraudes |
|---|---|---|---|
| Treino (70%) | 01/01 – 01/08 | 209.786 | 4.658 (2,22%) |
| Validação (15%) | 01/08 – 16/09 | 44.954 | 977 (2,17%) |
| Teste (15%) | 16/09 – 31/10 | 44.955 | 977 (2,17%) |

**Por quê:** em produção o modelo sempre prevê transações futuras a partir do passado. Uma divisão aleatória misturaria meses e permitiria aprender com dados "do futuro", inflando as métricas. A taxa de fraude é estável entre os meses (≈2,1–2,3%), então os três blocos são comparáveis.

## 2. Métrica principal: PR-AUC

Com ~2% de fraude, um classificador que nunca acusa fraude tem 97,8% de acurácia e não detecta nada. Por isso a acurácia foi descartada. A métrica de escolha é a **PR-AUC (average precision)**, que mede o equilíbrio entre precisão e recall na classe rara. ROC-AUC, precisão, recall e F1 são reportados como complemento.

## 3. Modelos: Random Forest e HistGradientBoosting

Os dois são ensembles de árvores de decisão, adequados a este problema porque:

- capturam interações entre atributos (ex.: conta nova + valor alto + falha de AVS) sem que seja preciso criá-las manualmente;
- não exigem padronização e não são distorcidos pelos valores extremos de `amount` e `shipping_distance_km`, mantidos de propósito na limpeza;
- treinam rapidamente em ~300 mil linhas.

Eles representam as duas famílias clássicas de ensemble: **bagging** (Random Forest: árvores independentes que votam) e **boosting** (HistGradientBoosting: árvores sequenciais que corrigem os erros das anteriores).

As 18 features recomendadas no notebook 03 foram usadas; as categóricas (`channel`, `merchant_category`, `country`, `bin_country`) foram codificadas com `OrdinalEncoder`, suficiente para modelos de árvore.

## 4. Ajuste de hiperparâmetros

Busca aleatória com 20 combinações por modelo, cada uma treinada no conjunto de treino e avaliada pela PR-AUC na validação. A busca incluiu `class_weight` (com e sem balanceamento), deixando os dados decidirem como tratar o desbalanceamento em vez de aplicar reamostragem (SMOTE).

| Modelo | Melhor configuração |
|---|---|
| Random Forest | 300 árvores, `max_depth=12`, `min_samples_leaf=5`, `max_features="sqrt"`, sem balanceamento |
| HistGradientBoosting | `learning_rate=0.05`, `max_leaf_nodes=15`, `min_samples_leaf=100`, `class_weight="balanced"`, early stopping |

## 5. Limiar de decisão escolhido na validação

O limiar padrão de 0,5 raramente é adequado a dados desbalanceados. Para cada modelo foi escolhido, **na validação**, o limiar que maximiza o F1: 0,495 para o Random Forest e 0,940 para o HistGradientBoosting (alto porque o balanceamento de classes desloca as probabilidades para cima). O conjunto de teste foi usado **uma única vez**, para a avaliação final.

## 6. Validação cruzada temporal

Para verificar se o resultado de uma única validação era estável, as configurações escolhidas foram avaliadas com **validação cruzada temporal** (`TimeSeriesSplit`, 5 folds) sobre treino + validação. Cada fold treina com todo o passado disponível e testa nas ~6 semanas seguintes; o teste final não participa. Um K-fold comum não foi usado porque embaralharia o tempo.

| Modelo | PR-AUC (média ± desvio) | Precisão | Recall |
|---|---|---|---|
| Random Forest | **0,870 ± 0,009** | 96,5% | 74,5% |
| HistGradientBoosting | 0,865 ± 0,014 | 91,1% | 75,4% |

O Random Forest teve PR-AUC maior nos **5 de 5 folds**, com menor variação. A vantagem diminui conforme o treino cresce (de +0,015 no 1º fold para +0,0001 no 5º): com dados suficientes os modelos empatam, mas o Random Forest é mais estável e consistentemente mais preciso.

*Ressalva:* os folds reutilizam os dados em que hiperparâmetros e limiares foram escolhidos, então os valores absolutos são levemente otimistas. A comparação entre os modelos continua válida, pois ambos tiveram a mesma vantagem.

## 7. Verificação de vazamento

`avg_amount_user`, `total_transactions_user` (e, por consequência, `amount_to_avg_ratio`) são constantes por usuário e calculados pelo dataset sobre os 10 meses inteiros. Ou seja, carregam um pouco de informação futura. Retreinando sem esses atributos, a PR-AUC de teste cai apenas de ~0,86 para ~0,84 nos dois modelos: o desempenho **não depende** desse vazamento.

## 8. Resultado no teste e modelo escolhido

| Modelo | PR-AUC | ROC-AUC | Precisão | Recall | F1 | Fraudes detectadas | Falsos alarmes |
|---|---|---|---|---|---|---|---|
| Random Forest | 0,867 | 0,979 | 95,3% | 74,7% | 0,838 | 730 de 977 | 36 |
| HistGradientBoosting | 0,861 | 0,978 | 89,1% | 77,0% | 0,826 | 752 de 977 | 92 |

Pela regra inicial (maior PR-AUC na validação única), o HistGradientBoosting foi escolhido por 0,0008, uma diferença dentro do ruído. A validação cruzada, mais confiável por usar cinco períodos e não depender do teste, favorece o **Random Forest**: melhor em todos os folds, mais estável e com cerca de metade dos falsos alarmes, ao custo de detectar um pouco menos de fraudes (≈1 p.p. de recall na validação cruzada, ≈2 p.p. no teste).

## 9. Limitações e próximos passos

- Calcular os perfis de usuário apenas com transações anteriores (média móvel), eliminando o vazamento descrito no item 7.
- Definir o limiar a partir de custos reais (custo de uma fraude não detectada × custo de bloquear um cliente legítimo) em vez do F1.
- Retreinar com treino + validação antes de colocar em produção, aproveitando os meses mais recentes.
