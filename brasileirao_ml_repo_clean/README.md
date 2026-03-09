
# Brasileirão Match Result Prediction (Machine Learning)

Projeto acadêmico para a disciplina **Fundamentos de Machine Learning**.

Objetivo: prever o resultado de partidas do Campeonato Brasileiro Série A
(vitória do mandante, empate ou vitória do visitante).

## Dataset

Dataset público do Campeonato Brasileiro (2003–2024).

Fonte:
https://raw.githubusercontent.com/afonsoleone25/brasileirao-ml-project/main/data/brasileirao_matches.csv

Arquivo utilizado:
campeonato-brasileiro-full.csv

## Estrutura

ml-brasileirao-ml-project-clean
│
├── notebook
│   └── brasileirao_prediction.ipynb
│
├── src
│   └── train_model.py
│
├── requirements.txt
└── README.md

## Como executar

Instalar dependências:

pip install -r requirements.txt

Executar treinamento:

python src/train_model.py

Ou abrir o notebook no Google Colab.
