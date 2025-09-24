# Detector de Spam com Rede Neural

Este é um projeto acadêmico simples que demonstra a construção e o uso de uma rede neural para classificar e-mails como "spam" ou "não spam". Foi desenvolvido como um estudo prático sobre a aplicação de modelos de Machine Learning para a faculdade que estou cursando, Análise e Desenvolvimento de Sistemas na Estácio.

## Visão Geral

O projeto é dividido em duas partes:

1.  **`main.py`**: Script responsável por treinar o modelo. Ele carrega o dataset Spambase, treina um classificador de rede neural (MLPClassifier do Scikit-learn) e salva o modelo treinado no arquivo `spam_model.pkl`.

2.  **`predict_email.py`**: Um script interativo que carrega o modelo treinado e permite que o usuário insira o texto de um e-mail. O script então extrai as características do texto e utiliza o modelo para prever se o e-mail é spam.

Os inputs utilizados são extremamente simples. Usei somente as sugestões disponibilizadas junto do dataset que baixei online.

## Tecnologias Utilizadas

- **Python**
- **Scikit-learn**: Para a implementação da rede neural e divisão dos dados.
- **Pandas**: Para manipulação do dataset.
- **NumPy**: Para operações numéricas.

## Como Executar

1.  **Clone o repositório:**

    ```bash
    git clone <URL-do-seu-repositorio>
    cd <nome-do-repositorio>
    ```

2.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Baixe o Dataset:**
    Faça o download do arquivo `spambase.data` do UCI Machine Learning Repository e coloque-o na pasta raiz do projeto.

4.  **Treine o modelo:**

    ```bash
    python main.py
    ```

5.  **Execute o detector de spam:**
    ```bash
    python predict_email.py
    ```
