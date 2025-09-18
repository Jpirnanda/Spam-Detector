import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

# Nomes das colunas conforme spambase.names
COLUMN_NAMES = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
    'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
    'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
    'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
    'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
    'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
    'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'spam'
]

def main():
    print("Spam Detector started.")
    # Carregar o dataset
    df = pd.read_csv('spambase.data', header=None, names=COLUMN_NAMES)
    print('Dados carregados!')
    print(df.head())

    # Separar variáveis de entrada (X) e saída (y)
    X = df.drop('spam', axis=1)
    y = df['spam']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'Treino: {X_train.shape}, Teste: {X_test.shape}')

    # Construir e treinar a rede neural
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    print('\nRede neural treinada!')

    # Avaliar o modelo
    score = clf.score(X_test, y_test)
    print(f'Acurácia no teste: {score:.2%}')

    # Salvar o modelo treinado
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print('Modelo salvo em spam_model.pkl')


if __name__ == "__main__":
    main()


