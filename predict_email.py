import pandas as pd
import numpy as np
import pickle
import re

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
    'capital_run_length_longest', 'capital_run_length_total'
]

def extract_features(email_text):
    # Função simplificada: extrai frequências de palavras e caracteres do email
    features = []
    words = email_text.lower().split()
    total_words = len(words)
    total_chars = len(email_text)
    # Frequência de palavras
    for col in COLUMN_NAMES[:48]:
        word = col.replace('word_freq_', '')
        freq = 100 * words.count(word) / total_words if total_words > 0 else 0
        features.append(freq)
    # Frequência de caracteres
    char_list = [';', '(', '[', '!', '$', '#']
    for char in char_list:
        freq = 100 * email_text.count(char) / total_chars if total_chars > 0 else 0
        features.append(freq)
    # Run-length de maiúsculas (simplificado)
    capital_runs = re.findall(r'[A-Z]+', email_text)
    if capital_runs:
        avg_run = np.mean([len(run) for run in capital_runs])
        longest_run = np.max([len(run) for run in capital_runs])
        total_run = np.sum([len(run) for run in capital_runs])
    else:
        avg_run = 0
        longest_run = 0
        total_run = 0
    features.extend([avg_run, longest_run, total_run])
    return np.array(features).reshape(1, -1)

# Carregar modelo treinado
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

def main():
    print("=== Detector de Spam ===")
    while True:
        print("\nMenu:")
        print("1 - Analisar e-mail")
        print("2 - Sair")
        opcao = input("Escolha uma opção: ")
        if opcao == "1":
            email_text = input('\nDigite o texto do e-mail para análise:\n')
            X_new = extract_features(email_text)
            pred = model.predict(X_new)[0]
            print('Resultado: Spam' if pred == 1 else 'Resultado: Não é spam')
        elif opcao == "2":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
