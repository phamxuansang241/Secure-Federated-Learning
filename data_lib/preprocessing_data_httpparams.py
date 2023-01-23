import data_lib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocessing_payload(payloads):
    new_payloads = []
    for i in range(len(payloads)):
        payload = payloads[i].strip()
        new_payloads.append(payload)

    return new_payloads


def preprocessing_httpparams_dataset(df, max_len):
    df.drop(['length', 'attack_type'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df['label'].replace({'norm': 0, 'anom': 1}, inplace=True)

    payloads = df['payload'].values
    x_data = preprocessing_payload(payloads)
    y_data = df['label'].values

    print("+++ httpparams dataset: +++")
    print("\tNumber of normal requests: ", len(df[df['label'] == 0]))
    print("\tNumber of anomalous requests: ", len(df[df['label'] == 1]))
    print("\tNumber of total requests: ", df.shape[0])

    vocab, reverse_vocab, vocab_size, alphabet = data_lib.create_vocab_set()
    x_data = data_lib.encode_data(x_data, max_len, vocab)

    return x_data, y_data


def httpparams_load_data(test_prob, max_len=500):
    df_train = pd.read_csv('datasets/httpparams/payload_train.csv')
    df_test = pd.read_csv('datasets/httpparams/payload_test.csv')
    df_full = pd.concat([df_train, df_test], ignore_index=True, sort=False)

    x_data, y_data = preprocessing_httpparams_dataset(df_full, max_len)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_prob,
                                                        shuffle=True, random_state=120124)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)
