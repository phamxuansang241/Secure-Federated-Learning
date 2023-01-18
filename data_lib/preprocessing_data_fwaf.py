import data_lib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def preprocessing_fwaf_dataset(data_file_lst, label_lst, max_len):
    requests = []
    labels = []

    for idx, data_file in enumerate(data_file_lst):
        fin = open(data_file, 'r', encoding='utf8')
        lines = fin.readlines()

        for i in range(len(lines)):
            line = lines[i].strip()

            if len(line):
                requests.append(line)
                labels.append(label_lst[idx])

        fin.close()

    data_tuples = list(zip(requests, labels))
    df_full = pd.DataFrame(data_tuples, columns=['requests', 'labels'])
    df_full.drop_duplicates(inplace=True)
    print("+++ fwaf dataset: +++")
    print("\tNumber of normal requests: ", len(df_full[df_full['labels'] == 0]))
    print("\tNumber of anomalous requests: ", len(df_full[df_full['labels'] == 1]))
    print("\tNumber of total requests: ", df_full.shape[0])

    x_data = df_full['requests'].values
    y_data = df_full['labels'].values

    vocab, reverse_vocab, vocab_size, alphabet = data_lib.create_vocab_set()
    x_data = data_lib.encode_data(x_data, max_len, vocab)
    return x_data, y_data


def fwaf_load_data(test_prob, max_len):
    FWAF_goodqueries_file = 'datasets/fwaf/goodqueries.txt'
    FWAF_badqueries_file = 'datasets/fwaf/badqueries.txt'

    FWAF_data_file_lst = [FWAF_goodqueries_file, FWAF_badqueries_file]
    FWAF_label_lst = [0, 1]

    x_data, y_data = preprocessing_fwaf_dataset(FWAF_data_file_lst, FWAF_label_lst, max_len)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_prob,
                                                        shuffle=True, random_state=120124)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)

