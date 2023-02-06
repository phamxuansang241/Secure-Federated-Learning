import urllib.parse
import pandas as pd
from tek4fed import data_lib
from sklearn.model_selection import train_test_split


def get_method_msgbody(file_in):
    fin = open(file_in)
    lines = fin.readlines()
    prefix = 'http://localhost:8080/'
    method_msgbody = []

    for i in range(len(lines)):
        line = lines[i].strip()

        if line.startswith('GET'):
            url = line.split(' ')[1]
            url = urllib.parse.unquote(urllib.parse.unquote(url)).replace('\n', '')
            url = url[len(prefix):]
            method_msgbody.append('GET ' + url)
        elif line.startswith('POST') or line.startswith('PUT'):
            method = line.split(' ')[0]
            url = line.split(' ')[1]
            url = url[len(prefix):]

            # merging the query to the method field
            j = 1
            while True:
                if lines[i + j].startswith('Content-Length'):
                    break
                j = j + 1
            j = j + 1

            data = lines[i + j + 1].strip()
            url = url + '?' + data
            url = urllib.parse.unquote(urllib.parse.unquote(url)).replace('\n', '')
            method_msgbody.append(method + ' ' + url)
    fin.close()

    return method_msgbody


def preprocessing_csic2010_dataset(data_file_lst, label_lst, max_len):
    df_full = pd.DataFrame()
    for idx, data_file in enumerate(data_file_lst):
        requests = get_method_msgbody(data_file)
        df = pd.DataFrame(requests)
        df.rename(columns={0: 'requests'}, inplace=True)
        df['labels'] = label_lst[idx]

        if idx == 0:
            df_full = df
        else:
            df_full = pd.concat([df_full, df], ignore_index=True, sort=False)

    df_full.drop_duplicates(inplace=True)
    print("+++ csic2010 dataset: +++")
    print("\tNumber of normal requests: ", len(df_full[df_full['labels'] == 0]))
    print("\tNumber of anomalous requests: ", len(df_full[df_full['labels'] == 1]))
    print("\tNumber of total requests: ", df_full.shape[0])

    x_data = df_full['requests'].values
    y_data = df_full['labels'].values

    vocab, _, _, _ = data_lib.create_vocab_set()
    x_data = data_lib.encode_data(x_data, max_len, vocab)
    return x_data, y_data


def csic2010_load_data(test_prob, max_len=500):
    normal_file_1 = 'datasets/csic2010/normalTrafficTraining.txt'
    normal_file_2 = 'datasets/csic2010/normalTrafficTest.txt'
    anomalous_file = 'datasets/csic2010/anomalousTrafficTest.txt'

    data_file_lst = [normal_file_1, normal_file_2, anomalous_file]
    label_lst = [0, 0, 1]
    x_data, y_data = preprocessing_csic2010_dataset(data_file_lst, label_lst, max_len)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_prob,
                                                        shuffle=True, random_state=120124)

    return (x_train, y_train), (x_test, y_test)
