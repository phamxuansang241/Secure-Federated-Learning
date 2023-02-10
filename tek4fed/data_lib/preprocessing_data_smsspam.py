from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import string
import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import Word
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def cleaning_smsspam_dataset(df_data):
    # covert uppercase letters to lowercase letters
    df_data['text'] = df_data['text'].apply(lambda x: ' '.join(
        x.lower() for x in x.split()
    ))

    # delete punctuation marks
    df_data['text'] = df_data['text'].str.replace('[^\w\s]', '')

    # delete numbers from texts
    df_data['text'] = df_data['text'].str.replace('\d', '')

    # delete stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop_words.update(punctuation)
    df_data['text'] = df_data['text'].apply(lambda x: ' '.join(
        x for x in x.split() if x not in stop_words
    ))

    # lemmatization and get the roots of the words
    df_data['text'] = df_data['text'].apply(lambda x: ' '.join(
        [Word(word).lemmatize() for word in x.split()]
    ))

    # remove words less than 3 letters
    df_data['text'] = df_data['text'].apply(lambda x: ' '.join(
        [x for x in x.split() if len(x) > 3]
    ))

    return df_data


def preprocessing_smsspam_dataset(datafile):
    # load dataset
    df = pd.read_csv(datafile, encoding='ISO-8859-1', engine='python')

    # rename dataset columns
    df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

    # drop unnecessary columns
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

    # drop duplicate data
    df.drop_duplicates(inplace=True)

    # cleaning data
    df = cleaning_smsspam_dataset(df)

    df['target'].replace({'ham': 0, 'spam': 1}, inplace=True)

    df_majority = df[df.target==0]
    df_minority = df[df.target==1]
    df_minority_oversampled = resample(df_minority, replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=123)
    df = pd.concat([df_majority, df_minority_oversampled])

    print("+++ smsspam dataset: +++")
    print("\tNumber of ham messages: ", len(df[df['target'] == 0]))
    print("\tNumber of spam messages: ", len(df[df['target'] == 1]))
    print("\tNumber of total requests: ", df.shape[0])
    return df


def smsspam_load_data(test_prob=0.2, max_len=71):
    data_file = './datasets/smsspam/spam.csv'
    data = preprocessing_smsspam_dataset(data_file)

    x_data = data['text'].values
    y_data = data['target'].values

    tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
    vocab = build_vocab_from_iterator([tokenizer(text) for text in x_data])

    text_pipeline = lambda x: vocab(tokenizer(x))

    temp_x_data = [text_pipeline(text) for text in x_data]
    temp_x_data = [np.asarray(sample, dtype=np.int32) for sample in temp_x_data]
    temp_x_data = [np.pad(sample, (0, max(0, max_len - len(sample))), mode='constant', constant_values=0) for sample in
                   temp_x_data]
    temp_x_data = [sample[:max_len] for sample in temp_x_data]
    x_data = np.asarray(temp_x_data, dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_prob,
                                                        shuffle=True, random_state=120124)

    return (x_train, y_train), (x_test, y_test)
