from tek4fed import data_lib
import pandas as pd
import torch
from torchtext import data


seed = 2023
torch.manual_seed(seed)


def preprocessing_smsspam_dataset(data_file):
    text = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
    label = data.LabelField(dtype = torch.float,batch_first=True)
    fields = [("type", label),('text', text)]

    training_data = data.TabularDataset(path=data_file, format="csv", fields=fields, skip_header=True)
    print(dir(training_data))

    df = pd.read_csv(data_file)
    print("+++ httpparams dataset: +++")
    print("\tNumber of ham texts: ", len(df[df['type'] == 'ham']))
    print("\tNumber of spam texts: ", len(df[df['type'] == 'spam'))
    print("\tNumber of total requests: ", df.shape[0])

def smsspam_load_data(test_prob):
    data_file = "/datasets/smsspam/spam.csv"
    preprocessing_smsspam_dataset(data_file)
    # print(dir(pre))

