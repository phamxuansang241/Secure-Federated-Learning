import numpy as np


def create_vocab_set():
    # create alphabet dictionary
    char_list = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    char_list_size = len(char_list)
    char_dict = {}
    reverse_char_dict = {}

    for ix, t in enumerate(char_list):
        char_dict[t] = ix
        reverse_char_dict[ix] = t

    return char_dict, reverse_char_dict, char_list_size, char_list


def encode_data(data, max_len, vocabulary):
    input_data = np.zeros((len(data), max_len), dtype=np.int8)
    for dix, sent in enumerate(data):
        counter = 0
        sent = str(sent)

        for c in sent:
            c = c.lower()
            if counter >= max_len:
                pass
            else:
                ix = vocabulary.get(c, 0)  # get index from vocab dictionary, if not in vocab, return 0
                input_data[dix, counter] = ix
                counter = counter + 1
    input_data = input_data.astype(np.float32)
    return input_data
