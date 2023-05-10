from tek4fed import model_lib
import torch
import gc
import numpy as np
import copy
from typing import List
from opacus.validators import ModuleValidator


def set_model_weights(model, weights_list: List[np.ndarray], to_device=True, used_device='cpu'):
    for weight, param in zip(weights_list, model.parameters()):
        if to_device:
            data = torch.from_numpy(weight).to(used_device)
        else:
            data = torch.from_numpy(weight)
        
        param.data = data


def get_model_weights(model):
    weights_list = []
    for param in model.parameters():
        param_np = param.cpu().detach().numpy()
        weights_list.append(param_np)

    return weights_list


def get_model_infor(model):
    model_weights = model_lib.get_model_weights(model)
    weights_shape = [model_weights[i].shape for i in range(len(model_weights))]
    total_params = sum([np.prod(weights_shape[i]) for i in range(len(weights_shape))])

    return weights_shape, total_params


def flatten_weight(weights: List[np.ndarray]):
    weights_flatten_vector = np.array([], dtype=np.float32)

    for i in range(len(weights)):
        weights_layer_i = weights[i].reshape(-1, )
        weights_flatten_vector = np.concatenate((weights_flatten_vector, weights_layer_i),
                                                axis=None)

    return weights_flatten_vector


def weight_to_mtx(weights: List[np.ndarray], mtx_size: int):
    weights_flatten_vector = flatten_weight(weights)
    weights_flatten_vector.resize(mtx_size*mtx_size, refcheck=False)
    weights_mtx = weights_flatten_vector.reshape(mtx_size, mtx_size)

    return weights_mtx


def split_weight(weights_flatten_vector: List[np.ndarray], weights_shape: List):
    weights_split = []
    start = 0

    for i in range(len(weights_shape)):
        weights_layer_i_flatten_end = start + np.prod(weights_shape[i])
        weights_layer_i = weights_flatten_vector[start: weights_layer_i_flatten_end]
        weights_layer_i = np.reshape(weights_layer_i, weights_shape[i])

        weights_split.append(weights_layer_i)
        start = weights_layer_i_flatten_end

    return weights_split


def get_rid_of_models(model=None):
    if model is not None:
        del model
    gc.collect()


def get_model_function(dataset_name, dp_mode):
    def model_function():
        if dataset_name == 'mnist':
            model = model_lib.Mnist_Net(num_class=10)
        elif dataset_name == 'smsspam':
            model = model_lib.LSTMNet(vocab_size=6972, embed_dim=64, hidden_dim=16, nb_classes=2, n_layers=2,
                                      dp_mode=dp_mode)
        else:
            model = model_lib.CNN(vocab_size=70, embed_dim=128, input_length=500, num_class=2)

        if dp_mode and not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

        return model

    return model_function


def get_dssgd_update(client_model, server_weights, weights_shape, theta_upload=0.9) -> List[np.ndarray]:
    client_weights = get_model_weights(client_model)

    # flatten client weights and server weights
    client_weights_row = flatten_weight(client_weights)
    server_weights_row = flatten_weight(server_weights)

    delta_weights = server_weights_row - client_weights_row
    indexes = np.argsort(delta_weights)[::-1][:int(len(delta_weights)*theta_upload)]
    
    server_weights_row[indexes] = client_weights_row[indexes]
    upload_weights = copy.deepcopy(server_weights_row)

    upload_weights = split_weight(upload_weights, weights_shape)
    return upload_weights
