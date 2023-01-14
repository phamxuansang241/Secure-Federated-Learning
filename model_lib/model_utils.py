import model_lib
import torch
import gc
import numpy as np


def set_model_weights(model, weights_list, to_device=True, used_device='cpu'):
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
    weights_shape = [model_weights[i].shape
                    for i in range(len(model_weights))]
    total_params = sum([np.prod(weights_shape[i]) for i in range(len(weights_shape))])

    return weights_shape, total_params


def flatten_weight(weights):
    weights_flatten_vector = np.array([], dtype=np.float32)

    for i in range(len(weights)):
        weights_layer_i = weights[i].reshape(-1, )
        weights_flatten_vector = np.concatenate((weights_flatten_vector, weights_layer_i),
                                                axis=None)

    return weights_flatten_vector


def split_weight(weights_flatten_vector, weights_shape):
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