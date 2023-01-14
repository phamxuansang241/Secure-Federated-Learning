import torch
import dp_lib
import model_lib
import numpy as np
import math
from typing import Callable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from functorch import make_functional, grad, grad_and_value, vmap
import time
from opacus.accountants.utils import get_noise_multiplier
import numpy


class Client:
    def __init__(self, index: int):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.index = index
        self.model = None
        self.encoded_weights = None
        self.ceil_max_weight = None
        self.x_train = None
        self.y_train = None

    def init_model(self, model_fn: Callable, model_weights):
        temp_model = model_fn().to(self.device)
        model_lib.set_model_weights(temp_model, model_weights, used_device=self.device)
        self.model = temp_model

    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def receive_and_init_model(self, model_fn: Callable, model_weights):
        self.init_model(model_fn, model_weights)

    def edge_train(self, training_params: dict):
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))

        batch_size = training_params['batch_size']
        epochs = training_params['local_epochs']
        init_lr = 0.001

        x_train = torch.from_numpy(self.x_train)
        y_train = torch.from_numpy(self.y_train)

        train_dataset = TensorDataset(x_train, y_train)
        data_size = len(train_dataset)
        train_steps = data_size // batch_size

        train_data_loader = DataLoader(train_dataset,
                                       shuffle=True,
                                       batch_size=batch_size)

        opt = Adam(self.model.parameters(), lr=init_lr)
        loss_fn = nn.CrossEntropyLoss()

        # measure how long training is going to take
        start_time = time.time()

        history = {'loss': [], 'accuracy': []}

        for e in range(0, epochs):
            # set the model_lib in training mode
            self.model.train()

            # initialize the total training and validation loss
            total_train_loss = 0
            # initialize the number of correct predictions iin the training
            train_correct = 0

            # loop over the training set
            for (x_batch, y_batch) in train_data_loader:
                # send the input to the device
                (x_batch, y_batch) = (x_batch.long().to(self.device),
                                      y_batch.long().to(self.device))
                
                # perform a forward pass and calculate the training loss
                pred = self.model(x_batch)
                loss = loss_fn(pred, y_batch)

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()

                # add the loss to the total training loss
                # and calculate the number of correct predictions
                total_train_loss = total_train_loss + loss
                train_correct = train_correct + (pred.argmax(1) == y_batch).type(
                    torch.float
                ).sum().item()

            # calculate the average training loss
            avg_train_loss = total_train_loss / train_steps
            # print(trainCorrect)
            train_correct = train_correct / data_size

            # update our training history
            history['loss'].append(avg_train_loss.cpu().detach().numpy())
            history['accuracy'].append(train_correct)

        # finish measuring how long training took
        end_time = time.time()
        print("\t\t Total time taken to train: {:.2f}s".format(end_time - start_time))
        return history

    def edge_train_dp(self, training_params: dict, dp_config: dict):
        if self.model is None:
            raise ValueError('Model is not created for client: {0}'.format(self.index))

        print(training_params)
        print(dp_config)
        # Hyperparameters for training
        BATCH_SIZE = training_params['batch_size']
        local_epochs = training_params['local_epochs']
        init_lr = 0.001

        # Hyperparameters for differential privacy
        clipping_norm = dp_config['clipping_norm']
        
        # Convert data to tensor
        x_train = torch.from_numpy(self.x_train)
        y_train = torch.from_numpy(self.y_train)

        # Get size of the data
        train_dataset = TensorDataset(x_train, y_train)
        data_size = len(train_dataset)
        train_steps = data_size // BATCH_SIZE

        # Get the dataloader
        train_data_loader = DataLoader(train_dataset,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE)

        noise_scale = get_noise_multiplier(
            target_epsilon=dp_config['epsilon'], 
            target_delta=dp_config['delta'], 
            sample_rate=BATCH_SIZE/data_size,
            epochs=training_params['local_epochs']*training_params['global_epochs']
        )
        print('Noise scale to archive the DP: ', noise_scale)

        
        # Choose the optimizer algorithm
        opt = Adam(self.model.parameters(), lr=init_lr)
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        # measure how long training is going to take
        start_time = time.time()

        history = {'loss': [], 'accuracy': []}

        self.model.train()
        for e in range(local_epochs):
            # initialize the total training and validation loss
            total_train_loss = 0
            # initialize the number of correct predictions iin the training
            train_correct = 0

            ft_model, ft_params = make_functional(self.model)

            def compute_stateless_loss(params, sample_x, sample_y):
                batch_input = sample_x.unsqueeze(0)
                batch_label = sample_y.unsqueeze(0)

                output = ft_model(params, batch_input)
                loss = F.cross_entropy(output, batch_label)
                
                return loss
            
            ft_compute_grad = grad_and_value(compute_stateless_loss)
            ft_compute_sample_grad = vmap(ft_compute_grad, 
                                        in_dims=(None, 0, 0),
                                        randomness="same")

            params = list(self.model.parameters())
 
            for (x_batch, y_batch) in train_data_loader:
                
                # send the input to the device
                (x_batch, y_batch) = (x_batch.long().to(self.device),
                                      y_batch.long().to(self.device))

                # perform a forward pass
                pred = self.model(x_batch)
                
                # compute per-sample for gradients and losses
                per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                    params, x_batch, y_batch
                )

                per_sample_grads = [g.detach() for g in per_sample_grads]
                
                # add the loss to the total training loss
                # and calculate the number of correct predictions
                total_train_loss = total_train_loss + torch.mean(per_sample_losses)
                train_correct = train_correct + (pred.argmax(1) == y_batch).type(
                    torch.float
                ).sum().item()
                
                for param, layer_per_sample_grads in zip(self.model.parameters(), per_sample_grads):
                    layer_per_sample_grads = torch.stack(
                        [grad / max(1.0, float(grad.data.norm(2)) / clipping_norm) 
                        for grad in layer_per_sample_grads]
                    )
                    param.grad = torch.mean(layer_per_sample_grads, dim=0)
                    param.grad = param.grad + dp_lib.gaussian_noise(
                        param.shape, clipping_norm, 0.01, self.device
                    )

                opt.step()
                opt.zero_grad()
                # del x_batch
                # del y_batch
                # gc.collect()
            
            # calculate the average training loss
            avg_train_loss = total_train_loss / 5
            # print(trainCorrect)
            train_correct = train_correct / 5

            # update our training history
            history['loss'].append(avg_train_loss.cpu().detach().numpy())
            history['accuracy'].append(train_correct)



        # gc.collect()
        # finish measuring how long training took
        end_time = time.time()

        print("\t\t Total time taken to train: {:.2f}s".format(end_time - start_time))
        return history

    def add_noise(self, noise_level):
        client_weights = model_lib.get_model_weights(self.model)
        for layer_index in range(len(client_weights)):
            std = np.std(client_weights[layer_index])
            noise = np.random.normal(0.0, std * noise_level, size=client_weights[layer_index].shape)
            client_weights[layer_index] = client_weights[layer_index] + noise
        model_lib.set_model_weights(self.model, client_weights, used_device=self.device)

    def encode_compress_model(self, compress_nb):
        client_weights = model_lib.get_model_weights(self.model)
        weights_shape, _ = model_lib.get_model_infor(self.model)

        flatten_weights = model_lib.flatten_weight(client_weights)
        self.ceil_max_weight = math.ceil(np.max(np.abs(flatten_weights)))
        flatten_encoded_weights = np.ceil((flatten_weights*compress_nb) / self.ceil_max_weight)
        self.encoded_weights = model_lib.split_weight(flatten_encoded_weights, weights_shape)

    def reset_model(self):
        model_lib.get_rid_of_models(self.model)
