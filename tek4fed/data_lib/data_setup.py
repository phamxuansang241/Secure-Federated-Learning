from tek4fed import data_lib
import numpy as np
import math


class DataSetup:
    def __init__(self, data_config) -> None:
        print('-' * 100)
        print('[INFO] SETTING UP DATASET ...')
        
        self.dataset_name = data_config['dataset_name']
        self.data_sampling_technique = data_config['data_sampling_technique']
        self.nb_class = None

        (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)
        (self.x_csic2010_train, self.y_csic2010_train) = (None, None), (None, None)
        (self.x_fwaf_train, self.y_fwaf_train) = (None, None), (None, None)
        (self.x_httpparams_train, self.y_httpparams_train) = (None, None), (None, None)

    def setup(self, server):
        self.preprocessing_data()
        self.distribute_data(server)

    def preprocessing_data(self):
        if self.dataset_name == 'csic2010':
            print('Using csic2010 dataset ...')
            (self.x_train, self.y_train), (self.x_test, self.y_test) = data_lib.csic2010_load_data(0.2)
            self.nb_class = 2
        elif self.dataset_name == 'fwaf':
            print('Using fwaf dataset ...')
            (self.x_train, self.y_train), (self.x_test, self.y_test) = data_lib.fwaf_load_data(0.2)
            self.nb_class = 2
        elif self.dataset_name == 'httpparams':
            print('Using httpparams dataset ...')
            (self.x_train, self.y_train), (self.x_test, self.y_test) = data_lib.httpparams_load_data(0.2)
            self.nb_class = 2
        elif self.dataset_name == 'fusion':
            print('Using three datasets: csic2010, fwaf, httpparams ...')
            (self.x_csic2010_train, self.y_csic2010_train), (x_csic2010_test, y_csic2010_test) = data_lib.csic2010_load_data(0.2)
            (self.x_fwaf_train, self.y_fwaf_train), (x_fwaf_test, y_fwaf_test) = data_lib.fwaf_load_data(0.2)
            (self.x_httpparams_train, self.y_httpparams_train), (x_httpparams_test, y_httpparams_test) = data_lib.httpparams_load_data(0.2)

            self.x_test = np.concatenate((x_csic2010_test, x_fwaf_test, x_httpparams_test), axis=0)
            self.y_test = np.concatenate((y_csic2010_test, y_fwaf_test, y_httpparams_test), axis=0)
            self.nb_class = 2
        elif self.dataset_name == 'mnist':
            print('Using mnist dataset ...')
            (self.x_train, self.y_train), (self.x_test, self.y_test) = data_lib.mnist_load_data()
            self.nb_class = 10

    def distribute_data(self, server):
        if self.dataset_name == 'fusion':
            nb_clients_each_datasets = math.ceil(len(server.clients) / 3)

            data_handler = data_lib.DataHandler(self.x_csic2010_train, self.y_csic2010_train)
            data_handler.assign_data_to_clients(server.clients[0:nb_clients_each_datasets],
                                                self.data_sampling_technique)
            del data_handler

            data_handler = data_lib.DataHandler(self.x_fwaf_train, self.y_fwaf_train)
            data_handler.assign_data_to_clients(server.clients[nb_clients_each_datasets:2*nb_clients_each_datasets],
                                                self.data_sampling_technique)
            del data_handler

            data_handler = data_lib.DataHandler(self.x_httpparams_train, self.y_httpparams_train)
            data_handler.assign_data_to_clients(server.clients[2*nb_clients_each_datasets:],
                                                self.data_sampling_technique)
            del data_handler
        else:
            data_handler = data_lib.DataHandler(self.x_train, self.y_train)
            data_handler.assign_data_to_clients(server.clients,
                                                self.data_sampling_technique)
            del data_handler

        server.receive_data(self.x_test, self.y_test)


    
