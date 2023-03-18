import json
import os
import shutil
import copy


training_modes = ['fedavg', 'fed_compress', 'fed_ecc', 'fed_elgamal']
nb_clients = [50]
datasets = ['csic2010', 'mnist', 'smsspam']
data_sample_technique = ['iid', 'noniid_labeldir']
global_epochs = [2, 50]
digits = [3, 5, 10]
fraction = [0.98, 0.94, 0.9, 0.8]

config = {
        "global_config": {
            "name": "Test",
            "overwrite_experiment": True,
            "device": "cuda",
            "training_mode": 'fedavg',
            "compress_digit": 3,
            "dp_mode": False
        },
        "data_config": {
            "dataset_name": 'csic2010',
            "data_sampling_technique": 'iid'
        },
        "fed_config": {
            "global_epochs": 5,
            "local_epochs": 1,
            "nb_clients": 5,
            "fraction": 1.0,
            "batch_size": 32
        },
        "optim_config": {
            "lr": 0.001
        },
        "dp_config": {
            "epsilon": 50,
            "delta": 3e-05,
            "clipping_norm": 2.0,
            "is_fixed_client_iter": True,
            "client_iter": 700
        }
    }

for training_mode in training_modes:
    folder = f'json_files\{training_mode}'
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    for nb_client in nb_clients:
        for dataset in datasets:
            for technique in data_sample_technique:
                for ge in global_epochs:
                    for fr in fraction:

                        temp_config = copy.deepcopy(config)
                        temp_config['global_config']['training_mode'] = training_mode
                        temp_config['data_config']['dataset_name'] = dataset
                        temp_config['data_config']['data_sampling_technique'] = technique
                        temp_config['fed_config']['global_epochs'] = ge
                        temp_config['fed_config']['nb_clients'] = nb_client
                        temp_config['fed_config']['fraction'] = fr

                        directory = f'json_files/{training_mode}/{dataset}/{nb_client}_clients/{technique}/{ge}_global_epochs/{fr}_fraction'
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        if training_mode == 'fed_compress':
                            for dg in digits:
                                temp_config['global_config']['compress_digit'] = dg
                                filename = f'config_dg_{dg}.json'
                                filepath = os.path.join(directory, filename)

                                with open(filepath, 'w') as f:
                                    json.dump(temp_config, f, indent=4)
                        else:
                            filename = f'config.json'
                            filepath = os.path.join(directory, filename)
                        
                            with open(filepath, 'w') as f:
                                json.dump(temp_config, f, indent=4)