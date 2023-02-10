import json
import os
import shutil

training_modes = ['fedavg', 'fed_compress', 'fed_ecc', 'fed_elgamal']
nb_clients = [5, 10, 20, 40, 50]
datasets = ['csic2010', 'mnist', 'smsspam']
data_sample_technique = ['iid', 'noniid_labeldir']
global_epochs = [2, 3]

for training_mode in training_modes:
    fodler = f'json_files/{training_mode}'
    if os.path.isdir(fodler):
        shutil.rmtree(fodler)

    for nb_client in nb_clients:
        for dataset in datasets:
            for technique in data_sample_technique:
                for ge in global_epochs:
                    config = {
                        "global_config": {
                            "name": "Test",
                            "overwrite_experiment": True,
                            "device": "cuda",
                            "training_mode": training_mode,
                            "compress_digit": 3,
                            "dp_mode": False
                        },
                        "data_config": {
                            "dataset_name": dataset,
                            "data_sampling_technique": technique
                        },
                        "fed_config": {
                            "global_epochs": ge,
                            "local_epochs": 1,
                            "nb_clients": nb_client,
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
                    
                    if training_mode == 'fed_compress':
                        compress_digit = config['global_config']['compress_digit']
                        filename = f'config_dg_{compress_digit}.json'
                    else:
                        filename = f'config.json'
                    directory = f'json_files/{training_mode}/{dataset}/{nb_client}_clients/{technique}/{ge}_global_epochs'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    filepath = os.path.join(directory, filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump(config, f, indent=4)