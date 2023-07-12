from tek4fed import get_args, server_lib
from tek4fed.summarizer_lib import FedAvg
from tek4fed.model_lib import get_model_function
from tek4fed.data_lib import DataSetup
from tek4fed.experiment_lib import Experiment, get_experiment_result
import json
import sys
import time


_supported_training_mode = ['fedavg', 'fed_compress', 'fed_ecc', 'fed_elgamal', 'dssgd']

# Argument parser and unpack JSON object
args = get_args()

with open(args.config_path, 'r') as openfile:
    config = json.load(openfile)

global_config = config['global_config']
data_config = config['data_config']
fed_config = config['fed_config']
dp_config = config['dp_config']


# Setting experiment

experiment_config = {
    'training_mode': global_config['training_mode'], 'name': global_config['name'], 'compress_digit': global_config['compress_digit'],
    'dataset_name': data_config['dataset_name'], 'data_sampling_technique': data_config['data_sampling_technique'],
    'overwrite_experiment': global_config['overwrite_experiment'],
    'nb_clients': fed_config['nb_clients'], 'fraction': fed_config['fraction'], 'global_epochs': fed_config['global_epochs']
}

experiment = Experiment(experiment_config)
experiment.serialize_config(config)


# sys.stdout = open(experiment.log_path, "w")
# Creating server and client
server_config = {
    'general_config': {
        'checkpoint_path': global_config['checkpoint_path'],
        'compress_digit': global_config['compress_digit'],
        'dp_mode': global_config['dp_mode'],
        'dataset_name': data_config['dataset_name'],
        'global_weight_path': experiment.global_weight_path
    },
    'fed_config': fed_config,
    'dp_config': dp_config
}

training_mode = global_config['training_mode']
server_cls = None
if training_mode == 'fedavg':
    server_cls = server_lib.FedServer
elif global_config['training_mode'] == 'fed_compress':
    server_cls = server_lib.FedCompressServer
elif global_config['training_mode'] == 'fed_elgamal':
    server_cls = server_lib.ElGamalEncryptionServer
elif global_config['training_mode'] == 'fed_ecc':
    server_cls = server_lib.EccEncryptionServer
elif global_config['training_mode'] == 'dssgd':
    server_cls = server_lib.DssgdServer

server = server_cls(
    model_fn=get_model_function(data_config['dataset_name'], global_config['dp_mode']),
    weight_summarizer=FedAvg(),
    server_config=server_config
    )

server.global_weight_path = experiment.global_weight_path
server.create_clients()

# Preprocessing data and distributing data
DataSetup(data_config).setup(server)


# Set up clients
server.setup()

# Training model
print('[INFO] TRAINING MODEL ...')
assert global_config['training_mode'] in _supported_training_mode, "Unsupported training mode, this shouldn't happen"

start_time = time.perf_counter()
server.train()
end_time = time.perf_counter()

print(f'[INFO] TOTAL TIME FOR TRAINING ALL EPOCHS {end_time-start_time}')


# Evaluating model
print('[INFO] GET EXPERIMENT RESULTS ...')
get_experiment_result(server, experiment, data_config['dataset_name'])


# sys.stdout.close()
# sys.stdout = sys.__stdout__
