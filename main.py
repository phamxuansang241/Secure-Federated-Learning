from tek4fed.fed_learn import FedAvg
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


sys.stdout = open(experiment.log_path, "w")
# Creating server and client
training_config = {
    'compress_digit': global_config['compress_digit'],
    'dataset_name': data_config['dataset_name'],
    'dp_mode': global_config['dp_mode'],
    'batch_size': fed_config['batch_size'], 
    'global_epochs': fed_config['global_epochs'], 
    'local_epochs': fed_config['local_epochs']
    }


server = Server(
    model_fn=get_model_function(data_config['dataset_name'], global_config['dp_mode']),
    weight_summarizer=FedAvg(),
    dp_mode=global_config['dp_mode'],
    fed_config=fed_config, dp_config=dp_config
    )
server.global_weight_path = experiment.global_weight_path
server.update_training_config(training_config)
server.create_clients()

# Preprocessing data and distributing data
DataSetup(data_config).setup(server)


# Set up clients
server.setup()

# Training model
print('[INFO] TRAINING MODEL ...')
assert global_config['training_mode'] in _supported_training_mode, "Unsupported training mode, this shouldn't happen"

start_time = time.perf_counter()

if global_config['training_mode'] == 'fedavg':
    server.train_fed()
elif global_config['training_mode'] == 'fed_compress':
    server.train_fed_compress()
elif global_config['training_mode'] == 'fed_elgamal':
    server.train_fed_elgamal_encryption()
elif global_config['training_mode'] == 'fed_ecc':
    server.train_fed_ecc_encryption(short_ver=True)
elif global_config['training_mode'] == 'dssgd':
    server.train_dssgd()
end_time = time.perf_counter()

print(f'[INFO] TOTAL TIME FOR TRAINING ALL EPOCHS {end_time-start_time}')

server.save_model_weights(experiment.global_weight_path)


# Evaluating model
print('[INFO] GET EXPERIMENT RESULTS ...')
get_experiment_result(server, experiment, data_config['dataset_name'])


sys.stdout.close()
sys.stdout = sys.__stdout__
