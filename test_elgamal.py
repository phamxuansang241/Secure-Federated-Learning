from tek4fed.fed_learn import get_args, Server, FedAvg
import json
import encryption_lib
from tek4fed.model_lib import get_model_function, set_model_weights


args = get_args()

with open(args.config_path, 'r') as openfile:
    config = json.load(openfile)

global_config = config['global_config']
data_config = config['data_config']
fed_config = config['fed_config']
dp_config = config['dp_config']


training_config = {
    'compress_digit': global_config['compress_digit'],
    'dataset_name': data_config['dataset_name'],
    'dp_mode': global_config['dp_mode'],
    'batch_size': fed_config['batch_size'], 
    'global_epochs': fed_config['global_epochs'], 
    'local_epochs': fed_config['local_epochs']
    }


weight_summarizer = FedAvg()
server = Server(get_model_function(data_config['dataset_name']), weight_summarizer, training_config, fed_config, dp_config)

server.update_training_config(training_config)
server.create_clients()




encrypt = encryption_lib.ElGamalEncryption(server.nb_clients, 10)

encrypt.generate_client_noise_mtx()
encrypt.generate_client_key()

for epoch in int(server.training_config['global_epochs']):
    selected_clients =server.select_clients()
    encrypt.calculate_server_public_key()