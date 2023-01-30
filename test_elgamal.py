from tek4fed.fed_learn import get_args, Server, FedAvg
from tek4fed.data_lib import DataSetup
import json
import encryption_lib
from tek4fed.model_lib import get_model_function, set_model_weights, get_model_weights, weight_to_mtx
import numpy as np



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
DataSetup(data_config).setup(server)
server.setup()

mtx_size = 5
encrypt = encryption_lib.ElGamalEncryption(server.nb_clients, mtx_size)

encrypt.generate_client_noise_mtx()
encrypt.generate_client_key()


for epoch in range(int(server.training_config['global_epochs'])):
    server.init_for_new_epoch()
    selected_clients = server.select_clients()
    encrypt.calculate_server_public_key(selected_clients)

    sum_r_mtx = np.zeros((mtx_size, mtx_size))
    client_sum_weight = np.zeros((mtx_size, mtx_size))
    for client in selected_clients:
        sum_r_mtx = sum_r_mtx + encrypt.client_noise_mtx['r_i'][client.index]
        encrypt.encoded_message(client)

        client_weights = get_model_weights(client.model)
        client_weight_mtx = weight_to_mtx(client_weights, mtx_size)

        client_sum_weight = client_sum_weight + client_weight_mtx

    print()
    print("Matrix r of client")
    print(sum_r_mtx)
    
    print()
    print("Server weights")
    
    print(encrypt.decoded_message(selected_clients))  
    np.set_printoptions(suppress=True, precision=8)
    
    print()
    print("Matrix r of server")
    print(encrypt.server_decoded_message['R'])


    print()
    print("Sum weights of clients")
    print(client_weight_mtx)
    encrypt.decoded_message(selected_clients)
