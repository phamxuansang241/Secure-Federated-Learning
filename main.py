from fed_learn import get_args, Server, FedAvg
import model_lib
from data_lib import DataSetup
from tek4fed.compress_params_lib import CompressParams
from tek4fed.experiment_lib import Experiment, get_experiment_result
import copy
import numpy as np
from pathlib import Path
import json
from datetime import date


print("*** FIX VER 11")
"""
    ARGUMENT PARSER AND UNPACK JSON OBJECT
"""
args = get_args()

with open(args.config_path, 'r') as openfile:
    config = json.load(openfile)

global_config = config['global_config']
data_config = config['data_config']
fed_config = config['fed_config']
dp_config = config['dp_config']

print('Overwrite experiment mode: ', global_config['overwrite_experiment'])


"""
    SETTING EXPERIMENT
"""

today = date.today().strftime("%b-%d-%Y")
global_config['name'] = today + '_' + global_config['name'] + '_'


experiment_folder_path = Path(__file__).resolve().parent / 'experiments' / data_config['dataset_name'] / global_config['name']
experiment = Experiment(experiment_folder_path, global_config['overwrite_experiment'])
experiment.serialize_config(config)


"""
    CREATING SERVER AND CLIENT
"""

training_config = {
    'dataset_name': data_config['dataset_name'],
    'dp_mode': global_config['dp_mode'],
    'batch_size': fed_config['batch_size'], 
    'global_epochs': fed_config['global_epochs'], 
    'local_epochs': fed_config['local_epochs']
    }


weight_summarizer = FedAvg()
server = Server(model_lib.get_model_function(data_config['dataset_name']), weight_summarizer, training_config, fed_config, dp_config)

server.update_training_config(training_config)
server.create_clients()

"""
    PREPROCESSING DATA AND DISTRIBUTING DATA
"""
DataSetup(data_config).setup(server)


"""
    SET UP CLIENTS
"""
server.setup()

"""
    TRAINING MODEL
"""
print('[INFO] TRAINING MODEL ...')
compress_params = CompressParams(global_config['compress_digit'])
print('\t Compress number:', compress_params.compress_number)

for epoch in range(fed_config['global_epochs']):
    print('[INFO] Global Epoch {0} is starting ...'.format(epoch))

    server.init_for_new_epoch()
    selected_clients = server.select_clients()
    clients_ids = [c.index for c in selected_clients]
    print('Selected clients for epoch: {0}'.format('| '.join(map(str, clients_ids))))

    for client in selected_clients:
        print('\t Client {} is starting the training'.format(client.index))
        
        if global_config['dp_mode']:
            if client.current_iter > client.max_allow_iter:
                break

        model_lib.set_model_weights(client.model, server.global_model_weights, client.device)
        client_losses = client.edge_train()

        print('\t\t Encoding parameters ...')
        compress_params.encode_model(client=client)

        server.epoch_losses.append(client_losses[-1])

    decoded_weights = compress_params.decode_model(selected_clients)
    server.client_model_weights = decoded_weights.copy()
    server.summarize_weights()

    epoch_mean_loss = np.mean(server.epoch_losses)
    server.global_train_losses.append(epoch_mean_loss)
    print('\tLoss (client mean): {0}'.format(server.global_train_losses[-1]))

    # testing current model_lib and save weights
    global_test_results = server.test_global_model()
    print("\t----- Evaluating on server's test dataset -----")

    test_loss = global_test_results['loss']
    test_acc = global_test_results['accuracy']
    print('{0}: {1}'.format('\tLoss', test_loss))
    print('{0}: {1}'.format('\tAccuracys', test_acc))
    print('-'*100)

    with open(str(experiment.train_hist_path), 'w') as f:
        test_dict = copy.deepcopy(server.global_test_metrics)
        json.dump(test_dict, f)

    server.save_model_weights(experiment.global_weight_path)


'''
    EVALUATING MODEL
'''
print('[INFO] GET EXPERIMENT RESULTS ...')
get_experiment_result(server, experiment, data_config['dataset_name'])


