import fed_learn
import model_lib
import data_lib
import compress_params_lib
import experiments

import copy
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from pathlib import Path
import json
from datetime import date
import math
from sklearn.metrics import SCORERS, classification_report
from sklearn.metrics import confusion_matrix


print("*** FIX VER 11")
"""
    ARGUMENT PARSER AND UNPACK JSON OBJECT
"""
args = fed_learn.get_args()

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
experiment = experiments.Experiment(experiment_folder_path,  global_config['overwrite_experiment'])
experiment.serialize_config(config)

'''
    CREATING MODEL
'''

def model_fn():
    model = None
    if data_config['dataset_name'] == 'mnist':
        model = model_lib.Mnist_Net(num_class=10)
    else:    
        model = model_lib.CNN(vocab_size=70, embed_dim=128, input_length=500, num_class=2)

    return model


"""
    CREATING SERVER AND CLIENT
"""

training_config = {
    'dp_mode': global_config['dp_mode'],
    'batch_size': fed_config['batch_size'], 
    'global_epochs': fed_config['global_epochs'], 
    'local_epochs': fed_config['local_epochs']
    }


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn, weight_summarizer, training_config, fed_config, dp_config)

server.update_training_config(training_config)
server.create_clients()

"""
    PREPROCESSING DATA AND DISTRIBUTING DATA
"""
data_lib.DataSetup().setup(server)


"""
    SET UP CLIENTS
"""
server.setup()

"""
    GET MODEL INFORMATION
"""
temp_model = model_fn()
weights_shape, total_params = model_lib.get_model_infor(temp_model)

print('-' * 100)
print("[INFO] MODEL INFORMATION ...")
print("\t Model Weight Shape: ", weights_shape)
print("\t Total Params of model: ", total_params)
print()


"""
    TRAINING MODEL
"""
print('[INFO] TRAINING MODEL ...')
compress_params = compress_params_lib.CompressParams(global_config['compress_digit'])
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
    global_test_results = server.test_global_model(x_test, y_test)
    print('--- Global test ---')

    test_loss = global_test_results['loss']
    test_acc = global_test_results['accuracy']
    print('{0}: {1}'.format('\tLoss: ', test_loss))
    print('{0}: {1}'.format('\tAccuracy: ', test_acc))
    print('-'*100)

    with open(str(experiment.train_hist_path), 'w') as f:
        test_dict = copy.deepcopy(server.global_test_metrics)
        json.dump(test_dict, f)

    server.save_model_weights(experiment.global_weight_path)


'''
    EVALUATING MODEL
'''
print('[INFO] Evaluating model ...')
batch_size = 32
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

test_dataset = TensorDataset(x_test, y_test)
test_data_loader = DataLoader(test_dataset,
                              shuffle=False,
                              batch_size=batch_size)
                              
test_steps = len(test_data_loader) // batch_size
loss_fn = nn.CrossEntropyLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
temp_model = torch.load(experiment.global_weight_path).to(device)

predictions = []
with torch.no_grad():
    temp_model.eval()

    for (x_batch, y_batch) in test_data_loader:
        x_batch = x_batch.long().to(device)
                             
        pred = temp_model(x_batch)
        pred_np = pred.detach().cpu().numpy()                
        
        predictions.extend(pred_np)

predictions = np.array(predictions)
score = confusion_matrix(y_test.detach().cpu().numpy(), predictions.argmax(axis=1))
score = score * 1.0
print(score)

tn, fp, fn, tp = score.ravel()
acc = ((tn+tp)/(tn+fp+fn+tp))
recall = tp / (tp+fn) 
precision = tp / (tp + fp)
fpr = fp/(tn+fp)
drn = tn / (fp+tn)
f_1 = (2 * recall * precision) / (recall + precision)

result_dict = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
               'Accuracy': acc, 'Recall (TPR)': recall,
               'Precision': precision, 'FPR (Fall-out)': fpr,
               'DRN': drn, 'F_1 score': f_1
               }

with open(str(experiment.train_hist_path), 'r+') as f:
    data = json.load(f)
    data.update(result_dict)
    json.dump(data, f)

print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)
print("TP: ", tp)
print("Accuracy: ", acc)
print("Recall (TPR): ", recall)
print("Precision: ", precision)
print("FPR (Fall-out): ", fpr)
print("DRN: ", drn)
print("F_1 score: ", f_1)

