import shutil
from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datetime import date
import copy
from torch.utils.data import DataLoader
import gc


def get_experiment_result(server, experiment, dataset_name):
    num_class = {
        'mnist': 10,
        'csic2010': 2, 'fwaf': 2, 'httpparams': 2, 'fusion': 2, 
        'smsspam': 2, 'covid': 3
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(experiment.global_weight_path).to(device)

    model.eval()
    predictions = []
    y_test = []
    
    batch_size = server.training_config['batch_size']
    data_loader = DataLoader(server.dataset, batch_size=batch_size)
    with torch.no_grad():
        for (x_batch, y_batch) in data_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            predictions.extend(pred.detach().cpu().numpy())
            y_test.extend(y_batch)
    del data_loader
    gc.collect()
    
    predictions = np.array(predictions)
    score = None
    if num_class[dataset_name] > 2:
        score = classification_report(y_test, predictions.argmax(axis=1), output_dict=True, zero_division=1)
        report_dict = {'report': score}
    else:
        score = confusion_matrix(y_test, predictions.argmax(axis=1)) * 1.0
        tn, fp, fn, tp = score.ravel()
        acc = (tn+tp) / (tn+fp+fn+tp)
        recall = tp / (tp+fn) 
        precision = tp / (tp + fp)
        fpr = fp/(tn+fp)
        drn = tn / (fp+tn)
        f_1 = (2 * recall * precision) / (recall + precision)

        result = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                  'Accuracy': acc, 'Recall (TPR)': recall,
                  'Precision': precision, 'FPR (Fall-out)': fpr,
                  'DRN': drn, 'F_1 score': f_1}
        report_dict = {'report': result}
        
    with open(str(experiment.train_hist_path), 'w') as f:
        data = copy.deepcopy(server.global_test_metrics)
        data.update(report_dict)
        json.dump(data, f, indent=4)
        

class Experiment:
    def __init__(self, experiment_config: dict):
        self.experiment_config = experiment_config
        self.experiment_folder_path = self.setup_experiment_folder_path()

        if self.experiment_folder_path.is_dir():
            if experiment_config['overwrite_experiment']:
                shutil.rmtree(str(self.experiment_folder_path))
            else:
                raise Exception('Experiment already exists')

        self.experiment_folder_path.mkdir(parents=True, exist_ok=False)

        self.config_json_path = self.experiment_folder_path / 'config.json'
        self.log_path = self.experiment_folder_path / 'log.txt'
        self.train_hist_path = self.experiment_folder_path / 'results.json'
        self.global_weight_path = self.experiment_folder_path / 'global_model.pth'

    def setup_experiment_folder_path(self):
        training_mode = self.experiment_config['training_mode']
        dataset_name = self.experiment_config['dataset_name']
        experiment_name = date.today().strftime("%b_%d_%Y") + '_' + self.experiment_config['name']
        if training_mode == 'fed_compress':
            experiment_name = experiment_name + '_dg_' + str(self.experiment_config['compress_digit'])

        global_epochs_str = str(self.experiment_config['global_epochs']) + '_global_epochs'
        nb_clients_str = str(self.experiment_config['nb_clients']) + "_clients"
        fraction = str(self.experiment_config['fraction']) + "_fraction"
        experiment_folder_path = Path('FL-DP').resolve().parent / 'experiments' / training_mode / dataset_name / nb_clients_str / self.experiment_config['data_sampling_technique'] / global_epochs_str / fraction / experiment_name

        print(experiment_folder_path)
        return experiment_folder_path

    def serialize_config(self, config):
        config_json_object = json.dumps(config, indent=4)
        with open(str(self.config_json_path), 'w') as f:
            f.write(config_json_object)
