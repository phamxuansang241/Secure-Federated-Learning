import shutil
from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def args_as_json(args):
    json_str = json.dumps(args.__dict__, sort_keys=True, indent=4)
    return json_str


def save_args_as_json(args, path):
    json_str = args_as_json(args)

    with open(str(path), 'w') as f:
        f.write(json_str)


def get_experiment_result(server, experiment, dataset_name):
    num_class = {
        'mnist': 10,
        'csic2010': 2, 'fwaf': 2, 'httpparams': 2, 'fusion': 2, 
        'smsspam': 2
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    temp_model = torch.load(experiment.global_weight_path).to(device)

    predictions = []
    with torch.no_grad():
        temp_model.eval()

        for (x_batch, _) in server.data_loader:
            x_batch = x_batch.float().to(device)
                                
            pred = temp_model(x_batch)
            pred_np = pred.detach().cpu().numpy()                
            
            predictions.extend(pred_np)

    predictions = np.array(predictions)
    y_test = server.y_test.detach().cpu().numpy()

    report_dict = None
    if num_class[dataset_name] > 2:
        score = classification_report(y_test, predictions.argmax(axis=1), output_dict=True)
        report_dict = {'report': score}
    else:
        score = confusion_matrix(y_test, predictions.argmax(axis=1))
        score = score * 1.0

        tn, fp, fn, tp = score.ravel()
        acc = (tn+tp) / (tn+fp+fn+tp)
        recall = tp / (tp+fn) 
        precision = tp / (tp + fp)
        fpr = fp/(tn+fp)
        drn = tn / (fp+tn)
        f_1 = (2 * recall * precision) / (recall + precision)

        report_dict = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                    'Accuracy': acc, 'Recall (TPR)': recall,
                    'Precision': precision, 'FPR (Fall-out)': fpr,
                    'DRN': drn, 'F_1 score': f_1
                    }
        
    with open(str(experiment.train_hist_path), 'r+') as f:
        data = json.load(f)
        data.update(report_dict)
        json.dump(data, f)
        
    print(report_dict)


class Experiment:
    def __init__(self, experiment_folder_path: Path, overwrite_if_exists: bool = False):
        self.experiment_folder_path = experiment_folder_path

        if self.experiment_folder_path.is_dir():
            if overwrite_if_exists:
                shutil.rmtree(str(self.experiment_folder_path))
            else:
                raise Exception('Experiment already exists')

        self.experiment_folder_path.mkdir(parents=True, exist_ok=False)

        self.config_json_path = self.experiment_folder_path / 'config.json'

        self.train_hist_path = self.experiment_folder_path / 'fed_learn_global_test_results.json'
        self.global_weight_path = self.experiment_folder_path / 'global_model.pth'

    def serialize_config(self, config):
        config_json_object = json.dumps(config, indent=4)
        with open(str(self.config_json_path), 'w') as f:
            f.write(config_json_object)

