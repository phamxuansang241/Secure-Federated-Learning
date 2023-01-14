import shutil
from pathlib import Path
import json


def args_as_json(args):
    json_str = json.dumps(args.__dict__, sort_keys=True, indent=4)
    return json_str


def save_args_as_json(args, path):
    json_str = args_as_json(args)

    with open(str(path), 'w') as f:
        f.write(json_str)


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

