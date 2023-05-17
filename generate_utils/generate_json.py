import argparse
import json
import os
import shutil
import copy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate configuration files.')

    parser.add_argument('--training_modes', nargs='+', default=['fedavg'], help='List of training modes.')
    parser.add_argument('--nb_clients', nargs='+', type=int, default=[50], help='List of the number of clients.')
    parser.add_argument('--datasets', nargs='+', default=['csic2010', 'smsspam'], help='List of dataset names.')
    parser.add_argument('--data_sample_techniques', nargs='+', default=['iid', 'noniid_label_dir'], help='List of data sampling techniques.')
    parser.add_argument('--global_epochs', nargs='+', type=int, default=[50], help='List of global epoch values.')
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help='List of fraction values.')
    parser.add_argument('--digits', nargs='+', type=int, default=[3, 5, 10], help='List of digits.')
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0.1], help='List of sigma values.')
    parser.add_argument('--dp_modes', nargs='+', type=str2bool, default=[False], help='List of dp_mode values (True or False).')
    
    return parser.parse_args()


def generate_configs(training_modes, nb_clients, datasets, data_sample_techniques, 
                     global_epochs, fractions, digits, dp_modes, sigmas, config):
    for training_mode in training_modes:
        folder = f'generate_utils/generated_files/json_files/{training_mode}'
        if os.path.isdir(folder):
            shutil.rmtree(folder)

        for nb_client in nb_clients:
            for dataset in datasets:
                for technique in data_sample_techniques:
                    for ge in global_epochs:
                        for fr in fractions:
                            for dp_mode in dp_modes:
                                if dp_mode:
                                    for sigma in sigmas:                                    
                                        temp_config = create_config(config, training_mode, dataset, technique, 
                                                                    ge, nb_client, fr, dp_mode, sigma)
                                        sub_directory = create_sub_directory(training_mode, dataset, nb_client, technique, 
                                                                            ge, fr, dp_mode, sigma)
                                        create_directory(sub_directory)
                                        
                                        print(sub_directory)
                                        if training_mode == 'fed_compress':    
                                            for dg in digits:
                                                    write_config(sub_directory, temp_config, dg)
                                        else:
                                            write_config(sub_directory, temp_config)
                            
                                else:
                                    temp_config = create_config(config, training_mode, dataset, technique, 
                                                                        ge, nb_client, fr, dp_mode, None)
                                    sub_directory = create_sub_directory(training_mode, dataset, nb_client, technique, 
                                                                                ge, fr, dp_mode, None)
                                    print(sub_directory)
                                    create_directory(sub_directory)

                                    if training_mode == 'fed_compress':
                                        for dg in digits:
                                            write_config(sub_directory, temp_config, dg)
                                    else:
                                        write_config(sub_directory, temp_config)



def create_config(config, training_mode, dataset, technique, 
                  ge, nb_client, fr, dp_mode, sigma):
    temp_config = copy.deepcopy(config)
    temp_config['global_config']['training_mode'] = training_mode
    temp_config['data_config']['dataset_name'] = dataset
    temp_config['data_config']['data_sampling_technique'] = technique
    temp_config['fed_config']['global_epochs'] = ge
    temp_config['fed_config']['nb_clients'] = nb_client
    temp_config['fed_config']['fraction'] = fr
    temp_config['global_config']['dp_mode'] = dp_mode
    
    if dp_mode:
        temp_config['dp_config']['sigma'] = sigma
        temp_config['global_config']['name'] = f"Test - dp_mode {dp_mode} - sigma {sigma}"
    else:
        temp_config['global_config']['name'] = f"Test - dp_mode {dp_mode}"
    
    return temp_config


def create_sub_directory(training_mode, dataset, nb_client, technique, 
                         ge, fr, dp_mode, sigma):
     
    sub_directory = f'generate_utils/generated_files/json_files/{training_mode}/{dataset}/{nb_client}_clients/{technique}/{ge}_global_epochs/{fr}_fraction/{dp_mode}_dp_mode'
    
    if dp_mode:  # Add sigma to the subdirectory name if dp_mode is True
        sub_directory += f"/{sigma}_sigma"
        
    return sub_directory
    

def create_directory(sub_directory):
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)


def write_config(sub_directory, config, dg=None):
    filename = f'config_dg_{dg}.json' if dg else 'config.json'
    filepath = os.path.join(sub_directory, filename)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
        
        
if __name__ == '__main__':
    args = parse_arguments()

    config_template_file_path = 'generate_utils/config_template.json'
    with open(config_template_file_path, 'r') as f:
        config = json.load(f)

    generate_configs(args.training_modes, args.nb_clients, args.datasets, args.data_sample_techniques, args.global_epochs, args.fractions, args.digits, args.dp_modes, args.sigmas, config)