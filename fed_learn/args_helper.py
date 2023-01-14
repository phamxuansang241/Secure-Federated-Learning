import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--config-path',
                        help='Path to the config file', type=str,
                        required=True)
                    
    args = parser.parse_args()

    return args
