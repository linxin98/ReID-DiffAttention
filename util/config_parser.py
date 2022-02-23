import argparse
import configparser
import sys


def get_config(argv):
    # 1 Get arguments.
    parser = argparse.ArgumentParser(
        description='Person Re-Identification Framework')
    # 1.1 Set argument items.
    parser.add_argument('--config', '-c', type=str,
                        default='config/default.ini', help='config file path')
    parser.add_argument('--basic_gpu_id', '-gpu', type=str,
                        help='basic.gpu_id')
    parser.add_argument('--dataset_path', '-dp', type=str,
                        help='dataset.path')
    parser.add_argument('--model_path', '-mp', type=str,
                        help='dataset.path')
    # 1.2 Parse arguments.
    args = parser.parse_args()

    # 2 Load config from config file.
    config_parser = configparser.ConfigParser()
    # 2.1 Load original config file.
    config_parser.read(args.config)
    # 2.2 Set config according to arguments.
    if args.basic_gpu_id is not None:
        config_parser.set('basic', 'gpu_id', args.basic_gpu_id)
    if args.dataset_path is not None:
        config_parser.set('dataset', 'path', args.dataset_path)
    if args.model_path is not None:
        config_parser.set('model', 'path', args.model_path)
    # 3 Return final config.
    return config_parser


def print_config(config_parser):
    print('=' * 25)
    for key in config_parser.keys():
        if key == 'DEFAULT':
            continue
        print('[' + key + ']')
        for sub_key in config_parser[key].keys():
            print(' ' * 4, sub_key + ':', config_parser[key][sub_key])
    print('=' * 25)


if __name__ == '__main__':
    config = get_config(sys.argv)
    print(config['dataset']['path'])
