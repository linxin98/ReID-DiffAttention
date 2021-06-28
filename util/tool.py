import random

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_config(config):
    print('=' * 25)
    for key in config.keys():
        if key == 'DEFAULT':
            continue
        print('config', '[' + key + ']')
        for subkey in config[key].keys():
            print(' ' * 4, subkey + ':', config[key][subkey])
    print('=' * 25)