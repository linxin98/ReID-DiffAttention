import argparse
import configparser
import copy
import os
import time
import sys
import warnings

warnings.filterwarnings('ignore')

import pynvml
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR

sys.path.append("/content/ReID-framework")
from data import dataloader
from loss import label_smooth, triplet_loss, center_loss, regularization
from metric import cmc_map, averager
from model import resnet, bag_tricks, diff_attention
from optimizer import lambda_calculator
from util import logger, tool

if __name__ == '__main__':
    # 0 introduction
    print('val')

    # 1 config and tools
    # 1.1 Get arguments.
    parser = argparse.ArgumentParser(description='Person Re-Identification')
    parser.add_argument('--config', type=str, default='../configs/default.ini', help='config file')
    args = parser.parse_args()
    # 1.2 Load config from config file.
    config = configparser.ConfigParser()
    config.read(args.config)
    tool.print_config(config)
    # 1.3 Get logger.
    logger = logger.get_logger()
    logger.info('Finishing program initialization.')
    # 1.4 Set device.
    if config['normal']['device'] == 'CUDA' and torch.cuda.is_available():
        use_gpu = True
        device = 'cuda:' + config['normal']['gpu_id']
        torch.backends.cudnn.benchmark = True
    else:
        use_gpu = False
        device = 'cpu'
    logger.info('Set device:' + device)
    device = torch.device(device)
    # 1.5 Set random seed.
    tool.setup_seed(config['normal'].getint('seed'))

    # 2 model
    # models_path is used for loading *.pth.
    base_model_name = config['model']['base_model_name']
    base_model_path = config['model']['base_model_path']
    num_classes = config['model'].getint('num_classes')
    num_feature = config['model'].getint('num_feature')
    diff_model_path = config['model']['diff_model_path']
    in_transform = config['model']['in_transform']
    diff_ratio = config['model'].getint('diff_ratio')
    out_transform = config['model']['out_transform']
    # 2.1 Get feature model.
    if base_model_name == 'bag':
        base_model = bag_tricks.Baseline(num_classes)
    else:
        base_model = resnet.ResNet50(num_classes)
    if use_gpu:
        base_model = base_model.to(device)
    base_model.load_state_dict(torch.load(base_model_path))
    # 2.2 Get diff attention model.
    diff_attention_model = diff_attention.DiffAttentionNet(num_feature, in_transform, diff_ratio, out_transform)
    if use_gpu:
        diff_attention_model = diff_attention_model.to(device)
    diff_attention_model.load_state_dict(torch.load(diff_model_path))

    # 3 data
    dataset_style = config['dataset']['style']
    dataset_path = config['dataset']['path']
    dataset_name = config['dataset']['name']
    height = config['dataset'].getint('height')
    width = config['dataset'].getint('width')
    size = (height, width)
    batch_size = config['dataset'].getint('batch_size')
    num_workers = config['dataset'].getint('num_workers')
    pin_memory = config['dataset'].getboolean('pin_memory')
    norm = config['dataset'].getboolean('norm')
    # 3.1 Get query set.
    query_path = os.path.join(dataset_path, 'query')
    query_name = dataset_name + '_query'
    query_loader = dataloader.get_feature_from_image_dataloader(style=dataset_style, path=query_path, name=query_name,
                                                                image_size=size, model=base_model, device=device,
                                                                batch_size=batch_size, is_train=False,
                                                                num_workers=num_workers, pin_memory=pin_memory,
                                                                norm=norm)
    # 3.2 Get gallery set.
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')
    gallery_name = dataset_name + '_gallery'
    gallery_loader = dataloader.get_feature_from_image_dataloader(style=dataset_style, path=gallery_path,
                                                                  name=gallery_name, image_size=size, model=base_model,
                                                                  device=device, batch_size=batch_size, is_train=False,
                                                                  num_workers=num_workers, pin_memory=pin_memory,
                                                                  norm=norm)

    # 4 loss

    # 5 optimizer

    # 6 metric
    # 6.1 Get CMC and mAP metric.
    cmc_map_function = cmc_map.cmc_map

    # 7 eval
    logger.info('Start validation.')
    torch.cuda.empty_cache()
    base_model.eval()
    diff_attention_model.eval()
    val_start = time.time()
    with torch.no_grad():
        # Get query feature.
        logger.info('Load query data.')
        query_features = []
        query_pids = []
        query_camids = []
        for query_batch, (query_feature, _, pids, camids) in enumerate(query_loader):
            if use_gpu:
                query_feature = query_feature.to(device)
            query_features.append(query_feature)
            query_pids.extend(pids)
            query_camids.extend(camids)
        # Get gallery feature.
        logger.info('Load gallery data.')
        gallery_features = []
        gallery_pids = []
        gallery_camids = []
        for gallery_batch, (gallery_feature, _, pids, camids) in enumerate(gallery_loader):
            if use_gpu:
                gallery_feature = gallery_feature.to(device)
            gallery_features.append(gallery_feature)
            gallery_pids.extend(pids)
            gallery_camids.extend(camids)
        # Calculate distance matrix.
        logger.info('Make up distance matrix.')
        distance_matrix = []
        for query_feature in query_features:
            distance = []
            m = query_feature.shape[0]

            for gallery_feature in gallery_features:
                n = gallery_feature.shape[0]

                val_template1 = []
                val_template2 = []
                for x in range(0, m):
                    for y in range(0, n):
                        val_template1.append(x)
                        val_template2.append(y)

                new_query_feature = query_feature[val_template1, :]
                new_gallery_feature = gallery_feature[val_template2, :]
                new_query_feature, new_gallery_feature = diff_attention_model(new_query_feature,
                                                                              new_gallery_feature,
                                                                              keep_dim=True)
                matrix = torch.nn.functional.pairwise_distance(new_query_feature, new_gallery_feature)
                matrix = matrix.reshape((m, n))
                distance.append(matrix)

            distance = torch.cat(distance, dim=1)
            distance_matrix.append(distance)

        distance_matrix = torch.cat(distance_matrix, dim=0)
        distance_matrix = torch.pow(distance_matrix, 2)
        distance_matrix = distance_matrix.detach().cpu().numpy()
        # Compute CMC and mAP.
        logger.info('Compute CMC and mAP.')
        cmc, mAP = cmc_map_function(distance_matrix, query_pids, gallery_pids, query_camids, gallery_camids)
        for r in [1]:
            logger.info("CMC curve, Rank-{}: {:.1%}".format(r, cmc[r - 1]))
        logger.info("mAP: {:.1%}".format(mAP))
        val_end = time.time()
        logger.info('Val time taken: ' + time.strftime("%H:%M:%S", time.localtime(val_end - val_start)))
    torch.cuda.empty_cache()

