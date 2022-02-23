import copy
import os
from re import template
import time
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append("")
from optimizer import lambda_calculator
from model import resnet50, classifier, diff_attention, agw, bag_tricks
from metric import cmc_map, re_ranking
from loss import id_loss, triplet_loss, center_loss, circle_loss, reg_loss
from data import transform, dataset, sampler
from util import config_parser, logger, tool, averager

if __name__ == '__main__':
    # 0 introduction
    print('Person Re-Identification')
    print('val')

    # 1 config and tools
    # 1.1 Get config.
    config = config_parser.get_config(sys.argv)
    config_parser.print_config(config)
    # 1.2 Get logger.
    logger = logger.get_logger()
    logger.info('Finishing program initialization.')
    # 1.3 Set device.
    if config['basic']['device'] == 'CUDA':
        os.environ['CUDA_VISIBLE_DEVICES'] = config['basic']['gpu_id']
    if config['basic']['device'] == 'CUDA' and torch.cuda.is_available():
        use_gpu, device = True, torch.device('cuda:0')
        logger.info('Set GPU: ' + config['basic']['gpu_id'])
    else:
        use_gpu, device = False, torch.device('cpu')
        logger.info('Set cpu as device.')
    # 1.4 Set random seed.
    seed = config['basic'].getint('seed')
    tool.setup_random_seed(seed)

    # 2 model
    model_path = config['model']['path']
    num_class = config['model'].getint('num_class')
    num_feature = config['model'].getint('num_feature')
    bias = config['model'].getboolean('bias')
    in_transform = config['da']['in_transform']
    diff_ratio = config['da'].getint('diff_ratio')
    out_transform = config['da']['out_transform']
    aggregate = config['da'].getboolean('aggregate')
    diff_model_path = config['da']['diff_model_path']
    # 2.1 Get feature model.
    base_model = agw.Baseline()
    if use_gpu:
        base_model = base_model.to(device)
    base_model.load_state_dict(torch.load(model_path))
    logger.info('Base Model: ' + str(tool.get_parameter_number(base_model)))
    # 2.2 Get Diff Attention Module.
    diff_model = diff_attention.DiffAttentionModule(
        num_feature=num_feature, in_transform=in_transform, diff_ratio=diff_ratio, out_transform=out_transform, aggregate=aggregate)
    if use_gpu:
        diff_model = diff_model.to(device)
    diff_model.load_state_dict(torch.load(diff_model_path))
    logger.info('Diff Attention Module: ' +
                str(tool.get_parameter_number(diff_model)))

    # 3 data
    dataset_style = config['dataset']['style']
    dataset_path = config['dataset']['path']
    verbose = config['dataset'].getboolean('verbose')
    height = config['dataset'].getint('height')
    width = config['dataset'].getint('width')
    size = (height, width)
    random_erasing = config['dataset'].getboolean('random_erasing')
    batch_size = config['dataset'].getint('batch_size')
    p = config['dataset'].getint('p')
    k = config['dataset'].getint('k')
    num_workers = config['dataset'].getint('num_workers')
    pin_memory = config['dataset'].getboolean('pin_memory')
    dataset_norm = config['dataset'].getboolean('norm')
    # 3.1 Get query set.
    query_path = os.path.join(dataset_path, 'query')
    query_transform = transform.get_transform(size=size, is_train=False)
    query_dataset = dataset.ImageDataset(
        style=dataset_style, path=query_path, transform=query_transform, name='Image Query', verbose=verbose)
    query_loader = DataLoader(dataset=query_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    # 3.2 Get gallery set.
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')
    gallery_transform = transform.get_transform(size=size, is_train=False)
    gallery_dataset = dataset.ImageDataset(
        style=dataset_style, path=gallery_path, transform=gallery_transform, name='Image Gallery', verbose=verbose)
    gallery_loader = DataLoader(dataset=gallery_dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)

    # 4 metric
    # 4.1 Get CMC and mAP metric.
    cmc_map_function = cmc_map.cmc_map

    # 5 eval
    val_norm = config['val'].getboolean('norm')
    re_rank = config['val'].getboolean('re_rank')
    minp = config['val'].getboolean('minp')
    base_model.eval()
    diff_model.eval()
    val_start = time.time()
    with torch.no_grad():
        # Get query feature.
        logger.info('Load query data.')
        query_features = []
        query_pids = []
        query_camids = []
        for query_batch, (query_image, _, pids, camids) in enumerate(query_loader):
            if use_gpu:
                query_image = query_image.to(device)
            query_feature = base_model(query_image)
            # if val_norm:
            #     query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1)
            query_features.append(query_feature)
            query_pids.extend(pids)
            query_camids.extend(camids)
        # Get gallery feature.
        logger.info('Load gallery data.')
        gallery_features = []
        gallery_pids = []
        gallery_camids = []
        for gallery_batch, (gallery_image, _, pids, camids) in enumerate(gallery_loader):
            if use_gpu:
                gallery_image = gallery_image.to(device)
            gallery_feature = base_model(gallery_image)
            # if val_norm:
            #     gallery_feature = torch.nn.functional.normalize(gallery_feature, p=2, dim=1)
            gallery_features.append(gallery_feature)
            gallery_pids.extend(pids)
            gallery_camids.extend(camids)
        if not re_rank:
            # Calculate distance matrix.
            logger.info('Make up distance matrix.')
            distance_matrix = []
            for query_feature in query_features:
                distance = []
                for gallery_feature in gallery_features:
                    m, n = query_feature.shape[0], gallery_feature.shape[0]
                    val_template1, val_template2 = tool.get_templates(
                        m, n, mode='val')
                    new_query_feature = query_feature[val_template1, :]
                    new_gallery_feature = gallery_feature[val_template2, :]
                    new_query_feature, new_gallery_feature = diff_model(
                        new_query_feature, new_gallery_feature, keep_dim=True)
                    if val_norm:
                        new_query_feature = torch.nn.functional.normalize(
                            new_query_feature, p=2, dim=1)
                        new_gallery_feature = torch.nn.functional.normalize(
                            new_gallery_feature, p=2, dim=1)
                    matrix = torch.nn.functional.pairwise_distance(
                        new_query_feature, new_gallery_feature)
                    matrix = matrix.reshape((m, n))
                    distance.append(matrix)
                distance = torch.cat(distance, dim=1)
                distance_matrix.append(distance)
            distance_matrix = torch.cat(distance_matrix, dim=0)
            distance_matrix = distance_matrix.detach().cpu().numpy()
        # Re-ranking.
        else:
            # query_feature = torch.cat(query_features, dim=0)
            # gallery_feature = torch.cat(gallery_features, dim=0)
            # distance_matrix = re_ranking.re_ranking(
            #     query_feature, gallery_feature)
            logger.info('Make up distance matrix.')
            features = query_features + gallery_features
            features1 = copy.deepcopy(features)
            features2 = copy.deepcopy(features)
            distance_matrix = []
            for feature1 in features1:
                distance = []
                for feature2 in features2:
                    m, n = feature1.shape[0], feature2.shape[0]
                    val_template1, val_template2 = tool.get_templates(
                        m, n, mode='val')
                    new_feature1 = feature1[val_template1, :]
                    new_feature2 = feature2[val_template2, :]
                    new_feature1, new_feature2 = diff_model(
                        new_feature1, new_feature2, keep_dim=True)
                    if val_norm:
                        new_feature1 = torch.nn.functional.normalize(
                            new_feature1, p=2, dim=1)
                        new_feature2 = torch.nn.functional.normalize(
                            new_feature2, p=2, dim=1)
                    matrix = torch.nn.functional.pairwise_distance(
                        new_feature1, new_feature2)
                    matrix = matrix.reshape((m, n))
                    matrix = matrix.detach().cpu().numpy()
                    distance.append(matrix)
                distance = np.concatenate(distance, axis=1)
                distance_matrix.append(distance)
            distance_matrix = np.concatenate(distance_matrix, axis=0)
            logger.info('Re-ranking.')
            query_feature = torch.cat(query_features, dim=0)
            gallery_feature = torch.cat(gallery_features, dim=0)
            distance_matrix = re_ranking.re_ranking(
                query_feature, gallery_feature, local_distmat=distance_matrix, only_local=True)
        # Compute CMC and mAP.
        if minp:
            logger.info('Compute CMC, mAP and mINP.')
            cmc, mAP, mINP = cmc_map_function(
                distance_matrix, query_pids, gallery_pids, query_camids, gallery_camids, minp=minp)
            logger.info("CMC curve, Rank-{}: {:.1%}".format(1, cmc[0]))
            logger.info("mAP: {:.1%}".format(mAP))
            logger.info("mINP: {:.1%}".format(mINP))
        else:
            logger.info('Compute CMC and mAP.')
            cmc, mAP = cmc_map_function(
                distance_matrix, query_pids, gallery_pids, query_camids, gallery_camids, minp=minp)
            logger.info("CMC curve, Rank-{}: {:.1%}".format(1, cmc[0]))
            logger.info("mAP: {:.1%}".format(mAP))
        val_end = time.time()
        logger.info('Val time taken: ' + time.strftime("%H:%M:%S",
                                                       time.gmtime(val_end - val_start)))
