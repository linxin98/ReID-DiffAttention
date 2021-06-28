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
    print('supervised online')

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
    num_classes = config['model'].getint('num_classes')
    num_feature = config['model'].getint('num_feature')
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
    # 2.2 Get diff attention model.
    diff_attention_model = diff_attention.DiffAttentionNet(num_feature, in_transform, diff_ratio, out_transform)
    if use_gpu:
        diff_attention_model = diff_attention_model.to(device)

    # 3 data
    dataset_style = config['dataset']['style']
    dataset_path = config['dataset']['path']
    dataset_name = config['dataset']['name']
    height = config['dataset'].getint('height')
    width = config['dataset'].getint('width')
    size = (height, width)
    batch_size = config['dataset'].getint('batch_size')
    p = config['dataset'].getint('p')
    k = config['dataset'].getint('k')
    num_workers = config['dataset'].getint('num_workers')
    pin_memory = config['dataset'].getboolean('pin_memory')
    norm = config['dataset'].getboolean('norm')
    random_erasing = config['dataset'].getboolean('random_erasing')
    # 3.1 Get train set.
    train_path = os.path.join(dataset_path, 'bounding_box_train')
    train_name = dataset_name + '_train'
    train_loader = dataloader.get_image_dataloader(style=dataset_style, path=train_path, name=train_name,
                                                   image_size=size, batch_size=batch_size, is_train=True,
                                                   num_workers=num_workers, pin_memory=pin_memory, p=p, k=k,
                                                   rea=random_erasing)
    # 3.2 Get query set.
    query_path = os.path.join(dataset_path, 'query')
    query_name = dataset_name + '_query'
    query_loader = dataloader.get_image_dataloader(style=dataset_style, path=query_path, name=query_name,
                                                   image_size=size, batch_size=batch_size, is_train=False,
                                                   num_workers=num_workers, pin_memory=pin_memory)
    # 3.3 Get gallery set.
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')
    gallery_name = dataset_name + '_gallery'
    gallery_loader = dataloader.get_image_dataloader(style=dataset_style, path=gallery_path, name=gallery_name,
                                                     image_size=size, batch_size=batch_size, is_train=False,
                                                     num_workers=num_workers, pin_memory=pin_memory)

    # 4 loss
    id_loss_weight = config['loss'].getfloat('id_loss_weight')
    smooth = config['loss'].getboolean('label_smooth')
    triplet_loss_weight = config['loss'].getfloat('triplet_loss_weight')
    margin = config['loss'].getfloat('margin')
    soft_margin = config['loss'].getboolean('soft_margin')
    center_loss_weight = config['loss'].getfloat('center_loss_weight')
    reg_loss_weight = config['loss'].getfloat('reg_loss_weight')
    reg_loss_p = config['loss'].getint('reg_loss_p')
    # 4.1 Get id loss.
    if smooth:
        id_loss_function = label_smooth.CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=use_gpu, device=device)
    else:
        id_loss_function = nn.CrossEntropyLoss()
    # 4.2 Get triplet loss.
    triplet_loss_function = triplet_loss.TripletLoss(margin=margin, batch_size=batch_size, p=p, k=k,
                                                     soft_margin=soft_margin)
    # 4.3 Get center loss.
    center_loss_function = center_loss.CenterLoss(num_classes=num_classes, feat_dim=num_feature, use_gpu=use_gpu,
                                                  device=device)
    # 4.4 Get regularization loss.
    reg_loss_function = regularization.Regularization(p=reg_loss_p)

    # 5 optimizer
    init_lr = config['optimizer'].getfloat('init_lr')
    center_loss_lr = config['optimizer'].getfloat('center_loss_lr')
    warmup = config['optimizer'].getboolean('warmup')
    # 5.1 Get base model optimizer.
    base_optimizer = Adam(base_model.parameters(), lr=init_lr, weight_decay=0.0005)
    base_lambda_function = lambda_calculator.get_lambda_calculator(milestones=[40, 70], warmup=warmup)
    base_scheduler = LambdaLR(base_optimizer, base_lambda_function)
    # 5.2 Get diff attention model optimizer.
    diff_attention_optimizer = Adam(diff_attention_model.parameters(), lr=init_lr, weight_decay=0.0005)
    diff_attention_lambda_function = lambda_calculator.get_lambda_calculator(milestones=[40, 70], warmup=warmup)
    diff_attention_scheduler = LambdaLR(diff_attention_optimizer, diff_attention_lambda_function)
    # 5.3 Get center loss optimizer.
    center_optimizer = SGD(center_loss_function.parameters(), lr=center_loss_lr, weight_decay=0.0005)
    center_lambda_function = lambda_calculator.get_lambda_calculator(milestones=[40, 70], warmup=False)
    center_scheduler = LambdaLR(center_optimizer, center_lambda_function)

    # 6 metric
    # 6.1 Get CMC and mAP metric.
    cmc_map_function = cmc_map.cmc_map
    # 6.2 Get averagers.
    id_loss_averager = averager.Averager()
    triplet_loss_averager = averager.Averager()
    center_loss_averager = averager.Averager()
    reg_loss_averager = averager.Averager()
    all_loss_averager = averager.Averager()
    acc_averager = averager.Averager()

    # 7 train and eval
    epochs = config['trainer'].getint('epochs')
    val_per_epochs = config['trainer'].getint('val_per_epochs')
    log_iteration = config['trainer'].getint('log_iteration')
    save = config['trainer'].getboolean('save')
    save_per_epochs = config['trainer'].getint('save_per_epochs')
    save_path = config['trainer']['save_path']
    save_path = os.path.join(save_path, time.strftime("%Y%m%d", time.localtime()))
    val_norm = config['trainer'].getboolean('val_norm')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # 7.1 Initialize env.
    torch.cuda.empty_cache()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    template1 = []
    template2 = []
    for x in range(0, batch_size - 1):
        for y in range(x + 1, batch_size):
            template1.append(x)
            template2.append(y)
    for epoch in range(1, epochs + 1):
        # 7.2 Start epoch.
        base_model.train()
        diff_attention_model.train()
        center_loss_function.train()
        id_loss_averager.reset()
        triplet_loss_averager.reset()
        center_loss_averager.reset()
        reg_loss_averager.reset()
        all_loss_averager.reset()
        acc_averager.reset()
        iteration = 0
        logger.info('Epoch[{}/{}] Epoch start.'.format(epoch, epochs))
        epoch_start = time.time()
        for images, class_indexs, _, _ in train_loader:
            # 7.3 Start iteration.
            iteration += 1
            base_optimizer.zero_grad()
            diff_attention_optimizer.zero_grad()
            center_optimizer.zero_grad()
            # 7.4 Train.
            # 7.4.1 Forward.
            if use_gpu:
                images = images.to(device)
                class_indexs = class_indexs.to(device)
            predict_classes, features = base_model(images)
            features1 = features[template1, :]
            features2 = features[template2, :]
            features1, features2 = diff_attention_model(features1, features2, keep_dim=True)
            # 7.4.2 Calculate loss.
            all_loss = torch.tensor(0)
            # id loss
            id_loss = id_loss_function(predict_classes, copy.deepcopy(class_indexs)) * id_loss_weight
            # triplet loss
            distance = nn.functional.pairwise_distance(features1, features2)
            distance_matrix = torch.zeros((batch_size, batch_size))
            index = 0
            for x in range(0, batch_size - 1):
                for y in range(x + 1, batch_size):
                    distance_matrix[x, y] = distance[index]
                    distance_matrix[y, x] = distance[index]
                    index += 1
            triplet_loss = triplet_loss_function(distance_matrix) * triplet_loss_weight
            # center loss
            center_loss = center_loss_function(features, copy.deepcopy(class_indexs)) * center_loss_weight
            # reg loss
            reg_loss = reg_loss_function(base_model) * reg_loss_weight
            # all loss
            all_loss = id_loss + triplet_loss + center_loss + reg_loss
            # 7.4.3 Optimize.
            all_loss.backward()
            base_optimizer.step()
            diff_attention_optimizer.step()
            for param in center_loss_function.parameters():
                param.grad.data *= (1. / center_loss_weight)
            center_optimizer.step()
            # 7.4.4 Log losses and acc.
            acc = (predict_classes.max(1)[1] == class_indexs).float().mean()
            acc_averager.update(acc.item())
            id_loss_averager.update(id_loss.item())
            triplet_loss_averager.update(triplet_loss.item())
            center_loss_averager.update(center_loss.item())
            reg_loss_averager.update(reg_loss.item())
            all_loss_averager.update(all_loss.item())
            # 7.5 End iteration.
            # 7.5.1 Summary iteration.
            if iteration % log_iteration == 0:
                logger.info('Epoch[{}/{}] Iteration[{}] Loss: {:.3f} Acc: {:.3f}'
                            .format(epoch, epochs, iteration, all_loss_averager.get_value(), acc_averager.get_value()))
        # 7.6 End epoch.
        epoch_end = time.time()
        # 7.6.1 Summary epoch.
        logger.info('Epoch[{}/{}] Loss: {:.3f} Acc: {:.3f} Base Lr: {:.2e}'
                    .format(epoch, epochs, all_loss_averager.get_value(), acc_averager.get_value(),
                            base_scheduler.get_last_lr()[0]))
        logger.info('Epoch[{}/{}] ID_Loss: {:.3f}'.format(epoch, epochs, id_loss_averager.get_value()))
        logger.info('Epoch[{}/{}] Triplet_Loss: {:.3f}'.format(epoch, epochs, triplet_loss_averager.get_value()))
        logger.info('Epoch[{}/{}] Center_Loss: {:.3f}'.format(epoch, epochs, center_loss_averager.get_value()))
        logger.info('Epoch[{}/{}] Reg_Loss: {:.3f}'.format(epoch, epochs, reg_loss_averager.get_value()))
        logger.info('Train time taken: ' + time.strftime("%H:%M:%S", time.localtime(epoch_end - epoch_start)))
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info('GPU Memory Used(GB): {:.3f} GB'.format(meminfo.used / 1024 ** 3))
        torch.cuda.empty_cache()
        # 7.6.2 Change learning rate.
        base_scheduler.step()
        diff_attention_scheduler.step()
        center_scheduler.step()
        # 7.7 Eval.
        if epoch % val_per_epochs == 0:
            logger.info('Start validation every {} epochs at epoch: {}'.format(val_per_epochs, epoch))
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
                for query_batch, (query_image, _, pids, camids) in enumerate(query_loader):
                    if use_gpu:
                        query_image = query_image.to(device)
                    query_feature = base_model(query_image)
                    if val_norm:
                        query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1)
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
                    if val_norm:
                        gallery_feature = torch.nn.functional.normalize(gallery_feature, p=2, dim=1)
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

                        template1 = []
                        template2 = []
                        for x in range(0, m):
                            for y in range(0, n):
                                template1.append(x)
                                template2.append(y)

                        new_query_feature = query_feature[template1, :]
                        new_gallery_feature = gallery_feature[template2, :]
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
        # 7.8 Save checkpoint.
        if epoch % save_per_epochs == 0:
            logger.info('Save checkpoint every {} epochs at epoch: {}'.format(save_per_epochs, epoch))
            base_save_name = 'supervise online-' + 'base-' + str(epoch) + '.pth'
            torch.save(base_model.state_dict(), os.path.join(save_path, base_save_name))
            diff_attention_save_name = 'supervise online-' + 'diff-' + str(epoch) + '.pth'
            torch.save(diff_attention_model.state_dict(), os.path.join(save_path, diff_attention_save_name))