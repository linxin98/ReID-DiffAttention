import os
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
from model import bag_tricks, classifier, diff_attention
from metric import cmc_map, re_ranking
from loss import id_loss, triplet_loss, center_loss, circle_loss, reg_loss
from data import transform, dataset, sampler
from util import config_parser, logger, tool, averager


if __name__ == '__main__':
    # 0 introduction
    print('Person Re-Identification')
    print('supervised BagTricks DAOff')

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
    base_path = config['model']['path']
    num_class = config['model'].getint('num_class')
    num_feature = config['model'].getint('num_feature')
    in_transform = config['da']['in_transform']
    diff_ratio = config['da'].getint('diff_ratio')
    out_transform = config['da']['out_transform']
    aggregate = config['da'].getboolean('aggregate')
    # 2.1 Get feature model.
    base_model = bag_tricks.Baseline()
    if use_gpu:
        base_model = base_model.to(device)
    logger.info('Base Model: ' + str(tool.get_parameter_number(base_model)))
    base_model.load_state_dict(torch.load(base_path))
    # 2.2 Get Diff Attention Module.
    diff_model = diff_attention.DiffAttentionModule(
        num_feature=num_feature, in_transform=in_transform, diff_ratio=diff_ratio, out_transform=out_transform, aggregate=aggregate)
    if use_gpu:
        diff_model = diff_model.to(device)
    logger.info('Diff Attention Module: ' +
                str(tool.get_parameter_number(diff_model)))

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
    # 3.1 Get train set.
    train_path = os.path.join(dataset_path, 'bounding_box_train')
    train_transform = transform.get_transform(
        size=size, is_train=True, random_erasing=random_erasing)
    train_image_dataset = dataset.ImageDataset(
        style=dataset_style, path=train_path, transform=train_transform, name='Train', verbose=verbose)
    train_dataset = dataset.FeatureDataset(origin_dataset=train_image_dataset, model=base_model, device=device,
                                           batch_size=batch_size, norm=dataset_norm, num_workers=num_workers, pin_memory=pin_memory)
    if p is not None and k is not None and p * k == batch_size:
        # Use triplet sampler.
        sampler = sampler.TripletSampler(
            labels=train_dataset.labels, batch_size=batch_size, p=p, k=k)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    # 3.2 Get query set.
    query_path = os.path.join(dataset_path, 'query')
    query_transform = transform.get_transform(size=size, is_train=False)
    query_image_dataset = dataset.ImageDataset(
        style=dataset_style, path=query_path, transform=query_transform, name='Query', verbose=verbose)
    query_dataset = dataset.FeatureDataset(origin_dataset=query_image_dataset, model=base_model, device=device,
                                           batch_size=batch_size, norm=dataset_norm, num_workers=num_workers, pin_memory=pin_memory)
    query_loader = DataLoader(dataset=query_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    # 3.3 Get gallery set.
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')
    gallery_transform = transform.get_transform(size=size, is_train=False)
    gallery_image_dataset = dataset.ImageDataset(
        style=dataset_style, path=gallery_path, transform=gallery_transform, name='Gallery', verbose=verbose)
    gallery_dataset = dataset.FeatureDataset(origin_dataset=gallery_image_dataset, model=base_model, device=device,
                                             batch_size=batch_size, norm=dataset_norm, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = DataLoader(dataset=gallery_dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)

    # 4 loss
    id_loss_weight = config['loss'].getfloat('id_loss_weight')
    smooth = config['loss'].getboolean('label_smooth')
    triplet_loss_weight = config['loss'].getfloat('triplet_loss_weight')
    margin = config['loss'].getfloat('margin')
    soft_margin = config['loss'].getboolean('soft_margin')
    center_loss_weight = config['loss'].getfloat('center_loss_weight')
    circle_loss_weight = config['loss'].getfloat('circle_loss_weight')
    reg_loss_weight = config['loss'].getfloat('reg_loss_weight')
    reg_loss_p = config['loss'].getint('reg_loss_p')
    # 4.1 Get triplet loss.
    triplet_loss_function = triplet_loss.TripletLoss(margin=margin, batch_size=batch_size, p=p, k=k,
                                                     soft_margin=soft_margin)
    # 4.2 Get regularization loss.
    reg_loss_function = reg_loss.Regularization(p=reg_loss_p)

    # 5 optimizer
    init_lr = config['optimizer'].getfloat('init_lr')
    center_loss_lr = config['optimizer'].getfloat('center_loss_lr')
    milestone = config['optimizer']['milestone']
    milestones = [] if milestone == '' else [
        int(x) for x in milestone.split(',')]
    weight_decay = config['optimizer'].getfloat('weight_decay')
    warmup = config['optimizer'].getboolean('warmup')
    # 5.1 Get Diff Attention Module optimizer.
    diff_optimizer = Adam(diff_model.parameters(),
                          lr=init_lr, weight_decay=weight_decay)
    diff_lambda_function = lambda_calculator.get_lambda_calculator(
        milestones=milestones, warmup=warmup)
    diff_scheduler = LambdaLR(diff_optimizer, diff_lambda_function)

    # 6 metric
    # 6.1 Get CMC and mAP metric.
    cmc_map_function = cmc_map.cmc_map
    # 6.2 Get averagers.
    triplet_loss_averager = averager.Averager()
    reg_loss_averager = averager.Averager()
    all_loss_averager = averager.Averager()

    # 7 train and eval
    epochs = config['train'].getint('epochs')
    val_per_epochs = config['train'].getint('val_per_epochs')
    log_iteration = config['train'].getint('log_iteration')
    save = config['train'].getboolean('save')
    save_per_epochs = config['train'].getint('save_per_epochs')
    save_path = config['train']['save_path']
    save_path = os.path.join(
        save_path, time.strftime("%Y%m%d", time.localtime()))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    val_norm = config['val'].getboolean('norm')
    re_ranking = config['val'].getboolean('re_ranking')
    minp = config['val'].getboolean('minp')
    # 7.1 Initialize env.
    # Make up batch templates.
    batch_template1, batch_template2 = tool.get_templates(
        batch_size, batch_size)
    for epoch in range(1, epochs + 1):
        # 7.2 Start epoch.
        # Set model to be trained.
        diff_model.train()
        # Reset averagers.
        triplet_loss_averager.reset()
        reg_loss_averager.reset()
        all_loss_averager.reset()
        # Initialize epoch.
        iteration = 0
        logger.info('Epoch[{}/{}] Epoch start.'.format(epoch, epochs))
        epoch_start = time.time()
        for features, labels, _, _ in train_loader:
            # 7.3 Start iteration.
            iteration += 1
            diff_optimizer.zero_grad()
            # 7.4 Train.
            # 7.4.1 Forward.
            if use_gpu:
                features = features.to(device)
            features1 = features[batch_template1, :]
            features2 = features[batch_template2, :]
            features1, features2 = diff_model(
                features1, features2, keep_dim=True)
            # 7.4.2 Calculate loss.
            # triplet loss
            triplet_loss = triplet_loss_function(
                features1, features2) * triplet_loss_weight
            # # reg loss
            reg_loss = reg_loss_function(diff_model) * reg_loss_weight
            # all loss
            all_loss = triplet_loss + reg_loss
            # 7.4.3 Optimize.
            all_loss.backward()
            diff_optimizer.step()
            # 7.4.4 Log losses and acc.
            triplet_loss_averager.update(triplet_loss.item())
            reg_loss_averager.update(reg_loss.item())
            all_loss_averager.update(all_loss.item())
            # 7.5 End iteration.
            # 7.5.1 Summary iteration.
            if iteration % log_iteration == 0:
                logger.info('Epoch[{}/{}] Iteration[{}] Loss: {:.3f}'
                            .format(epoch, epochs, iteration, all_loss_averager.get_value()))
        # 7.6 End epoch.
        epoch_end = time.time()
        # 7.6.1 Summary epoch.
        logger.info('Epoch[{}/{}] Loss: {:.3f} Base Lr: {:.2e}'.format(
            epoch, epochs, all_loss_averager.get_value(), diff_scheduler.get_last_lr()[0]))
        logger.info('Epoch[{}/{}] Triplet_Loss: {:.3f}'.format(
            epoch, epochs, triplet_loss_averager.get_value()))
        logger.info('Epoch[{}/{}] Reg_Loss: {:.3f}'.format(
            epoch, epochs, reg_loss_averager.get_value()))
        logger.info('Train time taken: ' + time.strftime("%H:%M:%S",
                                                         time.gmtime(epoch_end - epoch_start)))
        # 7.6.2 Change learning rate.
        diff_scheduler.step()
        # 7.7 Eval.
        if epoch % val_per_epochs == 0:
            logger.info('Start validation every {} epochs at epoch: {}'.format(
                val_per_epochs, epoch))
            diff_model.eval()
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
                for gallery_batch, (gallery_feature, _, pids, camids) in enumerate(gallery_loader):
                    if use_gpu:
                        gallery_feature = gallery_feature.to(device)
                    # if val_norm:
                    #     gallery_feature = torch.nn.functional.normalize(gallery_feature, p=2, dim=1)
                    gallery_features.append(gallery_feature)
                    gallery_pids.extend(pids)
                    gallery_camids.extend(camids)
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
                        matrix = matrix.detach().cpu().numpy()
                        distance.append(matrix)
                    distance = np.concatenate(distance, axis=1)
                    distance_matrix.append(distance)
                distance_matrix = np.concatenate(distance_matrix, axis=0)
                # Re-ranking.
                # if re_ranking:
                #     distance_matrix = re_ranking.re_ranking()
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
        # 7.8 Save checkpoint.
        if save:
            if epoch % save_per_epochs == 0:
                logger.info('Save checkpoint every {} epochs at epoch: {}'.format(
                    save_per_epochs, epoch))
                diff_save_name = '[supervised bag daoff]' + time.strftime(
                    "%H%M%S", time.localtime()) + '[diff]' + str(epoch) + '.pth'
                torch.save(diff_model.state_dict(),
                           os.path.join(save_path, diff_save_name))
