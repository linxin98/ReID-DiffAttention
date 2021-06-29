from torch.utils.data import DataLoader

from data.sampler import TripletSampler
from data.dataset import ImageMarket1501Dataset, FeatureFromImageDataset
from data.transform import get_transform


def get_image_dataloader(style, path, name, image_size, batch_size, is_train, num_workers=0, pin_memory=False, p=None,
                         k=None, rea=False):
    transform = get_transform(size=image_size, is_train=is_train, random_erasing=rea)
    dataset = None
    if style == 'market1501':
        dataset = ImageMarket1501Dataset(path=path, transform=transform, name=name)
    if p is not None and k is not None and p * k == batch_size:
        # Use triplet sampler.
        sampler = TripletSampler(pids=dataset.pids, pid_index=dataset.pid_index, batch_size=batch_size, p=p, k=k)
    else:
        sampler = None
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                            pin_memory=pin_memory)
    return dataloader


def get_feature_from_image_dataloader(style, path, name, image_size, model, device, batch_size, is_train, num_workers=0,
                                      pin_memory=False, norm=False, p=None, k=None, rea=None):
    transform = get_transform(size=image_size, is_train=is_train, random_erasing=rea)
    origin_dataset = None
    if style == 'market1501':
        origin_dataset = ImageMarket1501Dataset(path=path, transform=transform, name=name)
    dataset = FeatureFromImageDataset(origin_dataset=origin_dataset, model=model, device=device, batch_size=batch_size,
                                      norm=norm, num_workers=num_workers, pin_memory=pin_memory)
    if p is not None and k is not None and p * k == batch_size:
        # Use triplet sampler.
        sampler = TripletSampler(pids=dataset.pids, pid_index=dataset.pid_index, batch_size=batch_size, p=p, k=k)
    else:
        sampler = None
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                            pin_memory=pin_memory)
    return dataloader


if __name__ == '__main__':
    path = '../../dataset/Market-1501-v15.09.15/Market-1501-v15.09.15/bounding_box_train'
    dataloader = get_image_dataloader('market1501', path, 'Market1501_train', (256, 128), 64, True, p=16, k=4, rea=True)
    print(len(dataloader))
