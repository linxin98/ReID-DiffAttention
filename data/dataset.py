import os
import re
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageMarket1501Dataset(Dataset):
    def __init__(self, path, transform, name):
        super(ImageMarket1501Dataset, self).__init__()
        self.path = path
        self.transform = transform
        self.name = name
        self.dataset = []
        self.pid_index = defaultdict(list)
        self.length = 0
        self.pids = []
        # Preprocess dataset.
        self.process_folder()
        self.summary()

    def __len__(self):
        return self.length

    def process_folder(self):
        files = os.listdir(self.path)
        files.sort()
        pattern = re.compile(r'([-\d]+)_c(\d)')
        # Empty dataset.
        self.dataset = []
        self.pid_index = defaultdict(list)
        item_index = 0
        for file in files:
            if not file[-3:] == 'jpg':
                continue
            pid, camid = map(int, pattern.search(file).groups())
            if pid == -1 or pid == 0:
                continue
            # First, add item into dataset.
            self.dataset.append((file, pid, camid))
            # Then, count pid.
            self.pid_index[pid].append(item_index)
            # Finally, update item_index.
            item_index += 1
        self.length = item_index
        self.pids = list(self.pid_index.keys())
        self.pids.sort()

    def summary(self):
        print('-' * 25)
        print("Image Dataset Summary:", self.name)
        print('#pids: {:4d}'.format(len(self.pids)))
        print('#images: {:6d}'.format(self.length))
        print('-' * 25)

    def __getitem__(self, item):
        (file, pid, camid) = self.dataset[item]
        class_index = self.pids.index(pid)
        file_path = os.path.join(self.path, file)
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, class_index, pid, camid


class FeatureFromImageDataset(Dataset):
    def __init__(self, origin_dataset, model, device, batch_size, norm=None, num_workers=0, pin_memory=False):
        super(FeatureFromImageDataset, self).__init__()
        self.origin_dataset = origin_dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.norm = norm
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = []
        # Get origin_dataset statistics.
        self.name = self.origin_dataset.name
        self.pid_index = self.origin_dataset.pid_index
        self.length = self.origin_dataset.length
        self.pids = self.origin_dataset.pids
        # Preprocess dataset.
        self.detect_feature()
        self.summary()

    def __len__(self):
        return self.length

    def detect_feature(self):
        dataloader = DataLoader(dataset=self.origin_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory)
        print('Detect feature...')
        with torch.no_grad():
            self.model.eval()
            for images, _, pids, camids in dataloader:
                images = images.to(self.device)
                features = self.model(images)
                if self.norm is not None:
                    features = torch.nn.functional.normalize(features, p=self.norm, dim=1)
                features = features.detach().cpu()
                for i in range(self.batch_size):
                    self.dataset.append((features[i], pids[i], camids[i]))
        print('Finished.')

    def summary(self):
        print('-' * 25)
        print("Feature Dataset Summary:", self.name)
        print('#pids: {:4d}'.format(len(self.pids)))
        print('#images: {:6d}'.format(self.length))
        print('-' * 25)

    def __getitem__(self, item):
        (feature, pid, camid) = self.dataset[item]
        class_index = self.pids.index(pid)
        return feature, class_index, pid, camid
