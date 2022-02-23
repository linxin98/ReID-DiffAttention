import os
import re
from collections import defaultdict

import torch
from PIL import Image
from torch.functional import _return_inverse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataset import T


class ImageDataset(Dataset):
    def __init__(self, style, path, transform, name, verbose=False):
        super(ImageDataset, self).__init__()
        # dataset parameters
        self.style = style
        self.path = path
        self.transform = transform
        self.name = name
        # dataset variables
        self.length = 0
        self.images = []
        self.pids = []
        self.camids = []
        self.labels = []
        # Preprocess dataset.
        self.process_dataset()
        if verbose:
            self.summary_dataset()

    def __len__(self):
        return self.length

    def process_item(self, item, style):
        if style == 'market':
            pattern = re.compile(r'([-\d]+)_c(\d)')
        # Check item is a file.
        allow_type = ['jpg', 'png']
        if not item[-3:] in allow_type:
            return None
        pid, camid = map(int, pattern.search(item).groups())
        if pid == -1 or pid == 0:
            return None
        return pid, camid

    def process_dataset(self):
        # Empty dataset.
        self.length = 0
        self.images = []
        self.pids = []
        self.camids = []
        self.labels = []
        # Load folder.
        files = os.listdir(self.path)
        files.sort()
        for file in files:
            results = self.process_item(file, self.style)
            # Add item into dataset.
            if results is not None:
                self.length += 1
                self.images.append(file)
                self.pids.append(results[0])
                self.camids.append(results[1])
        # Create label for classifier.
        _, labels = np.unique(self.pids, return_inverse=True)
        self.set_labels(labels)

    def summary_dataset(self):
        print('=' * 25)
        print("Image Dataset Summary:", self.name)
        print('#pid:   {:d}'.format(len(np.unique(self.pids))))
        print('#image: {:d}'.format(self.length))
        print('=' * 25)

    def set_labels(self, new_labels):
        self.labels = new_labels

    def __getitem__(self, index):
        file = self.images[index]
        pid = self.pids[index]
        camid = self.camids[index]
        label = self.labels[index]
        # Load image.
        file_path = os.path.join(self.path, file)
        image = Image.open(file_path)
        # Convert image to tenser.
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, pid, camid


class FeatureDataset(Dataset):
    def __init__(self, origin_dataset, model, device, batch_size, norm=False, num_workers=4, pin_memory=False, verbose=False):
        super(FeatureDataset, self).__init__()
        self.origin_dataset = origin_dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.norm = norm
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose
        # Get statistics from origin_dataset.
        self.style = self.origin_dataset.style
        self.name = self.origin_dataset.name
        self.length = self.origin_dataset.length
        self.images = self.origin_dataset.images
        self.pids = self.origin_dataset.pids
        self.camids = self.origin_dataset.camids
        self.labels = self.origin_dataset.labels
        # dataset variables
        self.features = []
        # Preprocess dataset.
        self.detect_feature()
        if verbose:
            self.summary_dataset()

    def __len__(self):
        return self.length

    def detect_feature(self):
        # Empty dataset.
        self.features = []
        # Detect feature.
        dataloader = DataLoader(dataset=self.origin_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=self.pin_memory)
        print('Detect feature...')
        with torch.no_grad():
            self.model.eval()
            batch = 0
            for images, _, _, _ in dataloader:
                batch += 1
                if batch % 20 == 0:
                    print('Batch:{}...'.format(batch))
                images = images.to(self.device)
                features = self.model(images)
                if self.norm:
                    features = torch.nn.functional.normalize(
                        features, p=2, dim=1)
                features = features.detach().cpu()
                for i in range(features.shape[0]):
                    self.features.append(features[i])
        print('Finish detecting feature.')

    def summary_dataset(self):
        print('=' * 25)
        print("Feature Dataset Summary:", self.name)
        print('#pid:   {:d}'.format(len(np.unique(self.pids))))
        print('#image: {:d}'.format(self.length))
        print('=' * 25)

    def set_labels(self, new_labels):
        self.labels = new_labels

    def __getitem__(self, index):
        feature = self.features[index]
        pid = self.pids[index]
        camid = self.camids[index]
        label = self.labels[index]
        return feature, label, pid, camid


if __name__ == '__main__':
    style = 'market'
    path = '../dataset/Market-1501/bounding_box_train'
    dateset = ImageDataset(style, path, None, 'train', verbose=True)
    for i in range(100, 120):
        print(dateset[i])
