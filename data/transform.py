import math
import random

from torchvision.transforms import Normalize, Compose, transforms


class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_transform(size, is_train, random_erasing=False):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    # List transform items.
    transform_list = []
    transform_list.append(transforms.Resize(size))
    if is_train:
        transform_list.append(transforms.Pad(10))
        transform_list.append(transforms.RandomCrop(size))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)
    if random_erasing:
        transform_list.append(RandomErasing(probability=0.5))
    # Make up transform.
    transform = Compose(transform_list)
    return transform
