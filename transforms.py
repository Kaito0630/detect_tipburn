import random
from PIL import Image
import torchvision


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):
        width, height = image.size
        image = image.resize((self.width, self.height), Image.ANTIALIAS)
        target['boxes'][:, [0, 2]] *= (self.width / width)
        target['boxes'][:, [1, 3]] *= (self.height / height)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __call__(self, image, target):
        if random.random() < 0.5:
            image = image.flip(-1)
            height, width = image.shape[-2:]
            target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target


class RandomRotate(object):
    def __call__(self, image, target):
        r = random.random()
        r = 0.5
        if r < 0.25:
            # rotate 270 degrees
            image = image.rot90(3, [1, 2])
            height, width = image.shape[-2:]
            boxes = target['boxes'].clone()
            target['boxes'][:, [0, 2]] = width - boxes[:, [3, 1]]
            target['boxes'][:, [1, 3]] = boxes[:, [0, 2]]
        elif r < 0.5:
            # rotate 180 degrees
            image = image.rot90(2, [1, 2])
            height, width = image.shape[-2:]
            boxes = target['boxes'].clone()
            target['boxes'][:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'][:, [1, 3]] = height - boxes[:, [3, 1]]
        elif r < 0.75:
            # rotate 90 degrees
            image = image.rot90(1, [1, 2])
            height, width = image.shape[-2:]
            boxes = target['boxes'].clone()
            target['boxes'][:, [0, 2]] = boxes[:, [1, 3]]
            target['boxes'][:, [1, 3]] = height - boxes[:, [2, 0]]
        return image, target
