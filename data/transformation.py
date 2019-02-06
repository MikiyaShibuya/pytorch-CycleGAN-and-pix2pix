import numpy as np
from skimage import transform
import torch

class Compose(object):
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, fir_image, rgb_image):
        bound = ((-20 + 273.15)/0.04, (80 + 273.15)/0.04)
        fir_image = 2*fir_image/(bound[1] - bound[0]) - (bound[1]+bound[0])/(bound[1]-bound[0])
        fir_image = np.clip(fir_image, -1, 1)
        rgb_image = (rgb_image - 127.5)/127.5
        for t in self.transformations:
            fir_image, rgb_image = t(fir_image, rgb_image)

        fir_image = fir_image.reshape(1, fir_image.shape[0], fir_image.shape[1])
        fir_tensor = torch.tensor(fir_image)
        rgb_tensor = torch.tensor(rgb_image.transpose(2,0,1))
        return fir_tensor, rgb_tensor

class RandomHorizontalFlip(object):
    def __call__(self, fir_image, rgb_image):
        if np.random.rand() < 0.5:
            return np.fliplr(fir_image), np.fliplr(rgb_image)
        else:
            return fir_image, rgb_image

class RandomCrop(object):
    # crop_size以上のランダムなサイズで正方形クロップを行う
    def __init__(self, crop_size):
        self.min_crop_size = crop_size
    def __call__(self, fir_image, rgb_image):
        max_crop_size = np.min(fir_image.shape)
        crop_size = int(np.random.uniform(self.min_crop_size, max_crop_size))
        x = np.random.randint(fir_image.shape[1]-crop_size+1)
        y = np.random.randint(fir_image.shape[0]-crop_size+1)
        return fir_image[y:y+crop_size, x:x+crop_size], rgb_image[y:y+crop_size, x:x+crop_size, :]

class Scale(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, fir, rgb):
        fir_image = transform.resize(fir, (self.size, self.size), mode='reflect', anti_aliasing=True).astype(np.float32)
        rgb_image = transform.resize(rgb, (self.size, self.size), mode='reflect', anti_aliasing=True).astype(np.float32)
        return fir_image, rgb_image