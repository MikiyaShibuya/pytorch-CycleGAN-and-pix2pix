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
        fir_tensor = torch.tensor(fir_image).type(torch.float32)
        rgb_tensor = torch.tensor(rgb_image.transpose(2,0,1)).type(torch.float32)
        return fir_tensor, rgb_tensor

class RandomHorizontalFlip(object):
    def __call__(self, fir_image, rgb_image):
        if np.random.rand() < 0.5:
            return np.fliplr(fir_image), np.fliplr(rgb_image)
        else:
            return fir_image, rgb_image

class RandomCrop(object):
    # crop_sizeのランダムな位置で正方形クロップを行う
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, fir_image, rgb_image):
        crop_size = self.crop_size
        x = np.random.randint(fir_image.shape[1]-crop_size+1)
        y = np.random.randint(fir_image.shape[0]-crop_size+1)
        return fir_image[y:y+crop_size, x:x+crop_size], rgb_image[y:y+crop_size, x:x+crop_size, :]

class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, fir_image, rgb_image):
        crop_size = self.crop_size
        x = (fir_image.shape[1] - crop_size)//2
        y = (fir_image.shape[0] - crop_size)//2
        return fir_image[y:y + crop_size, x:x + crop_size], rgb_image[y:y + crop_size, x:x + crop_size, :]


class Scale(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, fir, rgb):
        fir_image = transform.resize(fir, (self.size, self.size), mode='reflect', anti_aliasing=True)
        rgb_image = transform.resize(rgb, (self.size, self.size), mode='reflect', anti_aliasing=True)
        return fir_image, rgb_image

class Rotate(object):
    def __init__(self, max_degree):
        self.max_degree = max_degree
    def __call__(self, fir, rgb):
        angle = np.random.uniform(-self.max_degree, self.max_degree)
        size_bef = np.array(fir.shape)
        fir_image = transform.rotate(fir, angle, resize=True)
        rgb_image = transform.rotate(rgb, angle, resize=True)
        size_aft = np.array(fir_image.shape)
        new_size = size_bef ** 2 / size_aft
        pad = np.ceil((size_aft - new_size) / 2) + 1
        h1 = int(pad[0])
        h2 = size_aft[0] - h1
        w1 = int(pad[1])
        w2 = size_aft[1] - w1
        return fir_image[h1:h2, w1:w2], rgb_image[h1:h2, w1:w2, :]
