import torch, cv2
from PIL import Image
import numpy.random as random
import numpy as np
from tools import utils
import math
from torchvision import transforms
import torchvision.transforms.functional as tf

class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        size: expected output size of each image
    """
    def __init__(self, size, flagvals=None):
        self.size = (size, size)
        self.flagvals = flagvals

    def __call__(self, sample):


        elems = list(sample.keys())

        for elem in elems:
            if self.flagvals is None:
                sample[elem] = utils.fixed_resize(sample[elem], self.size)
            else:
                sample[elem] = utils.fixed_resize(sample[elem], self.size,
                                                  flagval=self.flagvals[elem])

        return sample

    def __str__(self):
        return 'FixedResize: '+str(self.size)


# NDSM Normalize
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] /= 255.0
        sample['image'] -= self.mean
        sample['image'] /= self.std

        return sample


class Normalize_pretrained(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] /= 255.0
        sample['image'] -= self.mean
        sample['image'] /= self.std

        sample['gt'] /= 255.0
        sample['gt'] -= self.mean
        sample['gt'] /= self.std

        return sample

class RandomResizedCrop(object):
    """Crop the given Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.

    Args:
        size: expected output size of each image
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    # def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), flagvals=None):
    def __init__(self, size, scale=(0.08, 1.0), ratio=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0], flagvals=None):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.flagvals = flagvals

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarry): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            # aspect_ratio = random.uniform(*ratio)
            aspect_ratio = random.choice(ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w
            if w < img.shape[1] and h < img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        i, j, h, w = self.get_params(sample['image'], self.scale, self.ratio)
        elems = list(sample.keys())

        for elem in elems:
            if sample[elem].ndim == 2:
                sample[elem] = sample[elem][i:i + h, j:j + w]
            else:
                sample[elem] = sample[elem][i:i + h, j:j + w, :]
            if self.flagvals is None:
                sample[elem] = utils.fixed_resize(sample[elem], self.size)
            else:
                sample[elem] = utils.fixed_resize(sample[elem], self.size,
                                                  flagval=self.flagvals[elem])

        return sample

    def __str__(self):
        return 'RandomResizedCrop: (size={}, scale={}, ratio={}.'.format(str(self.size),
                                                        str(self.scale), str(self.ratio))

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        rots (tuple): (minimum, maximum) rotation angle
        scales (tuple): (minimum, maximum) scale
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2

        sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1


        for elem in sample.keys():

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine

            M = cv2.getRotationMatrix2D(center, rot, sc)

            if tmp.ndim == 2 or 'gt' in elem:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample
    def __str__(self):
        return 'RandomHorizontalFlip'

class GaussianBlur(object):
    def __call__(self, sample):
        sample['image'] = cv2.GaussianBlur(sample['image'], (5, 5), 0)
        return sample
    def __str__(self):
        return 'GaussianBlur'

class morphologyEx(object):
    def __call__(self, sample):
        kernel = np.ones((5, 5), np.uint8)
        img = sample['image']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        sample['image'] = blackhat
        return sample
    def __str__(self):
        return 'morphologyEx'

class CLAHE(object):
    def __call__(self, sample):
        image = sample['image']
        # b, g, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
        image = clahe.apply(image)

        sample['image'] = image
        return sample
    def __str__(self):
        return 'CLAHE'

class homomorphic_filter(object):
    def __call__(self, sample):
        d0 = 10
        r1 = 0.5
        rh = 2
        c = 4
        h = 2.0
        l = 0.5
        src = sample['image']
        gray = src.copy()
        if len(src.shape) > 2:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)
        rows, cols = gray.shape

        gray_fft = np.fft.fft2(gray)
        gray_fftshift = np.fft.fftshift(gray_fft)
        dst_fftshift = np.zeros_like(gray_fftshift)
        M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
        D = np.sqrt(M ** 2 + N ** 2)
        Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
        dst_fftshift = Z * gray_fftshift
        dst_fftshift = (h - l) * dst_fftshift + l
        dst_ifftshift = np.fft.ifftshift(dst_fftshift)
        dst_ifft = np.fft.ifft2(dst_ifftshift)
        dst = np.real(dst_ifft)
        dst = np.uint8(np.clip(dst, 0, 255))

        sample['image'] = dst
        return sample
    def __str__(self):
        return 'homomorphic_filter'

class adjustBrightness(object):
    def __call__(self, sample):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据的对比度
        image = Image.fromarray(sample['image'].astype(np.int))
        image = tf.adjust_brightness(image, factor)
        sample['image'] = np.asarray(image)
        return sample

class adjustSaturation(object):
    def __call__(self, sample):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据的对比度
        image = Image.fromarray(sample['image'].astype(np.int))
        image = tf.adjust_saturation(image, factor)
        sample['image'] = np.asarray(image)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem].astype(np.float32)

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp).float()

        return sample

    def __str__(self):
        return 'ToTensor'
