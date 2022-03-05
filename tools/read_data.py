import cv2
from PIL import Image
import numpy as np

p = './DRIVE/training/images/21_training.tif'
img = np.asarray(Image.open(p))
img = np.stack((img[:, :, 1], img[:, :, 1], img[:, :, 1]), axis=-1)
print(img.shape)