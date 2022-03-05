import numpy as np
import cv2
from PIL import Image
import os
from collections import Counter
def get_weights(weight_path):
    weights = []
    ss = []
    imgs = os.listdir(weight_path)
    for i in range(len(imgs)):
        p = os.path.join(weight_path, imgs[i])
        img = np.asarray(Image.open(p).convert('L'))
        img = img.flatten()
        ss+=list(img)
    ll = len(ss)
    dd = Counter(ss)
    for x in dd.keys():
        weights.append((ll - dd[x]) * 1.00 / ll)
        dd[x] = [(ll - dd[x]) * 1.00 / ll]
    print(dd)
    print(weights)
    return weights


# x = get_weights()