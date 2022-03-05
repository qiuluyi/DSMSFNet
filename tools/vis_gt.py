import numpy as np
from PIL import Image
import os
import cv2
import tqdm
import sys
dataset = sys.argv[1]
p = './data_slice_{}/label_train'.format(dataset)
tgt = p+'_vis'
if os.path.exists(tgt) is False:
    os.mkdir(tgt)
imgs=os.listdir(p)

def label_mapping(label_im):
    colorize = np.zeros([2, 3], dtype=np.int64)
    colorize[0, :] = [0, 0, 0]
    colorize[1, :] = [255, 255, 255]

    label = colorize[label_im, :].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label

for i in tqdm.tqdm(range(len(imgs))):
    # print(imgs[i])
    # img = cv2.imread('/home/weikai/Documents/go_phd/exam13/top_mosaic_09cm_area1.tif')
    # img = cv2.imread(p+imgs[i], 0)
    img = np.asarray(Image.open(os.path.join(p, imgs[i])))
    # print(set(img.flatten()))
    # x = np.asarray(img).astype(np.int)
    # x = np.where(x==1,1,0)
    # print(set(x.flatten()))
    img = label_mapping(img)
    cv2.imwrite(os.path.join(tgt, imgs[i]), img)