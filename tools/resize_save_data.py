import os
from PIL import Image
import cv2
import numpy as np
import tqdm
import sys

dataset = sys.argv[1]
p_new = '{}_new'.format(dataset)
train_imgpath = './{}/training/images'.format(dataset)
train_gtpath = './{}/training/1st_manual'.format(dataset)
assert os.path.exists(p_new) is False
os.mkdir(p_new)
os.mkdir(os.path.join(p_new, 'big_img'))
os.mkdir(os.path.join(p_new, 'big_gt'))
train_imgs = os.listdir(train_imgpath)
train_gts = os.listdir(train_gtpath)
for i in tqdm.tqdm(range(len(train_imgs))):
    img = Image.open(os.path.join(train_imgpath, train_imgs[i]))
    shape = img.size
    img_ = img.resize((shape[0]*5, shape[1]*5))
    img_.save(os.path.join(os.path.join(p_new, 'big_img', train_imgs[i])))

for i in tqdm.tqdm(range(len(train_gts))):
    img = Image.open(os.path.join(train_gtpath, train_gts[i]))
    shape = img.size
    img_ = img.resize((shape[0]*5, shape[1]*5))
    img_.save(os.path.join(os.path.join(p_new, 'big_gt', train_gts[i])))

