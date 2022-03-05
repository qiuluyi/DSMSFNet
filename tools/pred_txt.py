# -*- coding:utf-8 -*-
import cv2
import csv
import tqdm
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable
import os
import transform as tr
np.seterr(divide='ignore', invalid='ignore')
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from networks.get_net import get_net
from cal_iou import evaluate
from yimage import io
from config import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    # img, label = random_crop(img, label, crop_size)
    transform = transforms.Compose([
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']

def find_new_file(dir):
    if os.path.exists(dir) is False:
        os.mkdir(dir)
        dir = dir

    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None

def label_mapping(label_im):
    # colorize = np.zeros([2, 3], dtype=np.int64)
    colorize = np.array([[0, 0, 0],
                [255, 0, 0],
                [0, 255, 255],
                [0, 255, 0],
                [0, 0, 255],
                [128, 128, 0],
                [192, 128, 128],
                [64, 64, 128],
                [64, 0, 128],
                [64, 64, 0],
                [0, 128, 192],
                [255, 255, 255]
                ])
    # colorize[0, :] = [128, 128, 0]
    # colorize[1, :] = [255, 255, 255]
    label = colorize[label_im, :].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label

def predict(net, im): # 预测结果
    # cm = np.array(colormap).astype('uint8')
    with torch.no_grad():
        if use_gpu:
            im = im.unsqueeze(0).cuda()
        else:
            im = im.unsqueeze(0)
        output = net(im)
        pred = output.max(1)[1].squeeze().cpu().data.numpy()
        pred_ = label_mapping(pred)
    return pred_, pred

def pred_image(net, img_, output, output_gray):

    img = io.read_image(img_, driver = 'GDAL')
    size = img.shape
    # state_dict = torch.load('/home/weikai/chick_f_/pth/fcn-deconv-81.pth')
    #
    image_np = img_transforms(img)
    if use_gpu:
        res, gray = predict(net, Variable(torch.Tensor(image_np)).cuda())
    else:
        res, gray = predict(net, Variable(torch.Tensor(image_np)))
    im1 = Image.fromarray(np.uint8(res))
    im2 = Image.fromarray(np.uint8(gray))
    im1 = im1.resize((size[0], size[1]))
    im2 = im2.resize((size[0], size[1]))
    im1.save(os.path.join(output, img_.split('/')[-1]))
    im2.save(os.path.join(output_gray, img_.split('/')[-1]))

def dig_img(output_gray, src, tgt):
    imgs = os.listdir(output_gray)

    for i in range(len(imgs)):
        print(imgs[i])
        img1 = cv2.imread(os.path.join(output_gray, imgs[i]), 0)
        img2 = cv2.imread(os.path.join(src, imgs[i]))
        size = list(np.shape(img1))
        img3 = np.zeros((size[0], size[1], 3))

        for j in range(size[0]):
            for k in range(size[1]):
                if img1[j][k]==1:
                    img3[j][k]=img2[j][k]
        cv2.imwrite(os.path.join(tgt, imgs[i]), img3)


def test_my(input_bands, model_name, model_dir, img_size, num_class):
    net = get_net(model_name, input_bands, num_class, img_size)
    imgs = os.listdir(val_path)
    if os.path.exists(output):
        shutil.rmtree(output)
        os.mkdir(output)
        # print('delete sucessed')
    else:
        os.mkdir(output)
    if os.path.exists(output_gray):
        shutil.rmtree(output_gray)
        os.mkdir(output_gray)
        # print('delete sucessed')
    else:
        os.mkdir(output_gray)
    state_dict = torch.load(find_new_file(model_dir))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    if use_gpu:
        net.cuda()
    net.eval()
    for i in tqdm.tqdm(range(len(imgs))):
        pred_image(net, os.path.join(val_path, imgs[i]), output, output_gray)
    iou, acc = evaluate(output_gray, val_gt, num_class)
    return iou, acc