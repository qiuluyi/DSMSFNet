import os
import shutil
import tqdm
import cv2
import numpy as np
from tools.utils import label_mapping
from config import *
from networks.get_net import get_net
from collections import OrderedDict
import yimage
import tools.transform as tr
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from tools.utils import read_image

def tta_inference(inp,dsm, model, num_classes=8, scales=[1.0], flip=True):
    b, _, h, w = inp.size()
    preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
    for scale in scales:
        size = (int(scale * h), int(scale * w))
        resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True, )
        resized_dsm = F.interpolate(dsm, size=size, mode='bilinear', align_corners=True, )
        pred = model_inference(model,resized_img.to(inp.device),resized_dsm.to(dsm.device), flip)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True, )
        preds += pred

    return preds / (len(scales))


def model_inference(model, image,ndsm, flip=True):
    with torch.no_grad():
        output = model(image,ndsm)
        if flip:
            fimg = image.flip(2)
            fdsm = ndsm.flip(2)
            output += model(fimg,fdsm).flip(2)
            fimg = image.flip(3)
            fdsm = ndsm.flip(3)
            output += model(fimg,fdsm).flip(3)
            return output / 3
        return output

def pred_img(model, image,dsm):
    with torch.no_grad():
        output = model(image,dsm)
    return output


def slide_pred(model, image_path, nsdm_path, num_classes=6, crop_size=512, overlap=256, scales=[1.0], flip=True):
    # torch.Size([2569, 1919, 3])
    scale_image = read_image(image_path).astype(np.float32)
    # torch.Size([3, 2569, 1919])
    # scale_image = np.expand_dims(scale_image, axis=2)
    scale_image = img_transforms(scale_image)
    # 增加一通道
    scale_image = scale_image.unsqueeze(0).cuda()

    # nsdm数据  # torch.Size([1, 3, 2569, 1919])
    # (2569, 1919)
    #(2569, 1919)
    scale_nsdm = read_image(nsdm_path).astype(np.float32)
    # 0 1 2
    # (2569, 1919, 1)
    scale_nsdm = (scale_nsdm-np.min(scale_nsdm))/(np.max(scale_nsdm)-np.min(scale_nsdm))

    scale_nsdm = np.expand_dims(scale_nsdm, axis=2)
    scale_nsdm = np.transpose(scale_nsdm,(2,0,1))
    scale_nsdm = torch.from_numpy(scale_nsdm)

    # scale_nsdm =np.transpose()
    # scale_image = np.asarray(Image.open(image_path).convert('RGB')).astype(np.float32)
    # nsdm_image = img_transforms(scale_nsdm)
    nsdm_image = scale_nsdm.unsqueeze(0).cuda()

    N, C, H_, W_ = scale_image.shape
    nsdm_N, nsdm_C, nsdm_H_, nsdm_W_ = nsdm_image.shape
    print(f"Height: {H_} Width: {W_}")

    full_probs = torch.zeros((N, num_classes, H_, W_), device=scale_image.device)
    count_predictions = torch.zeros((N, num_classes, H_, W_), device=scale_image.device)

    # nsdm_full_probs = torch.zeros((nsdm_N, num_classes, nsdm_H_, nsdm_W_), device=nsdm_image.device)  #
    # nsdm_count_predictions = torch.zeros((nsdm_N, num_classes, nsdm_H_, nsdm_W_), device=nsdm_image.device)  #

    h_overlap_length = overlap
    w_overlap_length = overlap

    h = 0
    slide_finish = False
    while not slide_finish:

        if h + crop_size <= H_:
            # print(f"h: {h}")
            # set row flag
            slide_row = True
            # initial row start
            w = 0
            while slide_row:
                if w + crop_size <= W_:
                    # print(f" h={h} w={w} -> h'={h + crop_size} w'={w + crop_size}")
                    patch_image = scale_image[:, :, h:h + crop_size, w:w + crop_size]
                    patch_nsdm = nsdm_image[:, :, h:h + crop_size, w:w + crop_size]

                    patch_pred_image = tta_inference(patch_image,patch_nsdm, model, num_classes=num_classes, scales=scales,flip=flip)
                    # patch_pred_image = pred_img(model, patch_image)
                    count_predictions[:, :, h:h + crop_size, w:w + crop_size] += 1
                    full_probs[:, :, h:h + crop_size, w:w + crop_size] += patch_pred_image

                else:
                    # print(f" h={h} w={W_ - crop_size} -> h'={h + crop_size} w'={W_}")
                    patch_image = scale_image[:, :, h:h + crop_size, W_ - crop_size:W_]
                    patch_nsdm = nsdm_image[:, :, h:h + crop_size, nsdm_W_ - crop_size:nsdm_W_]

                    patch_pred_image = tta_inference(patch_image,patch_nsdm ,model, num_classes=num_classes, scales=scales,
                                                     flip=flip)
                    # patch_pred_image = pred_img(model, patch_image)
                    count_predictions[:, :, h:h + crop_size, W_ - crop_size:W_] += 1
                    full_probs[:, :, h:h + crop_size, W_ - crop_size:W_] += patch_pred_image
                    slide_row = False

                w += w_overlap_length

        else:
            # print(f"h: {h}")
            # set last row flag
            slide_last_row = True
            # initial row start
            w = 0
            while slide_last_row:
                if w + crop_size <= W_:
                    # print(f"h={H_ - crop_size} w={w} -> h'={H_} w'={w + crop_size}")
                    patch_image = scale_image[:, :, H_ - crop_size:H_, w:w + crop_size]
                    patch_nsdm = nsdm_image[:, :, nsdm_H_ - crop_size:nsdm_H_, w:w + crop_size]
                    patch_pred_image = tta_inference(patch_image,patch_nsdm, model, num_classes=num_classes, scales=scales,
                                                     flip=flip)
                    count_predictions[:, :, H_ - crop_size:H_, w:w + crop_size] += 1
                    full_probs[:, :, H_ - crop_size:H_, w:w + crop_size] += patch_pred_image

                else:
                    # print(f"h={H_ - crop_size} w={W_ - crop_size} -> h'={H_} w'={W_}")
                    patch_image = scale_image[:, :, H_ - crop_size:H_, W_ - crop_size:W_]
                    patch_nsdm = nsdm_image[:, :, nsdm_H_ - crop_size:nsdm_H_, w:w + crop_size]
                    patch_pred_image = tta_inference(patch_image,patch_nsdm, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:, :, H_ - crop_size:H_, W_ - crop_size:W_] += 1
                    full_probs[:, :, H_ - crop_size:H_, W_ - crop_size:W_] += patch_pred_image

                    slide_last_row = False
                    slide_finish = True

                w += w_overlap_length

        h += h_overlap_length

    full_probs /= count_predictions
    return full_probs


def load_model(model_path):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = get_net(model_name, input_bands, num_class, img_size)
    # model = torch.nn.DataParallel(model, device_ids=[0])
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if use_gpu:
        model.cuda()
    model.eval()
    return model


# def img_transforms(img):
#     img = np.array(img).astype(np.float32)
#     sample = {'image': img}
#     transform = transforms.Compose([
#         tr.Normalize(mean=mean, std=std),
#         tr.ToTensor()])
#     sample = transform(sample)
#     return sample['image']

def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    transform = transforms.Compose([
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']

def parse_color_table(color_txt):
    f = open(color_txt, 'r').readlines()
    color_table = []
    for info in f:
        x = info.split('#')[0].split('/')
        color_table.append((int(x[0]), int(x[1]), int(x[2])))
    return color_table

if __name__ == '__main__':
    test_path = '../vai_data/train_img/test'
    model_path = '../finalvision3_files/DeepLabV3Plus_3/pth_DeepLabV3Plus/281.pth'
    color_txt = '../vai_data/color_table_isprs.txt'
    color_table = parse_color_table(color_txt)
    if os.path.exists(save_path) is True:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'color_big'))
        os.mkdir(os.path.join(save_path, 'gray_big'))
    else:
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'color_big'))
        os.mkdir(os.path.join(save_path, 'gray_big'))

    model = load_model(model_path)
    test_imgs = os.listdir(test_path)
    # overlap=256,
    for name in tqdm.tqdm(test_imgs):
        output = slide_pred(
            model=model,
            image_path= os.path.join(test_path, name),
            nsdm_path= os.path.join('../vai_data/dsm/test', name.split(".")[0]+'.jpg'),
            num_classes=6,
            crop_size=512,
            overlap=64,
            scales=[1.0], flip=True)
        pred_gray = torch.argmax(output, 1)

        # pred_vis = label_mapping(pred_gray)
        # pred_gray = pred_gray[0].cpu().data.numpy().astype(np.int32)

        pred_gray = torch.argmax(output, 1)
        pred_gray = pred_gray[0].cpu().data.numpy().astype(np.int32)
        pred_vis = label_mapping(pred_gray)
        # cv2.imwrite(os.path.join(save_path,  'gray_big', name), pred_gray)
        yimage.io.write_image(os.path.join(save_path,  'gray_big', name), pred_gray+1, color_table=color_table)
        cv2.imwrite(os.path.join(save_path,  'color_big', name), pred_vis)
