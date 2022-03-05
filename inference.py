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
from PIL import Image
from tools.utils import read_image


def tta_inference(inp,nsdm, model, num_classes=8, scales=[1.0], flip=True):
    b, _, h, w = inp.size()
    preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
    for scale in scales:
        size = (int(scale * h), int(scale * w))
        resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True, )
        resized_nsdm = F.interpolate(nsdm, size=size, mode='bilinear', align_corners=True, )
        pred = model_inference(model, resized_img.to(inp.device), resized_nsdm.to(nsdm.device),flip)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True, )
        preds += pred
    return preds / (len(scales))

def model_inference(model, image, nsdm, flip=True):
    with torch.no_grad():
        output = model(image,nsdm)
        if flip:
            fimg = image.flip(2)
            output += model(fimg,nsdm).flip(2)
            fimg = image.flip(3)
            output += model(fimg,nsdm).flip(3)
            return output / 3
        return output


def pred_img(model, image,nsdm):
    with torch.no_grad():
        output = model(image,nsdm)
    return output

def slide_pred(model, image_path, nsdm_path, num_classes=6, crop_size=512, overlap=256, scales=[1.0], flip=True):
    # torch.Size([2569, 1919, 3])
    scale_image = read_image(image_path).astype(np.float32)
    # torch.Size([3, 2569, 1919])
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
    model = get_net(model_name, input_bands, num_class, img_size)
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


def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    transform = transforms.Compose([
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']




if __name__ == '__main__':
    # lab_path = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\vai_data\val_gt\top_mosaic_09cm_area1.tif'
    # img_path = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\vai_data\train_img\val\top_mosaic_09cm_area1.tif'
    # model_path = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\0913_files\DeepLabV3Plus_3\pth_DeepLabV3Plus\101.pth'
    lab_path = '/nfs/project/netdisk/192.168.0.31/d/private/dongsj/0sjcode/code0906_vaiseg/vai_data/val_gt/top_mosaic_09cm_area1.tif'
    img_path = '/nfs/project/netdisk/192.168.0.31/d/private/dongsj/0sjcode/code0906_vaiseg/vai_data/train_img/val/top_mosaic_09cm_area1.tif'
    model_path = '/nfs/project/netdisk/192.168.0.31/d/private/dongsj/0sjcode/code0906_vaiseg/0913_files/DeepLabV3Plus_3/pth_DeepLabV3Plus/101.pth'

    model = load_model(model_path)
    # image = yimage.io.read_image(img_path)
    image = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float32)

    # image = img_transforms(image)
    # image = image.unsqueeze(0).cuda()
    pred = slide_pred(
        model=model,
        image_path=img_path,
        num_classes=6,
        crop_size=512,
        overlap=256,
        scales=[1.0],
        flip=True)
    print(pred.shape)
    pred_gray = torch.argmax(pred, 1)
    pred_gray = pred_gray[0].cpu().data.numpy().astype(np.int32)
    lab = yimage.io.read_image(lab_path)
    from sklearn.metrics import accuracy_score, classification_report
    print(classification_report(lab.flatten(), pred_gray.flatten()))
    pred_vis = label_mapping(pred_gray)
    cv2.imwrite('test_vis.tif', pred_vis)
    print('ok')
