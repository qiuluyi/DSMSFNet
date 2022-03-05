import os
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, roc_auc_score, confusion_matrix
# def accuracy(preds, label):
#     valid = (label >= 0)
#     acc_sum = (valid * (preds == label)).sum()
#     valid_sum = valid.sum()
#     acc = float(acc_sum) / (valid_sum + 1e-10)
#     return acc, valid_sum

def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum-np.where(label==255,1,0).sum()+1e-13)
    return acc, valid_sum

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
def evaluate(data_dir, pred_dir, num_class):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    names = os.listdir(pred_dir)
    pred_all = []
    gt_all = []
    for name in names:
        pred = Image.open('{}{}'.format(pred_dir, name)).convert('L')
        print(set(np.asarray(pred).flatten()))
        # gt = Image.open(gt_name).convert('L')
        gt = Image.open(os.path.join(data_dir, 'img_lab', name.replace('_sat.jpg', '_mask.png'))).convert('L')
        # pred = pred.resize(gt.size)
        pred = np.array(pred, dtype=np.int64)
        pred = np.where(pred > 0, 1, 0)
        gt = np.array(gt, dtype=np.int64)
        gt = np.where(gt>0, 1, 0)
        pred_all+=list(pred.flatten())
        gt_all += list(gt.flatten())
        acc, pix = accuracy(pred, gt)
        intersection, union = intersectionAndUnion(pred, gt, num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))
    print(set(gt_all))
    print(set(pred_all))
    recall = recall_score(np.asarray(gt_all), np.asarray(pred_all))
    precision = precision_score(np.asarray(gt_all), np.asarray(pred_all))
    # print('kappa is {}'.format(str(cohen_kappa_score(np.asarray(gt_all), np.asarray(pred_all)))))
    print('recall is {}'.format(str(recall)))
    print('precision is {}'.format(str(precision)))
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.4f}%'
          .format(iou.mean(), acc_meter.average() * 100))
    # print(acc_meter.val)
    return iou.mean(), acc_meter.average(), recall, precision

def evaluate2(pred_dir, gt_dir):
    names = os.listdir(pred_dir)
    s = 0
    note = []
    for name in names:

        pred = Image.open('{}{}'.format(pred_dir, name))
        size = pred.size
        gt_name = '{}{}'.format(gt_dir, name.replace('jpg', 'png'))

        x = os.path.isfile(gt_name)
        if x is False:
            gt = np.zeros((size[1], size[0]))
        else:
            gt = Image.open(gt_name).convert('L')
        # gt = Image.open('{}{}'.format(gt_dir, name))
        # pred = pred.resize(gt.size)
        pred = np.array(pred, dtype=np.int64)
        gt = np.array(gt, dtype=np.int64)
        gt = np.where(gt>0, 1, 0)
        p1 = np.sum(pred.flatten())
        g1 = np.sum(gt.flatten())
        error = abs(p1-g1)*1.00/(size[0]*size[1])
        note.append([name, str(error)])
        # print(error)
        if error>0.2:
            # print(error)
            s+=1
            print(name)
    print(s*1.00/len(names))
    return note
# @numba.jit
# def remove_se(files):
#     for i in range(len(files)):
#         print(files[i])
#         img = Image.open(p + files[i])
#         size = img.size
#         arr = np.asarray(img)
#         print(arr.shape)
#
#         gray_img = np.zeros((size[1], size[0]))
#         for j in range(size[1]):
#             for k in range(size[0]):
#                 # print([j, k])
#                 # print(str(list(arr[j][k])))
#                 idx = dict[str(list(arr[j][k]))]
#                 gray_img[j][k] = idx
#                 # print(idx)
#         g_img = Image.fromarray(gray_img.astype(np.uint8))
#         g_img.save(tgt + files[i])


if __name__ == '__main__':
    p = './whole_predict_gray/'

    files = os.listdir(p)
    label = [[0, 0, 0], [255, 255, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0], [128, 0, 0], [128, 0, 128], [0, 128, 0],
             [0, 255, 255], [0, 255, 0], [0, 128, 128], [0, 255, 128], [255, 0, 128], [0, 128, 255]]
    indx = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13]
    print(len(indx))
    dict = {}

    for i in range(len(label)):
        dict[str(label[i])] = indx[i]

    # remove_se(files)

    gt = './data/VOC/VOCdevkit/VOC2012/test_gt/'
    evaluate(p, gt, 14)
