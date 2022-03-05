import os
import tqdm
from PIL import Image
import numpy as np
import yimage
from sklearn.metrics import f1_score, accuracy_score, classification_report

from sklearn.metrics import f1_score, confusion_matrix
def cal_acc(pred, gt):
    boundary_id = 0
    pixel_num = pred.shape[0] * pred.shape[1]
    pred[gt == 0] = 0
    boundary_num = np.sum(gt == boundary_id)

    true_num = np.sum(gt == pred)
    pixel_acc = (true_num - boundary_num) / (pixel_num - boundary_num)
    return pixel_acc, true_num - boundary_num, pixel_num - boundary_num


def cal_f1(pred, gt):
    pred[gt == 0] = 0
    f1 = f1_score(gt.flatten(), pred.flatten(), average=None)
    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=[i for i in range(7)])
    return f1, cm


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
    acc = float(acc_sum) / (valid_sum - np.where(label == 255, 1, 0).sum() + 1e-13)
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

def metrics(confu_mat_total, save_path='./'):
    '''
    :param confu_mat: 总的混淆矩阵
    backgound：是否干掉背景
    :return: txt写出混淆矩阵, precision，recall，IOU，f-score
    '''
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, f1-score，f1-m以及mIOU

    f1_m = []
    iou_m = []
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)

    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    if save_path is not None:
        with open(save_path + 'test_tlmfnet_accuracy.txt', 'w') as f:
            f.write('OA:\t%.4f\n' % (oa * 100))
            f.write('kappa:\t%.4f\n' % (kappa * 100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m) * 100))
            f.write('mIou:\t%.4f\n' % (np.mean(iou_m) * 100))

            # 写出precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')

            # 写出recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')

            # 写出f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(f1_m[i]) * 100))
            f.write('\n')

            # 写出 IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(iou_m[i]) * 100))
            f.write('\n')

def evaluate(val_gt, pred_dir, num_class):

    names = os.listdir(pred_dir)
    # pred_all = []
    # gt_all = []
    p_gt = '../vai_data/gt_nobd_2'
    gts = sorted(os.listdir(p_gt))
    preds = sorted(os.listdir(pred_dir))

    up_all = []
    down_all = []
    cm_init = np.zeros((7, 7))
    f1_total=0
    f1_all =0
    for name in tqdm.tqdm(preds):
        pred = yimage.io.read_image(os.path.join(pred_dir, name))
        gt = yimage.io.read_image(os.path.join(p_gt, name))
        gt = gt + 1
        gt = np.where(gt == 7, 0, gt)
        
        # pred = Image.open(os.path.join(pred_dir, name)).convert('L')
        # gt = Image.open(gt_name).convert('L')
        # gt = Image.open(os.path.join(p_gt, name)).convert('L')
        # pred = pred.resize(gt.size)
        # pred = np.array(pred, dtype=np.int64)
        # gt = np.array(gt, dtype=np.int64)
        acc, up, down = cal_acc(pred, gt)
        up_all.append(up)
        down_all.append(down)
        f1, cm = cal_f1(pred, gt)
        f1_total= (f1[1]+f1[2]+f1[3]+f1[4]+f1[5])/5
        f1_all = f1_all + f1_total
        cm_init += cm

    metrics(cm_init)
    f1_all = f1_all/17
    acc_all = np.sum(up_all)/np.sum(down_all)

    """
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
        # acc, pix = accuracy(pred, gt)
        pred_all += list(pred.flatten())
        gt_all += list(gt.flatten())
        intersection, union = intersectionAndUnion(pred, gt, num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

    f1 = f1_score(gt_all, pred_all, labels=[0, 1, 2, 3, 4], average='macro')
    acc_sklearn = accuracy_score(gt_all, pred_all)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))
    print('f1_sklearn is {}'.format(str(f1)))
    print('acc_sklearn is {}'.format(str(acc_sklearn)))
    # print(classification_report(gt_all, pred_all))
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.4f}%'
          .format(iou.mean(), acc_meter.average() * 100))
    """
    return  acc_all, f1_all


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
        gt = np.where(gt > 0, 1, 0)
        p1 = np.sum(pred.flatten())
        g1 = np.sum(gt.flatten())
        error = abs(p1 - g1) * 1.00 / (size[0] * size[1])
        note.append([name, str(error)])
        # print(error)
        if error > 0.2:
            # print(error)
            s += 1
            print(name)
    print(s * 1.00 / len(names))
    return note


if __name__ == '__main__':
    p = './whole_predict_gray/'

    files = os.listdir(p)
    label = [[0, 0, 0], [255, 255, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0], [128, 0, 0], [128, 0, 128],
             [0, 128, 0],
             [0, 255, 255], [0, 255, 0], [0, 128, 128], [0, 255, 128], [255, 0, 128], [0, 128, 255]]
    indx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    print(len(indx))
    dict = {}

    for i in range(len(label)):
        dict[str(label[i])] = indx[i]

    # remove_se(files)

    gt = './data/VOC/VOCdevkit/VOC2012/test_gt/'
    evaluate(p, gt, 14)
