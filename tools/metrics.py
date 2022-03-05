import os
import yimage
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import tqdm
from collections import Counter
import config

def cal_acc(pred, gt):
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

def metrics(confu_mat_total, save_path='./'):
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)
    raw_sum = np.sum(confu_mat, axis=0)
    oa = 0

    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)
    TP = []

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP
    f1_m = []
    iou_m = []
    for i in range(class_num):
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)

    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    if save_path is not None:
        with open(save_path + 'metrics.txt', 'w') as f:
            f.write('OA:\t%.4f\n' % (oa * 100))
            f.write('kappa:\t%.4f\n' % (kappa * 100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m) * 100))
            f.write('mIou:\t%.4f\n' % (np.mean(iou_m) * 100))
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(f1_m[i]) * 100))
            f.write('\n')
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(iou_m[i]) * 100))
            f.write('\n')

if __name__ == '__main__':
    boundary_id = 0
    gts = sorted(os.listdir(p_gt))
    preds = sorted(os.listdir(p_pred))
    up_all = []
    down_all = []
    cm_init = np.zeros((7, 7))
    for name in tqdm.tqdm(preds):
        print("******************{}******************".format(name))
        pred = yimage.io.read_image(os.path.join(p_pred, name))
        gt = yimage.io.read_image(os.path.join(p_gt, name))
        gt = gt + 1
        gt = np.where(gt == 7, 0, gt)
        acc, up, down = cal_acc(pred, gt)
        f1, cm = cal_f1(pred, gt)
        up_all.append(up)
        down_all.append(down)
        cm_init += cm
        print('file is {}, acc is {}\n, f1 is'.format(name, acc))
        print(f1)
    metrics(cm_init)
    acc_all = np.sum(up_all)/np.sum(down_all)
    print('####acc is {}'.format(acc_all))
