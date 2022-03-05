from sklearn.metrics import cohen_kappa_score,classification_report
import numpy as np
import cv2
import os

output = './whole_predict/'
gt = './data/VOC/VOCdevkit/VOC2012/test_gt_vis/'
imgs = os.listdir(gt)
gt_ = np.asarray([0,])
pred_ = np.asarray([0])

for i in range(len(imgs)):
    img_gt = cv2.imread(gt+imgs[i], 0)
    img_gt = img_gt.flatten()
    gt_ = np.concatenate((gt_, img_gt))
    img_pred = cv2.imread(output + imgs[i], 0)
    img_pred = img_pred.flatten()
    pred_ = np.concatenate((pred_, img_pred))
    # print(cohen_kappa_score(gt_, pred_))
    print(imgs[i])

print(classification_report(gt_, pred_))
print('ok')


