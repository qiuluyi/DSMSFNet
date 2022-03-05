import os
import shutil

p1 = '/home/weikai/chick_f_/data/tmp_ch/'
p2 = '/home/weikai/chick_f_/data/JPEGImages/'
p3 = '/home/weikai/chick_f_/data/lb/'
p4 = '/home/weikai/chick_f_/data/SegmentationClass/'

imgs = os.listdir('./data/tmp_ch')
for i in range(len(imgs)):
    if 'im' in imgs[i]:
        shutil.move(p1+imgs[i], p2+imgs[i])
    elif 'lb' in imgs[i]:
        shutil.move(p1 + imgs[i], p3 + imgs[i])
    else:
        shutil.move(p1 + imgs[i], p4 + imgs[i])