import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from config import *

def get_pic_1(ddict, name1, pic_name):
    plt.figure(pic_name)
    ax = plt.gca()
    ax.set_xlabel('epoch')
    x_list = [i*save_iter + 1 for i in range(len(ddict[name1]))]
    ax.plot(x_list, ddict[name1], color='r', linewidth=1, alpha=0.6, label=name1)
    plt.legend()
    plt.savefig(os.path.join(save_dir_model, pic_name))
    plt.clf()


def draw_pic(cur_log):
    f = cur_log.split('\n')
    title = [i.split(':')[0] for i in f[0].split(', ')]
    ddict = {}
    for i in range(len(title)):
        ddict[title[i]] = []
    for i in range(len(f)):
        if i == len(f)-1:
            continue
        cur_data = f[i].split(', ')
        cur_data = [float(i.split(':')[-1]) for i in cur_data]
        for j in range(len(cur_data)):
            ddict[title[j]].append(cur_data[j])

    get_pic_1(ddict, 'train_loss', 'train_loss.png')
    get_pic_1(ddict, 'learning_rate', 'learning_rate.png')
    get_pic_1(ddict, 'vallosses', 'vallosses.png')
    get_pic_1(ddict, 'val_rec', 'val_rec.png')
    get_pic_1(ddict, 'val_prec', 'val_prec.png')
    get_pic_1(ddict, 'val_acc', 'val_acc.png')
