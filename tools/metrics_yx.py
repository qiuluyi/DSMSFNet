from __future__ import print_function
import numpy as np
import cv2
import gdal
import sys
import os
import shutil


def check_path(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')
    else:
        shutil.rmtree(pathname)
        os.makedirs(pathname)
        print(pathname + ' has been reset!')


def check_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)
        print(filename + ' has been removed!')
    else:
        print(filename + ' will be created!')


def generate_baselist(file_path, suffix):
    suffix_length = len(suffix)
    basename_list = []
    listfile = os.listdir(file_path)
    listfile.sort()
    for basename in listfile:
        if basename[(-suffix_length):] != suffix:
            continue
        basename_list.append(basename[:(-suffix_length)])

    return basename_list


def generate_list(file_path, basename_list, suffix):
    filename_list = []
    for basename in basename_list:
        filename = file_path + '/' + basename + suffix
        filename_list.append(filename)

    return filename_list


def gdal_to_cv(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        print('FATAL: GDAL open file failed. [%s]' % filename)
        sys.exit(1)

    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    img_nbands = dataset.RasterCount

    band_list = [i + 1 for i in range(img_nbands)]
    if img_nbands == 3 or img_nbands == 4:
        band_list[0] = 3
        band_list[2] = 1

    img_array = np.zeros((img_width, img_height, 1))
    for i in range(img_nbands):
        band = dataset.GetRasterBand(band_list[i])
        data_type = band.DataType
        if data_type == gdal.GDT_Byte:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.uint8)
        elif data_type == gdal.GDT_UInt16:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.uint16)
        elif data_type == gdal.GDT_Int16:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.int16)
        elif data_type == gdal.GDT_UInt32:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.uint32)
        elif data_type == gdal.GDT_Int32:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.int32)
        elif data_type == gdal.GDT_Float32:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.float32)
        elif data_type == gdal.GDT_Float64:
            img_arr = band.ReadAsArray(0, 0, img_width, img_height).astype(np.float64)
        else:
            print('ERROR: GDAL unknown data type. [%s]' % filename)

        if i == 0:
            img_array = img_arr.reshape((img_height, img_width, 1))
        else:
            img_arr_reshape = img_arr.reshape((img_height, img_width, 1))
            img_array = np.append(img_array, img_arr_reshape, axis=2)

    return img_array


def load_name_and_color_config_file(filepath, num_classes):
    name_list = []
    color_table = []
    if filepath is not None:
        idx = 0
        with open(filepath, 'r') as f:
            for line in f:
                _name = line.strip().split('#')[1]
                if _name == '':
                    _name = f'class {idx}'
                name_list.append(_name)
                _color_table = line.strip().split('#')[0]
                if _color_table == '':
                    _color_table = f'{idx}/{idx}/{idx}'
                color_table.append(tuple([int(i) for i in _color_table.split('/')]))
                idx += 1
    else:
        for i in range(len(num_classes)):
            name_list.append(f'class {i}')
            color_table.append(tuple([int(item) for item in f'{i}/{i}/{i}'.split('/')]))
    return name_list, color_table


def f_measure(TP, FP, FN, f_beta):
    F_score_up = (1 + f_beta ** 2) * TP
    F_score_down = (1 + f_beta ** 2) * TP + (f_beta ** 2) * FN + FP

    F_score_down_ = F_score_down.copy()
    F_score_down_[F_score_down == 0] = 1
    F_score = np.true_divide(F_score_up, F_score_down_)
    F_score[F_score_down == 0] = 1.0

    return F_score


def compute_precision_recall(TP, FP, FN):
    TPFP = TP + FP
    TPFP_ = TPFP.copy()
    TPFP_[TPFP == 0] = 1
    precision = np.true_divide(TP, TPFP_)
    precision[TPFP == 0] = 1.0

    TPFN = TP + FN
    TPFN_ = TPFN.copy()
    TPFN_[TPFN == 0] = 1
    recall = np.true_divide(TP, TPFN_)
    recall[TPFN == 0] = 1.0

    return precision, recall


def calc_average(values, label_count, with_background):
    mean_value = 0.0
    index_start = 0
    _label_count = label_count
    if with_background == 0:
        index_start = 1
        _label_count -= 1
    for i in range(index_start, label_count):
        mean_value += values[i]
    mean_value /= _label_count

    return mean_value


def compute(gt_img_filename, pred_img_filename, label_count, f_beta, with_background, ignore_value, accumulator):
    intersection = np.zeros(label_count)
    union = np.zeros(label_count)
    IoU = np.zeros(label_count)

    TP = np.zeros(label_count)
    # TN = np.zeros(label_count)
    FP = np.zeros(label_count)
    FN = np.zeros(label_count)
    F_score = np.zeros(label_count)

    gt_img = gdal_to_cv(gt_img_filename)
    gt_img = gt_img + 1
    gt_img = np.where(gt_img == 7, 0, gt_img)
    pred_img = gdal_to_cv(pred_img_filename)
    pred_img = pred_img + 1
    pred_img[gt_img == ignore_value] = ignore_value

    img_sum = gt_img + pred_img
    img_eroded_count = np.sum(img_sum == ignore_value)

    for i in range(label_count):
        gt_copy = gt_img.copy()
        pred_copy = pred_img.copy()

        gt_copy[gt_img != i] = 0
        gt_copy[gt_img == i] = 1
        pred_copy[pred_img != i] = 0
        pred_copy[pred_img == i] = 2
        gt_pred = gt_copy + pred_copy

        intersection[i] = np.sum(gt_pred == 3)
        union[i] = np.sum(gt_pred != 0)

        TP[i] = np.sum(gt_pred == 3)
        FN[i] = np.sum(gt_pred == 1)
        FP[i] = np.sum(gt_pred == 2)
        # TN[i] = np.sum(gt_pred == 0)

    img_diff = gt_img - pred_img
    pixel_right = np.sum(img_diff == 0)
    elem_count = gt_img.shape[0] * gt_img.shape[1]

    real_elem_count = float(elem_count)
    pixel_acc = (pixel_right - img_eroded_count) / (real_elem_count - img_eroded_count)

    union_ = union.copy()
    union_[union == 0] = 1
    IoU = np.true_divide(intersection, union_)
    IoU[union == 0] = 1.0
    mIoU = calc_average(IoU, label_count, with_background)

    F_score = f_measure(TP, FP, FN, f_beta)
    F_score[np.isnan(F_score)] = 1.0
    mF_score = calc_average(F_score, label_count, with_background)

    precision, recall = compute_precision_recall(TP, FP, FN)
    precision[np.isnan(precision)] = 1.0
    recall[np.isnan(recall)] = 1.0
    mPrecision = calc_average(precision, label_count, with_background)
    mRecall = calc_average(recall, label_count, with_background)

    if with_background == 0:
        real_IoU = IoU[1:]
        real_F_score = F_score[1:]
        real_precision = precision[1:]
        real_recall = recall[1:]
    else:
        real_IoU = IoU
        real_F_score = F_score
        real_precision = precision
        real_recall = recall

    accumulator['TP'] += TP
    accumulator['FN'] += FN
    accumulator['FP'] += FP
    accumulator['num_pixels'] += elem_count
    accumulator['num_eroded'] += img_eroded_count

    return accumulator, dict(
        pixel_acc=pixel_acc,
        IoU=real_IoU,
        mIoU=mIoU,
        F_score=real_F_score,
        mF_score=mF_score,
        precision=real_precision,
        mPrecision=mPrecision,
        recall=real_recall,
        mRecall=mRecall
    )


def compute_average_accuracy(accumulator, label_count, f_beta, with_background):
    tps = 0.0
    for i in range(label_count):
        tps += accumulator['TP'][i]
    pixel_acc = (tps - accumulator['num_eroded']) / (accumulator['num_pixels'] - accumulator['num_eroded'])

    intersection = accumulator['TP']
    union = accumulator['TP'] + accumulator['FN'] + accumulator['FP']
    union_ = union.copy()
    union_[union == 0] = 1
    IoU = np.true_divide(intersection, union_)
    IoU[union == 0] = 1.0
    mIoU = calc_average(IoU, label_count, with_background)

    F_score = f_measure(accumulator['TP'], accumulator['FP'], accumulator['FN'], f_beta)
    F_score[np.isnan(F_score)] = 1.0
    mF_score = calc_average(F_score, label_count, with_background)

    precision, recall = compute_precision_recall(accumulator['TP'], accumulator['FP'], accumulator['FN'])
    precision[np.isnan(precision)] = 1.0
    recall[np.isnan(recall)] = 1.0
    mPrecision = calc_average(precision, label_count, with_background)
    mRecall = calc_average(recall, label_count, with_background)

    if with_background == 0:
        real_IoU = IoU[1:]
        real_F_score = F_score[1:]
        real_precision = precision[1:]
        real_recall = recall[1:]
    else:
        real_IoU = IoU
        real_F_score = F_score
        real_precision = precision
        real_recall = recall

    return dict(
        pixel_acc=pixel_acc,
        IoU=real_IoU,
        mIoU=mIoU,
        F_score=real_F_score,
        mF_score=mF_score,
        precision=real_precision,
        mPrecision=mPrecision,
        recall=real_recall,
        mRecall=mRecall
    )


def compute_batch(gt_filename_list, pred_filename_list, result_file, label_count, f_beta, with_background, ignore_value,
                  name_and_color_config_file):
    np.set_printoptions(suppress=True)
    name_list, _ = load_name_and_color_config_file(name_and_color_config_file, label_count)
    if with_background == 0:
        name_list = name_list[1:]
    accumulator = {}
    accumulator['TP'] = np.zeros(label_count, dtype=np.float64)
    accumulator['FN'] = np.zeros(label_count, dtype=np.float64)
    accumulator['FP'] = np.zeros(label_count, dtype=np.float64)
    accumulator['num_pixels'] = 0.0
    accumulator['num_eroded'] = 0.0

    file_count = len(gt_filename_list)
    for i in range(file_count):
        print('Index:           {}'.format(i))
        gt_filename = gt_filename_list[i].strip()
        pred_filename = pred_filename_list[i].strip()
        print(' GT   File:      {}'.format(gt_filename))
        print(' Pred File:      {}'.format(pred_filename))
        accumulator, acc_dict = compute(gt_filename, pred_filename, label_count, f_beta, with_background, ignore_value,
                                        accumulator)
        print(' Pixel Acc:      {}'.format(str(acc_dict['pixel_acc'])))
        print(' Mean IoU:       {}'.format(str(acc_dict['mIoU'])))
        print(' Mean F1 score:  {}'.format(str(acc_dict['mF_score'])))
        print(' Mean Precision: {}'.format(str(acc_dict['mPrecision'])))
        print(' Mean Recall:    {}'.format(str(acc_dict['mRecall'])))
        print(' Class Names:    {}'.format(' # '.join(name_list)))
        print(' IoU:            {}'.format(str(acc_dict['IoU'])))
        print(' F{} score:       {}'.format(str(f_beta), str(acc_dict['F_score'])))
        print(' Precision:      {}'.format(str(acc_dict['precision'])))
        print(' Recall:         {}'.format(str(acc_dict['recall'])))
        with open(result_file, 'a') as f:
            f.write('Index:           {}\n'.format(i))
            f.write(' GT   File:      {}\n'.format(gt_filename))
            f.write(' Pred File:      {}\n'.format(pred_filename))
            f.write(' Pixel Acc:      {}\n'.format(str(acc_dict['pixel_acc'])))
            f.write(' Mean IoU:       {}\n'.format(str(acc_dict['mIoU'])))
            f.write(' Mean F1 score:  {}\n'.format(str(acc_dict['mF_score'])))
            f.write(' Mean Precision: {}\n'.format(str(acc_dict['mPrecision'])))
            f.write(' Mean Recall:    {}\n'.format(str(acc_dict['mRecall'])))
            f.write(' Class Names:    {}\n'.format(' # '.join(name_list)))
            f.write(' IoU:            {}\n'.format(str(acc_dict['IoU'])))
            f.write(' F{} score:       {}\n'.format(str(f_beta), str(acc_dict['F_score'])))
            f.write(' Precision:      {}\n'.format(str(acc_dict['precision'])))
            f.write(' Recall:         {}\n\n'.format(str(acc_dict['recall'])))

    acc_dict = compute_average_accuracy(accumulator, label_count, f_beta, with_background)
    print('============================================================')
    print(' Pixel Acc:      {}'.format(str(acc_dict['pixel_acc'])))
    print(' Mean IoU:       {}'.format(str(acc_dict['mIoU'])))
    print(' Mean F1 score:  {}'.format(str(acc_dict['mF_score'])))
    print(' Mean Precision: {}'.format(str(acc_dict['mPrecision'])))
    print(' Mean Recall:    {}'.format(str(acc_dict['mRecall'])))
    print(' Class Names:    {}'.format(' # '.join(name_list)))
    print(' IoU:            {}'.format(str(acc_dict['IoU'])))
    print(' F{} score:       {}'.format(str(f_beta), str(acc_dict['F_score'])))
    print(' Precision:      {}'.format(str(acc_dict['precision'])))
    print(' Recall:         {}'.format(str(acc_dict['recall'])))
    print('============================================================')
    with open(result_file, 'a') as f:
        f.write('============================================================\n')
        f.write(' Pixel Acc:      {}\n'.format(str(acc_dict['pixel_acc'])))
        f.write(' Mean IoU:       {}\n'.format(str(acc_dict['mIoU'])))
        f.write(' Mean F1 score:  {}\n'.format(str(acc_dict['mF_score'])))
        f.write(' Mean Precision: {}\n'.format(str(acc_dict['mPrecision'])))
        f.write(' Mean Recall:    {}\n'.format(str(acc_dict['mRecall'])))
        f.write(' Class Names:    {}\n'.format(' # '.join(name_list)))
        f.write(' IoU:            {}\n'.format(str(acc_dict['IoU'])))
        f.write(' F{} score:       {}\n'.format(str(f_beta), str(acc_dict['F_score'])))
        f.write(' Precision:      {}\n'.format(str(acc_dict['precision'])))
        f.write(' Recall:         {}\n'.format(str(acc_dict['recall'])))
        f.write('============================================================\n\n')

    print('Writing Result File: ', result_file)


def run_core(gt_path, gt_suffix, pred_path, pred_suffix, result_file, label_count, f_beta, with_background,
             ignore_value, name_and_color_config_file):
    check_file(result_file)
    basename_list = generate_baselist(pred_path, pred_suffix)
    gt_filename_list = generate_list(gt_path, basename_list, gt_suffix)
    pred_filename_list = generate_list(pred_path, basename_list, pred_suffix)
    compute_batch(gt_filename_list, pred_filename_list, result_file, label_count, f_beta, with_background, ignore_value,
                  name_and_color_config_file)


def run():
    gt_path = r'Z:\private\dongsj\0sjcode\code0914_vaiseg\vai_data\gt_nobd'
    gt_suffix = 'tif'
    pred_path = r'Z:\private\dongsj\0sjcode\code0914_vaiseg\results281\gray_big'
    pred_suffix = 'tif'
    result_file = 'res'
    label_count = 7
    f_beta = 1
    with_background = 0
    ignore_value = 0
    name_and_color_config_file = r'Z:\private\dongsj\0sjcode\code0914_vaiseg\vai_data\color_table_isprs.txt'

    run_core(gt_path, gt_suffix, pred_path, pred_suffix, result_file, label_count, f_beta, with_background,
             ignore_value, name_and_color_config_file)


if __name__ == '__main__':
    run()

 # Pixel Acc:      0.9024778934681423
 # Mean IoU:       0.8112621987959662
 # Mean F1 score:  0.8940129975987935
 # Mean Precision: 0.8925159651032745
 # Mean Recall:    0.8969245643578336