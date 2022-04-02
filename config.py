import os
batch_size = 4
use_gpu = True
img_size = 512
overlap = 32
epoches = 2000
base_lr = 0.001
weight_decay = 2e-5
momentum = 0.9
power = 0.99
gpu_id = '0'
loss_type = 'ce'
save_iter = 10
num_workers = 1
val_visual = True
image_driver = 'gdal'   #pillow, gdal
num_class = 6
model_name = 'DSMSFNet'
input_bands = 3

if input_bands == 4:
    mean = (0.472455, 0.320782, 0.318403, 0.357)
    std = (0.144, 0.151, 0.211, 0.195)
else:
    mean = (0.472455, 0.320782, 0.318403)
    std = (0.215084, 0.408135, 0.409993)  # 标准化参数

#data path
dataset = '../Vaihingen' # Potsdam
root_data = os.path.join(dataset,'slice_data')

exp_name = 'DSMSFNet'
save_dir = '../{}_files'.format(exp_name)
# The number of channels of the input image
model_experision = '3'
save_dir_model = os.path.join(save_dir, model_name+'_'+model_experision)

if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
if os.path.exists(save_dir_model) is False:
    os.mkdir(save_dir_model)

# train set
irrg_train_path = os.path.join(root_data, 'irrg_train')
dsm_train_path =  os.path.join(root_data, 'dsm_train')
gt_train_path = os.path.join(root_data, 'label_train')

# val set
irrg_val_path = os.path.join(dataset, 'irrg_val')
dsm_val_path = os.path.join(dataset, 'dsm_val')
gt_val_path = os.path.join(dataset, 'gt_val')

# test set
irrg_test_path = os.path.join(dataset, 'irrg_test')
dsm_test_path = os.path.join(dataset, 'dsm_test')
gt_test_path = os.path.join(dataset, 'gt_test')

# predict result
save_path = '../{}_resultsucnet_val/'.format(exp_name)

# label
p_gt = os.path.join(root_data, 'label')

# gray image of the predict result
p_pred = '../{}_resultsucnet_val/gray_big'.format(exp_name)

# save path
output = os.path.join(save_dir_model, './result_{}/'.format(model_name))
output_gray = os.path.join(save_dir_model, './result_gray_{}/'.format(model_name))
model_dir = os.path.join(save_dir_model, './pth_{}/'.format(model_name))
