import albumentations
import cv2
import numpy as np


train_transform = albumentations.Compose([
        albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
            albumentations.GaussianBlur(),
        ]),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])

p = '/media/dyzy/DATA/aacode/code_eyes_Cgirl/data_slice_DRIVE/image_train/34.png'
img1 = cv2.imread(p)
img1 = np.stack((img1[:, :, 1], img1[:, :, 1], img1[:, :, 1]), axis=-1)
img2 = train_transform(image=img1)['image']
print('ok')