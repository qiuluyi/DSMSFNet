from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from tools import utils

class IsprsSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=None,
                 split='train',
                 transform=None
                 ):
        self.split = split
        self._base_dir = base_dir

        # irrg image
        self.images = os.listdir(os.path.join(self._base_dir, 'irrg_{}'.format(self.split)))
        self.images = [os.path.join(self._base_dir, 'image_{}'.format(self.split), i).replace('\\', '/') for i in self.images]
        self.images = [i for i in self.images if i.endswith('.tif')]

        # dsm data
        self.nsdm = os.listdir(os.path.join(self._base_dir, 'dsm_{}'.format(self.split)))
        self.nsdm = [os.path.join(self._base_dir, 'dsm_{}'.format(self.split), i).replace('\\', '/')  for i in self.nsdm]
        self.nsdm = [i for i in self.nsdm if i.endswith('.jpg')]

        self.categories = [i.replace('image_{}'.format(self.split), 'label_{}'.format(self.split)) for i in self.images]

        self.transform = transform
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _ndsm, _target, _name = self._make_img_gt_point_pair(index)
        sample = {'image': _img,'nsdm':_ndsm, 'gt': _target }

        if self.transform is not None:
            sample = self.transform(sample)
        sample['name'] = _name
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        _img = utils.read_image(os.path.join(self.images[index])).astype(np.float32)
        _nsdm = utils.read_image(os.path.join(self.nsdm[index])).astype(np.float32)
        _nsdm= np.expand_dims(_nsdm,axis=2)
        _target = utils.read_image(os.path.join(self.categories[index]), 'gt').astype(np.int32)
        return _img,_nsdm , _target, os.path.join(self.images[index]).replace('\\', '/').split('/')[-1]

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

