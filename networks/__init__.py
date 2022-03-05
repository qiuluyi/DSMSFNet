# from networks.DeepLab_v3 import DeepLabv3
from networks.unet_model import unet
from networks.res_unet import Res_UNet_50, Res_UNet_34
from networks.res_unet_longconv import Res_UNet_34 as Res_UNet_34_longconv
from networks.res_unet_longconv import Res_UNet_50 as Res_UNet_50_longconv
from networks.res_unet_attention import Res_UNet_34 as Res_UNet_34_att
from networks.res_unet_attention import Res_UNet_50 as Res_UNet_50_att
# from networks.setr.SETR import SETR_PUP_S
from networks.segnet import segnet
from networks.deeplabv3_plus import DeepLabv3_plus
# from  networks.CamDeeplabv3plus import CamDeeplabv3plus

from networks.unet_sfam import unet_sfam
from networks.unet_att import unet_att
from networks.deeplab.DSMSFNet import DSMSFNet
# if __name__ == '__main__':
#     import torch
#     import time
#     x = torch.rand((1, 3, 256, 256))
#     model = DeepLabv3_plus(2)
#     t1 = time.time()
#     y = model(x)
#     print(time.time()-t1)
#     x = x.cuda()
#     model.cuda()
#     t2 = time.time()
#     z = model(x)
#     print(time.time()-t2)