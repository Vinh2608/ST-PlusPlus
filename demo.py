import tqdm
from utils import count_params, meanIOU, color_map
import torch
from PIL import Image
import numpy as np
import os
from dataset.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet

cmap = np.zeros((256, 3), dtype='uint8')
cmap[0] = np.array([253, 244, 157])
cmap[1] = np.array([159, 150, 119])
cmap[2] = np.array([76, 17, 87])

def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
model = model_zoo['deeplabv3plus']('resnet50', 3)

model.load_state_dict(torch.load('/home/aivn2023/Documents/BloodCell/ST++/ST-PlusPlus/outdir/models/dataset1/1_4/split_0/deeplabv3plus/resnet50/deeplabv3plus_resnet50_100.00.pth'))
model.eval()
img = Image.open('Dataset1/Image/train/097.bmp')
mask = Image.open('Dataset1/Masks/train/097.png')
img, mask = resize(img, mask, 128, (0.5, 2.0))
img, mask = crop(img, mask, 128)
img, mask = hflip(img, mask, p=0.5)
img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
img = transforms.RandomGrayscale(p=0.2)(img)
img = blur(img, p=0.5)
img, mask = cutout(img, mask, p=0.5)
# img = normalize(img)
# pred = model(img.unsqueeze(0), True)
# pred = torch.argmax(pred, dim=1).cpu()
# pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
# pred.putpalette(cmap)


img.save('./97.bmp')
mask.save('./97.png')
# closing all open windows 
