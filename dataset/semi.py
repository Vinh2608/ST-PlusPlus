from dataset.transform import crop, hflip, normalize, resize, blur, cutout, obtain_cutmix_box, cutout_circular_region
from dataset.transform import crop_img, hflip_img, resize_img, cutout_img
from albumentations.augmentations.domain_adaptation import FDA
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import ImageFilter
from copy import deepcopy
import yaml


class SemiDataset(Dataset):
    def read_img_pil(self, file_path):
        with Image.open(file_path) as img:
            return np.array(img)
        
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None, unreliable_image_paths=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.cfg = yaml.load(
            open('configs/' + self.name + '.yaml', "r"), Loader=yaml.Loader)

        self.pseudo_mask_path = pseudo_mask_path
        
        if unreliable_image_paths != None:
            self.unreliable_image_paths = unreliable_image_paths
            self.beta_limit = 0.01
            
            self.fda_transform = FDA(
                reference_images=unreliable_image_paths,
                beta_limit=self.beta_limit,
                read_fn=self.read_img_pil,
                p=0.5
            )

        else:
            self.unreliable_image_paths = None
        
        
        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * \
                math.ceil(len(self.unlabeled_ids) /
                          len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'val2':
                id_path = 'dataset/splits/%s/val2.txt' % name
            elif mode == 'label' or mode == 'consistency_training':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path
            elif mode == 'warm_up':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            self.class_values = [255, 128]

    
        
    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))
        # img_old = Image.open(os.path.join(self.root, id.split(' ')[0]))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # if self.name == 'dataset1':
        #     mean = [0.8775001,  0.77965562, 0.64973898]
        #     std = [0.20316671, 0.30358231, 0.21943319]
        # elif self.name == 'dataset2':
        #     mean = [0.82557683, 0.67072675, 0.68503826]
        #     std = [0.18739882, 0.2288236,  0.09994926]
        # elif self.name == 'lisc':
        #     mean = [0.84130926, 0.76650068,0.89864814]
        #     std = [0.15318868, 0.2175517, 0.11220778]
        # This is for dataset lisc only
        base_size = 128
        if self.name == 'lisc':
            img = img.resize((base_size, base_size), Image.BILINEAR)

        if self.mode == 'val' or self.mode == 'label' or self.mode == 'val2':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            # mask_old = Image.open(os.path.join(self.root, id.split(' ')[1]))
            # This is for LISC only
            if self.name == 'lisc':
                mask = mask.resize((base_size, base_size), Image.NEAREST)
            if self.name == 'dataset1':
                img = img.resize((128, 128), Image.NEAREST)
                mask = mask.resize((128,128), Image.NEAREST)
            if self.name == 'dataset2':
                img = img.resize((320, 320), Image.NEAREST)
                mask = mask.resize((320,320), Image.NEAREST)

            mask = np.array(mask)
            mask[mask == 0] = 0
            mask[mask == 255] = 1
            mask[mask == 128] = 2
            mask = Image.fromarray(mask)

            img, mask = normalize(img, mean, std, mask)
            return img, mask, id

        if self.mode == 'train' or self.mode == 'warm_up' or self.mode == 'consistency_training' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        mask = np.array(mask)
        msk_name = id.split(' ')[1]
        if msk_name == 'Dataset2/Mask/047.png':
            mask[mask == 7] = 0
        if msk_name == 'Dataset2/Mask/034.png':
            mask[mask == 7] = 0
            mask[mask == 45] = 0
            mask[mask == 25] = 0
            mask[mask == 44] = 0
            mask[mask == 42] = 0
            mask[mask == 2] = 0
        if msk_name == 'Dataset2/Mask/035.png':
            mask[mask == 3] = 0
        if msk_name == 'Dataset2/Mask/050.png':
            mask[mask == 9] = 0
        if msk_name == 'Dataset2/Mask/076.png':
            mask[mask == 37] = 0
            mask[mask == 50] = 0
            mask[mask == 24] = 0

        if self.name != 'raabin':
            mask[mask == 0] = 0
            mask[mask == 255] = 1
            mask[mask == 128] = 2
        elif self.name == 'raabin':
            mask[mask == 0] = 0
            mask[mask == 255] = 2
            mask[mask == 100] = 1

        mask = Image.fromarray(mask)
        # basic augmentation on all training images
        # base_size = 400 if self.name == 'pascal' else 2048
        if self.name == 'dataset1' or self.name == 'lisc':
            base_size = 128
        elif self.name == 'dataset2':
            base_size = 320
        elif self.name == 'raabin':
            base_size = 480
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train' or self.mode == 'warm_up':
            return normalize(img, mean, std, mask)

        img_weak, img_strong = deepcopy(img), deepcopy(img)

        # strong augmentation on unlabeled images
        if (self.mode == 'semi_train' and id in self.unlabeled_ids) or self.mode == 'consistency_training':
            if random.random() < self.cfg['rand_thres']:
                # img = img.filter(ImageFilter.MedianFilter(size=7))
                # for color-jtter only, contrast (2.0,2.0) for lisc
                if self.name == 'lisc':
                    img_strong = transforms.ColorJitter(
                        0.5, (2.0, 2.0), 0.5, 0.25)(img_strong)
                else:
                    img_strong = transforms.ColorJitter(
                        0.5, 0.5, 0.5, 0.25)(img_strong)

            img_strong = transforms.RandomGrayscale(p=0.2)(img_strong)
            img_strong = blur(img_strong, p=0.5)
            # cutmix_box = obtain_cutmix_box(img_strong.size[0], p=0.5)
            
            if self.unreliable_image_paths != None:
                transformed_data = self.fda_transform(image=np.array(img_strong))
                img_strong = Image.fromarray(transformed_data['image'])

            #img_strong, mask = cutout(img_strong, mask, p=0.5)
            img_strong, mask = cutout_circular_region(img_strong, mask, 15, 0.5)
            

        img_strong, mask = normalize(img_strong, mean, std, mask)

        if self.mode == 'semi_train':
            return img_strong, mask
        else:
            return normalize(img_weak, mean, std), img_strong  # , cutmix_box

    def __len__(self):
        return len(self.ids)
