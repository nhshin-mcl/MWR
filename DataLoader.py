from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from PIL import Image
import math

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def get_age_bounds(age, age_unique, tau=0.1, mode='geometric'):
    lb_list = []
    up_list = []

    if mode == 'geometric':

        for age_tmp in range(age + 1):

            lb_age = sum(np.arange(age) < age_tmp * math.exp(-tau))
            up_age = sum(np.arange(age) < age_tmp * math.exp(tau))

            lb_sub_unique = abs(age_unique - lb_age)
            up_sub_unique = abs(age_unique - up_age)

            lb_nearest = np.argsort(lb_sub_unique)
            up_nearest = np.argsort(up_sub_unique)

            lb_list.append(int(age_unique[lb_nearest[0]]))
            up_list.append(int(age_unique[up_nearest[0]]))

    elif mode == 'arithmetic':

        for age_tmp in range(age + 1):

            if age_tmp in age_unique:
                lb_age = sum(np.arange(age) < age_tmp - tau)
                up_age = sum(np.arange(age) < age_tmp + tau)

                lb_sub_unique = abs(age_unique - lb_age)
                up_sub_unique = abs(age_unique - up_age)

                lb_nearest = np.argsort(lb_sub_unique)
                up_nearest = np.argsort(up_sub_unique)

                lb_list.append(age_unique[lb_nearest[0]])
                up_list.append(age_unique[up_nearest[0]])

            else:
                lb_list.append(sum(np.arange(age) < age_tmp - tau))
                up_list.append(sum(np.arange(age) < age_tmp + tau))

    return lb_list, up_list

############################################ Img Aug ############################################

def ImgAugTransform(img):

    aug = iaa.Sequential([
        iaa.CropToFixedSize(width=224, height=224),
        iaa.Fliplr(0.5)
    ])

    img = np.array(img)
    img = aug(image=img)
    return img

def ImgAugTransform_Test(img):

    aug = iaa.Sequential([
            iaa.CropToFixedSize(width=224, height=224, position="center")
        ])

    img = np.array(img)
    img = aug(image=img)
    return img

def ImgAugTransform_Test_Aug(img):
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)

    aug = iaa.Sequential([
            iaa.CropToFixedSize(width=224, height=224, position="center"),
            iaa.Fliplr(0.5),
            sometimes(iaa.LogContrast(gain=(0.8, 1.2))),
            sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
        ])

    img = np.array(img)
    img = aug(image=img)
    return img

############################################ For Test ############################################

class ImageLoader(Dataset):
    def __init__(self, arg, data, img_size=256, aug=False):

        self.df = data
        self.img_size = img_size
        self.aug = aug

        self.img = []

        for i in range(len(self.df)):
            img_path = Path(arg.im_path, self.df['database'].iloc[i], self.df["filename"].iloc[i])
            self.img.append(str(img_path))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_path = self.img[idx]

        img = Image.open(str(img_path))
        img = img.resize((self.img_size, self.img_size))

        if self.aug == False:
            img = ImgAugTransform_Test(img).astype(np.float32) / 255.
        else:
            img = ImgAugTransform_Test_Aug(img).astype(np.float32) / 255.

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        dtype = img.dtype
        mean = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=img.device)
        std = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=img.device)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img