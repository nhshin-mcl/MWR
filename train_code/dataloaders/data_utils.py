import os

import numpy as np
from imgaug import augmenters as iaa
import pandas as pd

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

def data_select(cfg):

    if (cfg.dataset_name == 'clap2015') & (cfg.training_scheme == 'test'):

        total_data = pd.read_excel(os.path.join(cfg.dataset_root, 'clap2015/total_2015.xlsx'))

        train_data = total_data[total_data['database'] != 'test'].reset_index(drop=True)
        test_data = total_data[total_data['database'] == 'test'].reset_index(drop=True)

        train_data['database'].loc[train_data['database'] == 'train'] = '2015/train'
        train_data['database'].loc[train_data['database'] == 'val'] = '2015/val'
        test_data['database'] = '2015/test'

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 18], [10, 29], [19, 44], [30, 62], [45, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (cfg.dataset_name == 'clap2015') & (cfg.training_scheme == 'val'):

        total_data = pd.read_excel(os.path.join(cfg.dataset_root, 'clap2015/total_2015.xlsx'))

        train_data = total_data[total_data['database'] == 'train'].reset_index(drop=True)
        test_data = total_data[total_data['database'] == 'val'].reset_index(drop=True)

        train_data['database'].loc[train_data['database'] == 'train'] = '2015/train'
        test_data['database'] = '2015/val'

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 18], [10, 29], [19, 44], [30, 62], [45, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (cfg.dataset_name == 'morph') & (cfg.training_scheme == 'RS_partial'):

        sep = [' ', ',', ',', ',', ',']

        train_data = pd.read_csv(os.path.join(cfg.dataset_root, 'datalist/morph/RS_partial/Setting_1_fold%d_train.txt' % cfg.fold), sep=sep[cfg.fold])
        test_data = pd.read_csv(os.path.join(cfg.dataset_root, 'datalist/morph/RS_partial/Setting_1_fold%d_test.txt' % cfg.fold), sep=sep[cfg.fold])

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (cfg.dataset_name == 'morph') & (cfg.training_scheme == '3_Fold_BW'):

        total_fold = [0, 1, 2]
        total_fold.pop(cfg.fold)
        test_data_1 = pd.read_csv(os.path.join(cfg.dataset_root, 'morph/3_Fold_BW/Setting_2_fold%d.txt' % (total_fold[0])), sep=' ')
        test_data_2 = pd.read_csv(os.path.join(cfg.dataset_root, 'morph/3_Fold_BW/Setting_2_fold%d.txt' % (total_fold[1])), sep=' ')
        test_data = pd.concat([test_data_1, test_data_2]).reset_index(drop=True)
        train_data = pd.read_csv(os.path.join(cfg.dataset_root, 'morph/3_Fold_BW/Setting_2_fold%d.txt' % (cfg.fold)), sep=' ')

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (cfg.dataset_name == 'morph') & (cfg.training_scheme == 'SE'):

        data = pd.read_csv(os.path.join(cfg.dataset_root, 'datalist/morph/SE/Setting_4.txt'), sep='\t')
        train_data = data[data['fold'] != cfg.fold].reset_index(drop=True)
        test_data = data[data['fold'] == cfg.fold].reset_index(drop=True)

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = True
        sample_rate = 0.20

    elif (cfg.dataset_name == 'morph') & (cfg.training_scheme == 'RS'):

        data = pd.read_csv(os.path.join(cfg.dataset_root, 'morph/RS/Setting_3.txt'), sep='\t')
        train_data = data[data['fold'] != cfg.fold].reset_index(drop=True)
        test_data = data[data['fold'] == cfg.fold].reset_index(drop=True)

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = True
        sample_rate = 0.20

    elif cfg.dataset_name == 'cacd':

        total_data = pd.read_csv(os.path.join(cfg.dataset_root, 'cacd/CACD.csv'))
        train_data = total_data.loc[total_data['fold'] == cfg.training_scheme].reset_index(drop=True)
        test_data = total_data.loc[total_data['fold'] == 'test'].reset_index(drop=True)

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 23], [19, 29], [24, 38], [30, 48], [39, int(age_max)]]

        if cfg.training_scheme == 'train':
            sampling = True
            sample_rate = 0.05

        elif cfg.training_scheme == 'val':
            sampling = False
            sample_rate = -1

    elif cfg.dataset_name == 'utk':

        train_data = pd.read_csv(os.path.join(cfg.dataset_root, 'utk/UTK_train_coral.csv'))
        test_data = pd.read_csv(os.path.join(cfg.dataset_root, 'utk/UTK_test_coral.csv'))

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 28], [24, 33], [29, 40], [34, 49], [41, int(age_max)]]

        sampling = True
        sample_rate = 0.50

    return train_data, test_data, age_group, sampling, sample_rate