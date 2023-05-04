import os
from PIL import Image
import itertools

import torch

from dataloaders.data_utils import data_select
from dataloaders.data_utils import ImgAugTransform, ImgAugTransform_Test, ImgAugTransform_Test_Aug

import numpy as np
import pandas as pd

from network.network_modules import PairGenerator
from utils.util import get_age_bounds

from torch.utils.data import Dataset

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}



class MorphTrain(Dataset):
    def __init__(self, cfg, tau=5, dataset_dir='dataset/MORPH/'):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, cfg.dataset_name)

        train_data, _, reg_bound, sampling, sample_rate = data_select(cfg)
        self.df = train_data

        self.pg = PairGenerator(train_data=self.df, tau=tau)
        self.idx_min, self.idx_max, self.idx_test, self.label = None, None, None, None

        self.mean = torch.as_tensor(imagenet_stats['mean'], dtype=torch.float32)
        self.std = torch.as_tensor(imagenet_stats['mean'], dtype=torch.float32)

    def __getitem__(self, idx):
        base_min_name = os.path.join(self.image_dir, self.df['filename'].iloc[self.idx_min[idx]])
        base_max_name = os.path.join(self.image_dir, self.df['filename'].iloc[self.idx_max[idx]])
        test_name = os.path.join(self.image_dir, self.df['filename'].iloc[self.idx_test[idx]])

        base_min_age = self.df['age'].iloc[self.idx_min[idx]]
        base_max_age = self.df['age'].iloc[self.idx_max[idx]]
        test_age = self.df['age'].iloc[self.idx_test[idx]]

        label = self.label[idx]

        base_min_image = Image.open(base_min_name)
        base_max_image = Image.open(base_max_name)
        test_image = Image.open(test_name)

        base_min_image_trans = ImgAugTransform(base_min_image) / 255.
        base_max_image_trans = ImgAugTransform(base_max_image) / 255.
        test_image_trans = ImgAugTransform(test_image) / 255.

        base_min_image_trans = torch.from_numpy(np.transpose(base_min_image_trans, (2, 0, 1)).astype('float32'))
        base_max_image_trans = torch.from_numpy(np.transpose(base_max_image_trans, (2, 0, 1)).astype('float32'))
        test_image_trans = torch.from_numpy(np.transpose(test_image_trans, (2, 0, 1)).astype('float32'))

        base_min_image_trans.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        base_max_image_trans.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        test_image_trans.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        sample = dict()
        sample['base_min_impath'] = base_min_name
        sample['base_max_impath'] = base_max_name
        sample['test_impath'] = test_name

        sample['base_min_age'] = base_min_age
        sample['base_max_age'] = base_max_age
        sample['test_age'] = test_age

        sample['label'] = label

        sample['base_min_image_orig'] = np.array(base_min_image)
        sample['base_max_image_orig'] = np.array(base_max_image)
        sample['test_image_orig'] = np.array(test_image)

        sample['base_min_image'] = base_min_image_trans
        sample['base_max_image'] = base_max_image_trans
        sample['test_image'] = test_image_trans

        return sample

    def __len__(self):
        return len(self.df)

    def get_pair_lists(self):
        self.idx_min, self.idx_max, self.idx_test, self.label = self.pg.get_ternary_mwr_pairs(rank=self.df['age'])

class MorphTest(Dataset):
    def __init__(self, cfg, tau=5, dataset_dir='dataset/MORPH/'):
        self.dataset_dir = dataset_dir
        self.tau = tau
        self.image_dir = os.path.join(dataset_dir, cfg.dataset_name)

        _, test_data, reg_bound, sampling, sample_rate = data_select(cfg)
        self.df_test = test_data

    def __getitem__(self, idx):
        test_name = os.path.join(self.image_dir, self.df_test['filename'].iloc[idx])
        test_age = self.df_test['age'].iloc[idx]
        test_image = Image.open(test_name)
        test_image_trans = ImgAugTransform_Test(test_image).astype(np.float32) / 255.

        test_image_trans = torch.from_numpy(np.transpose(test_image_trans, (2, 0, 1)))

        dtype = test_image_trans.dtype
        mean = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=test_image_trans.device)
        std = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=test_image_trans.device)
        test_image_trans.sub_(mean[:, None, None]).div_(std[:, None, None])

        sample = dict()
        sample['test_impath'] = test_name
        sample['test_age'] = test_age
        sample['test_image_orig'] = np.array(test_image)
        sample['test_image'] = test_image_trans

        return sample

    def __len__(self):
        return len(self.df_test)

class MorphRef(Dataset):
    def __init__(self, cfg, tau=5, dataset_dir='dataset/MORPH/'):
        self.dataset_dir = dataset_dir
        self.tau = tau
        self.image_dir = os.path.join(dataset_dir, cfg.dataset_name)

        train_data, _, reg_bound, sampling, sample_rate = data_select(cfg)
        if cfg.mwr_sampling:

            if os.path.exists(cfg.sampled_refs):
                train_data = pd.read_csv(cfg.sampled_refs)
                print('Load sampled datalist: ', cfg.sampled_refs)
            else:
                data_train_age = train_data['age'].to_numpy().astype('int')
                data_train_age_unique = np.unique(data_train_age)

                sampled_idx = []

                ### Sample data from training set ###
                for tmp in data_train_age_unique:
                    valid_idx = np.where(data_train_age == tmp)[0]
                    sample_num = int(len(valid_idx) * sample_rate + 0.5)

                    if sample_num < 1:
                        sample_num = 1
                        idx_choice = np.random.choice(valid_idx, sample_num, replace=True)
                    else:
                        idx_choice = np.random.choice(valid_idx, sample_num, replace=False)

                    sampled_idx.append(idx_choice.tolist())

                sampled_idx = np.array(sum(sampled_idx, []))
                train_data_sampled = train_data.iloc[sampled_idx].reset_index(drop=True)

                train_data_sampled.to_csv(cfg.sampled_refs)
                print('Save sampled datalist to', cfg.sampled_refs)

        self.df_base = train_data

    def __getitem__(self, idx):
        ref_name = os.path.join(self.image_dir, self.df_base['filename'].iloc[idx])
        ref_image = Image.open(ref_name)
        ref_image_trans = ImgAugTransform_Test(ref_image).astype(np.float32) / 255.
        ref_image_trans = torch.from_numpy(np.transpose(ref_image_trans, (2, 0, 1)))

        dtype = ref_image_trans.dtype
        mean = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=ref_image_trans.device)
        std = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=ref_image_trans.device)
        ref_image_trans.sub_(mean[:, None, None]).div_(std[:, None, None])

        sample = dict()
        sample['ref_impath'] = ref_name
        sample['ref_image_orig'] = np.array(ref_image)
        sample['ref_image'] = ref_image_trans

        return sample

    def __len__(self):
        return len(self.df_base)


class MorphRefSampling(Dataset):
    def __init__(self, cfg, tau=5, dataset_dir='dataset/MORPH/'):
        self.dataset_dir = dataset_dir
        self.tau = tau
        self.image_dir = os.path.join(dataset_dir, cfg.dataset_name)

        train_data, _, reg_bound, sampling, sample_rate = data_select(cfg)
        self.df_base = train_data
        self.ref_sampling(pair_num=5)

    def __getitem__(self, idx):
        ref_name = os.path.join(self.image_dir, self.base_filename_np[idx])
        ref_image = Image.open(ref_name)
        ref_image_trans = ImgAugTransform_Test(ref_image).astype(np.float32) / 255.
        ref_image_trans = torch.from_numpy(np.transpose(ref_image_trans, (2, 0, 1)))

        dtype = ref_image_trans.dtype
        mean = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=ref_image_trans.device)
        std = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=ref_image_trans.device)
        ref_image_trans.sub_(mean[:, None, None]).div_(std[:, None, None])

        sample = dict()
        sample['ref_impath'] = ref_name
        sample['ref_image_orig'] = np.array(ref_image)
        sample['ref_image'] = ref_image_trans

        return sample

    def __len__(self):
        return len(self.base_filename_np)

    def ref_sampling(self, pair_num=5):
        print(f'Sampling References / Sampling Num {pair_num}')
        base_age = self.df_base['age'].values
        base_age_unique, base_age_num = np.unique(base_age, return_counts=True)
        lower_bounds, upper_bounds = get_age_bounds(int(base_age_unique.max()), base_age_unique, self.tau)

        base_min_idx_list = list()
        base_max_idx_list = list()

        base_min_age_list = list()
        base_max_age_list = list()

        for base_min_age in base_age_unique:

            base_min_idx = np.where(base_age == base_min_age)[0]

            base_max_age = upper_bounds[base_min_age]
            base_max_idx = np.where(base_max_age == base_age)[0]

            pair_list = np.array(list(itertools.product(base_min_idx, base_max_idx)))
            pair_idx = np.arange(len(pair_list))

            if len(pair_list) >= pair_num:
                pair_idx_selected = np.random.choice(pair_idx, pair_num, replace=False)
            else:
                pair_idx_selected = np.random.choice(pair_idx, pair_num, replace=True)

            base_min_idx_list.extend(pair_list[:, 0][pair_idx_selected].tolist())
            base_max_idx_list.extend(pair_list[:, 1][pair_idx_selected].tolist())

            base_min_age_list.extend([base_min_age] * pair_num)
            base_max_age_list.extend([base_max_age] * pair_num)

        base_min_idx_list = np.array(base_min_idx_list)
        base_max_idx_list = np.array(base_max_idx_list)

        self.base_min_age_np = np.array(base_min_age_list)
        self.base_max_age_np = np.array(base_max_age_list)

        base_unique_idx = np.unique(np.concatenate([base_min_idx_list, base_max_idx_list]))
        self.base_filename_np = self.df_base['filename'].iloc[base_unique_idx].values
        self.base_age_np = self.df_base['age'].iloc[base_unique_idx].values

        mapping = dict()
        for i, unique_idx in enumerate(base_unique_idx):
            mapping[str(unique_idx)] = i

        self.base_min_idx_np = np.array(list(map(lambda x: mapping[str(x)], base_min_idx_list)))
        self.base_max_idx_np = np.array(list(map(lambda x: mapping[str(x)], base_max_idx_list)))

        print(f'Sampling Finished / Total Pair Num {len(self.base_min_idx_np)}')

if __name__ == '__main__':
    from configs.config_v1 import ConfigV1 as Config
    from torch.utils.data import DataLoader

    pass