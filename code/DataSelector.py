import numpy as np
import pandas as pd
import os

def data_select(arg):

    if (arg.dataset == 'clap2015') & (arg.experiment_setting == 'test'):

        total_data = pd.read_excel(os.path.join(arg.data_path, 'clap2015/total_2015.xlsx'))

        train_data = total_data[total_data['database'] != 'test'].reset_index(drop=True)
        test_data = total_data[total_data['database'] == 'test'].reset_index(drop=True)

        train_data['database'].loc[train_data['database'] == 'train'] = '2015/train'
        train_data['database'].loc[train_data['database'] == 'val'] = '2015/val'
        test_data['database'] = '2015/test'

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 18], [10, 29], [19, 44], [30, 62], [45, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (arg.dataset == 'clap2015') & (arg.experiment_setting == 'val'):

        total_data = pd.read_excel(os.path.join(arg.data_path, 'clap2015/total_2015.xlsx'))

        train_data = total_data[total_data['database'] == 'train'].reset_index(drop=True)
        test_data = total_data[total_data['database'] == 'val'].reset_index(drop=True)

        train_data['database'].loc[train_data['database'] == 'train'] = '2015/train'
        test_data['database'] = '2015/val'

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 18], [10, 29], [19, 44], [30, 62], [45, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (arg.dataset == 'morph') & (arg.experiment_setting == 'RS_partial'):

        sep = [' ', ',', ',', ',', ',']

        train_data = pd.read_csv(os.path.join(arg.data_path, 'morph/RS_partial/Setting_1_fold%d_train.txt' % arg.fold), sep=sep[arg.fold])
        test_data = pd.read_csv(os.path.join(arg.data_path, 'morph/RS_partial/Setting_1_fold%d_test.txt' % arg.fold), sep=sep[arg.fold])

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (arg.dataset == 'morph') & (arg.experiment_setting == '3_Fold_BW'):

        total_fold = [0, 1, 2]
        total_fold.pop(arg.fold)
        test_data_1 = pd.read_csv(os.path.join(arg.data_path, 'morph/3_Fold_BW/Setting_2_fold%d.txt' % (total_fold[0])), sep=' ')
        test_data_2 = pd.read_csv(os.path.join(arg.data_path, 'morph/3_Fold_BW/Setting_2_fold%d.txt' % (total_fold[1])), sep=' ')
        test_data = pd.concat([test_data_1, test_data_2]).reset_index(drop=True)
        train_data = pd.read_csv(os.path.join(arg.data_path, 'morph/3_Fold_BW/Setting_2_fold%d.txt' % (arg.fold)), sep=' ')

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = False
        sample_rate = -1

    elif (arg.dataset == 'morph') & (arg.experiment_setting == 'SE'):

        data = pd.read_csv(os.path.join(arg.data_path, 'morph/SE/Setting_4.txt'), sep='\t')
        train_data = data[data['fold'] != arg.fold].reset_index(drop=True)
        test_data = data[data['fold'] == arg.fold].reset_index(drop=True)

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = True
        sample_rate = 0.20

    elif (arg.dataset == 'morph') & (arg.experiment_setting == 'RS'):

        data = pd.read_csv(os.path.join(arg.data_path, 'morph/RS/Setting_3.txt'), sep='\t')
        train_data = data[data['fold'] != arg.fold].reset_index(drop=True)
        test_data = data[data['fold'] == arg.fold].reset_index(drop=True)

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 27], [22, 35], [28, 46], [36, 59], [47, int(age_max)]]

        sampling = True
        sample_rate = 0.20

    elif arg.dataset == 'cacd':

        total_data = pd.read_csv(os.path.join(arg.data_path, 'cacd/CACD.csv'))
        train_data = total_data.loc[total_data['fold'] == arg.experiment_setting].reset_index(drop=True)
        test_data = total_data.loc[total_data['fold'] == 'test'].reset_index(drop=True)

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 23], [19, 29], [24, 38], [30, 48], [39, int(age_max)]]

        if arg.experiment_setting == 'train':
            sampling = True
            sample_rate = 0.05

        elif arg.experiment_setting == 'val':
            sampling = False
            sample_rate = -1

    elif arg.dataset == 'utk':

        train_data = pd.read_csv(os.path.join(arg.data_path, 'utk/UTK_train_coral.csv'))
        test_data = pd.read_csv(os.path.join(arg.data_path, 'utk/UTK_test_coral.csv'))

        age_min, age_max = train_data['age'].min(), train_data['age'].max()

        age_group = [[int(age_min), 28], [24, 33], [29, 40], [34, 49], [41, int(age_max)]]

        sampling = True
        sample_rate = 0.50

    return train_data, test_data, age_group, sampling, sample_rate