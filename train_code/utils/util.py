import os
import sys
from datetime import datetime
from collections import OrderedDict
import math

import numpy as np
from tqdm import tqdm

import torch

def get_current_time():
    _now = datetime.now()
    _now = str(_now)[:-7]
    return _now

def load_model(args, net, optimizer=None, load_optim_params=False):
    checkpoint = torch.load(args.init_model, map_location=torch.device("cuda:%s" % (0) if torch.cuda.is_available() else "cpu"))
    model_dict = net.state_dict()

    new_model_state_dict = OrderedDict()
    for k, v in model_dict.items():
        if k in checkpoint['model_state_dict'].keys():
            new_model_state_dict[k] = checkpoint['model_state_dict'][k]
            #print(f'Loaded\t{k}')
        else:
            new_model_state_dict[k] = v
            print(f'Not Loaded\t{k}')
    net.load_state_dict(new_model_state_dict)

    print("=> loaded checkpoint '{}'".format(args.init_model))

    if load_optim_params == True:

        optimizer_dict = optimizer.state_dict()
        optimizer_dict.update(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(optimizer_dict)

        print("=> loaded optimizer params '{}'".format(args.init_model))

def save_model(args, net, optimizer, epoch, results, criterion):

    mae, cs = results[0], results[1]

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_folder + '/' + '%s_Epoch_%d'%(criterion, epoch) + '_MAE_%.4f_CS_%.4f' % (mae, cs) + '.pth'))
    print('Saved model to ' + args.save_folder + '/' + '%s_Epoch_%d'%(criterion, epoch) + '_MAE_%.4f_CS_%.4f' % (mae, cs) + '.pth')

def tensor2np(tensor):
    numpy_data = tensor.cpu().detach().numpy()
    return numpy_data

def extract_features(net, data_loader, data_type):
    f_torch = []

    for idx, sample in enumerate(data_loader):
        if idx % 1 == 0:
            sys.stdout.write(f'\rExtract {data_type} Features... [{idx + 1}/{len(data_loader)}]')

        image = sample[f'{data_type}_image']
        image = image.cuda()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            f = net('extraction', {'img': image})

        f_torch.append(f)

    return torch.cat(f_torch)

def select_pair_geometric_random(cfg, pred, y1_list, y2_list, lower_bounds, upper_bounds, ref_age_unique):
    y1_age = int(np.argsort(np.abs(np.log(pred) - np.log(ref_age_unique) - cfg.tau / 2))[0])
    y1_age = int(ref_age_unique[y1_age])
    y2_age = int(upper_bounds[y1_age])

    y1_age, y2_age = np.array([y1_age]), np.array([y2_age])

    y1 = np.where(y1_list == y1_age.squeeze())[0]
    y1 = np.array(y1).reshape(-1)

    return y1, y1, y1_age, y2_age

def select_pair_geometric(cfg, pred, y1_list, y2_list, lower_bounds, upper_bounds, ref_age_unique):

    y1_age = int(np.argsort(np.abs(np.log(pred) - np.log(ref_age_unique) - cfg.tau / 2))[0])
    y1_age = int(ref_age_unique[y1_age])
    y2_age = int(upper_bounds[y1_age])

    y1_age, y2_age = np.array([y1_age]), np.array([y2_age])

    y1 = [y1_list[y1_age[tmp]] for tmp in range(len(y1_age))]
    y2 = [y2_list[y1_age[tmp]] for tmp in range(len(y1_age))]

    y1 = np.array(sum(y1, []))
    y2 = np.array(sum(y2, []))

    y1 = np.array(y1).reshape(-1)
    y2 = np.array(y2).reshape(-1)

    return y1, y2, y1_age, y2_age

def select_reference_global_regression(train_data, pair_idx, loss, limit=1):
    # ref vector
    refer_idx, refer_pair_idx = [], []

    for i in range(90):
        ref_image = train_data.loc[train_data['age'] == i]
        idx = train_data.loc[train_data['age'] == i].index.to_numpy()

        if len(ref_image) > 0:
            loss_tmp = np.array(loss[idx].tolist()).flatten().tolist()

            if len(loss_tmp) < limit:
                idx_tmp = np.random.choice(np.arange(len(loss_tmp)), limit)
            else:
                idx_tmp = np.argsort(loss_tmp)[:limit]

            row, column = idx_tmp // len(np.array(loss)[idx[0]]), idx_tmp % len(np.array(loss)[idx[0]])

            y_1_idx_tmp, y_2_idx_tmp = idx[row], np.array(pair_idx)[idx[0]][column]

            refer_idx.append(y_1_idx_tmp.tolist())
            refer_pair_idx.append(y_2_idx_tmp.tolist())

        else:
            refer_idx.append([])
            refer_pair_idx.append([])

    return refer_idx, refer_pair_idx

def get_best_pairs_global_regression(arg, train_data, test_data, features, model, load):

    data_age_max = int(train_data['age'].max())
    data_age = train_data['age'].to_numpy().astype('int')
    data_test_age = test_data['age'].to_numpy().astype('int')
    lb_list, up_list = get_age_bounds(int(data_age_max), np.unique(data_age), arg.tau)

    feat_train = features.squeeze()
    feat_test = features.squeeze()

    loss_total = []
    index_y2_total = []

    for i, age_y1 in enumerate(tqdm(data_age)):

        age_y2 = up_list[age_y1]
        index_y2 = np.where(data_age == age_y2)[0]
        index_y2_total.append(index_y2)
        index_test = np.where((data_test_age >= (age_y1 - 6)) & (data_test_age <= (age_y2 + 6)))[0]

        loss_tmp = []
        batch_size = 5000

        if load == False:

            gt_total = get_rho(np.array([age_y1] * len(index_test)), np.array([age_y2] * len(index_test)), data_test_age[index_test])
            gt_total = np.array(gt_total).repeat(len(index_y2))

            feat_y1 = feat_train[i].repeat(len(index_y2), 1).reshape(-1, 512, 1, 1)
            feat_y2 = feat_train[index_y2].reshape(-1, 512, 1, 1)

            for index in index_test:

                feat_x = feat_test[index].repeat(1, len(index_y2)).reshape(-1, 512, 1, 1)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    for k in range(0, len(feat_y1), batch_size):
                        batch = min(batch_size, len(feat_y1))
                        outputs = model('comparison', {'base_min_f': feat_y1[k:k + batch], 'base_max_f': feat_y2[k:k + batch], 'test_f': feat_x[k:k + batch]})
                        loss_tmp.extend(outputs.squeeze().cpu().detach().numpy().reshape(-1))

            outputs = np.array(loss_tmp)
            loss_total.append(np.power((outputs - gt_total), 2).reshape(len(index_test), len(index_y2)).mean(axis=0))

    return np.array(loss_total), index_y2_total



def find_kNN(queries, samples, k=1, metric='L2', check_time=False):

    queries = queries.squeeze()
    samples = samples.squeeze()

    if len(queries.shape) == 2:
        queries = queries.view(1, queries.shape[0], queries.shape[1])
    if len(samples.shape) == 2:
        samples = samples.view(1, samples.shape[0], samples.shape[1])

    dist_mat = -torch.cdist(queries, samples)  # BxNxM
    vals, inds = torch.topk(dist_mat, k, dim=-1)

    return vals[0], inds[0]

def get_rho(min_base, max_base, test, geometric=True):

    if geometric:
        min_base = np.log(min_base)
        max_base = np.log(max_base)
        test = np.log(test)

    mean = (max_base + min_base) / 2
    tau = np.abs(mean - min_base)

    rhos = np.array([-1. if test[tmp] < min_base[tmp]
                      else 1. if test[tmp] > max_base[tmp]
                      else (test[tmp] - mean[tmp]) / (tau[tmp] + 1e-10) for tmp in range(len(min_base))], dtype=np.float32)

    return rhos

def get_absolute_score(y1_age, y2_age, pred_rho):

    mean = (np.log(y2_age) + np.log(y1_age)) / 2
    tau = abs(mean - np.log(y1_age))

    absolute_scores = np.exp(tensor2np(pred_rho) * tau + mean).mean()

    return int(absolute_scores + 0.5)

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

def get_results(arg, test_data, pred_age, is_return=False):

    Error = np.array(abs(np.array(pred_age) - test_data['age']), dtype=np.float32)
    CS = np.array(abs(np.array(pred_age) - test_data['age']) <= 5, dtype=np.float32)

    if arg.dataset_name != 'clap2015':
        print('Dataset: %s, Setting: %s, MAE: %.2f, CS: %.3f'%(arg.dataset_name, arg.training_scheme, Error.mean(), CS.mean()))
    else:
        Eps_clap = 1 - np.mean((1 / np.exp(np.square(np.subtract(pred_age, test_data['age'])) / (2 * np.square(test_data['stdv'])))))
        print('Dataset: %s, Setting: %s, MAE: %.2f, CS: %.3f, Eps: %.4f' % (arg.dataset, arg.experiment_setting, Error.mean(), CS.mean(), Eps_clap.mean()))

    if is_return:
        return Error.mean(), CS.mean()

