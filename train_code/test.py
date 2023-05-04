import os
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader

from configs.config_v1 import ConfigV1 as Config
from network.network_utils import build_model
from network.optimizer_utils import get_optimizer, get_scheduler
from dataloaders import morph
from utils.util import load_model, extract_features, find_kNN, get_absolute_score, select_pair_geometric, get_age_bounds, select_reference_global_regression, \
    get_best_pairs_global_regression, get_results



def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    net = build_model(cfg)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        net = net.cuda()

    optimizer = get_optimizer(cfg, net)
    lr_scheduler = get_scheduler(cfg, optimizer)

    if cfg.dataset_name == 'morph':
        test_ref_dataset = morph.MorphRef(cfg=cfg, tau=cfg.tau, dataset_dir=cfg.dataset_root)
        test_dataset = morph.MorphTest(cfg=cfg, dataset_dir=cfg.dataset_root)

        test_ref_loader = DataLoader(test_ref_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

    else:
        raise ValueError(f'Undefined database ({cfg.dataset_name}) has been given')

    if cfg.load:
        load_model(cfg, net, optimizer=optimizer, load_optim_params=False)

    if lr_scheduler:
       lr_scheduler.step()

    net.eval()
    test(cfg, net, test_ref_loader, test_loader)

def test(cfg, net, ref_data_loader, data_loader):
    net.eval()

    train_data = ref_data_loader.dataset.df_base

    if os.path.isfile(cfg.gamma_errors):
        gamma_list = np.load(cfg.gamma_errors, allow_pickle=True)
        load_gamma_list = True
    else:
        load_gamma_list = False

    lower_bounds, upper_bounds = get_age_bounds(int(ref_data_loader.dataset.df_base['age'].max()), np.unique(ref_data_loader.dataset.df_base['age']), cfg.tau)
    ref_age_unique = np.unique(ref_data_loader.dataset.df_base['age'])

    pred_total = []
    with torch.no_grad():

        ############################### Extract Reference & Test Features ###############################

        base_f_torch = extract_features(net, ref_data_loader, data_type='ref')
        test_f_torch = extract_features(net, data_loader, data_type='test')

        ############################### Get Best Reference Pairs ##########################
        if load_gamma_list:
            _, pair_idx = get_best_pairs_global_regression(cfg, train_data, train_data, base_f_torch, net, load=load_gamma_list)
        else:
            gamma_list, pair_idx = get_best_pairs_global_regression(cfg, train_data, train_data, base_f_torch, net, load=load_gamma_list)
        y1_list, y2_list = select_reference_global_regression(train_data=train_data, pair_idx=pair_idx, loss=gamma_list, limit=1)

        ############################### Initial Prediction ###############################

        vals, inds = find_kNN(queries=test_f_torch, samples=base_f_torch, k=cfg.k_neighbors)
        init_preds = ref_data_loader.dataset.df_base['age'].values[inds.flatten().cpu()].reshape(-1, cfg.k_neighbors).mean(axis=1)
        init_preds = (init_preds + 0.5).astype('int')

        ############################### MWR Process ###############################

        for idx, test_f in enumerate(test_f_torch):
            if idx % 1 == 0:
                sys.stdout.write(f'\rTesting... [{idx + 1}/{len(test_f_torch)}]')

            pred = init_preds[idx]
            y1, y2, y1_age, y2_age = select_pair_geometric(cfg=cfg, pred=pred, y1_list=y1_list, y2_list=y2_list, lower_bounds=lower_bounds, upper_bounds=upper_bounds, ref_age_unique=ref_age_unique)

            for _ in range(cfg.max_mwr_iter):

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = net('comparison', {'base_min_f': base_f_torch[y1], 'base_max_f': base_f_torch[y2], 'test_f': test_f.expand_as(base_f_torch[y1])})
                    out = out.squeeze()

                pred_mwr = get_absolute_score(y1_age=y1_age, y2_age=y2_age, pred_rho=out)
                y1, y2, y1_age, y2_age = select_pair_geometric(cfg=cfg, pred=pred, y1_list=y1_list, y2_list=y2_list, lower_bounds=lower_bounds, upper_bounds=upper_bounds, ref_age_unique=ref_age_unique)

                if pred_mwr == pred:
                    pred = pred_mwr
                    break

                else:
                    pred = pred_mwr

            pred_total.append(pred)

    print('\nMWR result')
    get_results(cfg, data_loader.dataset.df_test, np.array(pred_total))


if __name__ == "__main__":
    cfg = Config()
    main(cfg)