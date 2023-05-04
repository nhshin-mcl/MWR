import os
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader

from configs.config_v1 import ConfigV1 as Config
from network.network_utils import build_model
from network.optimizer_utils import get_optimizer, get_scheduler
from dataloaders import morph
from utils.util import load_model, save_model, extract_features, find_kNN, select_pair_geometric_random, get_absolute_score, get_results, get_age_bounds



def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    net = build_model(cfg)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        net = net.cuda()

    optimizer = get_optimizer(cfg, net)
    lr_scheduler = get_scheduler(cfg, optimizer)

    if cfg.dataset_name == 'morph':

        train_dataset = morph.MorphTrain(cfg=cfg, tau=cfg.tau, dataset_dir=cfg.dataset_root)
        test_ref_dataset = morph.MorphRefSampling(cfg=cfg, tau=cfg.tau, dataset_dir=cfg.dataset_root)
        test_dataset = morph.MorphTest(cfg=cfg, dataset_dir=cfg.dataset_root)

        test_ref_loader = DataLoader(test_ref_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

    else:
        raise ValueError(f'Undefined database ({cfg.dataset_name}) has been given')

    if cfg.load | cfg.imdb_wiki_pretrain:
        load_model(cfg, net, optimizer=optimizer, load_optim_params=True)

    if lr_scheduler:
       lr_scheduler.step()

    net.eval()
    mae, cs = test_random(cfg, net, test_ref_loader, test_loader)
    sys.stdout.write(f'Init: [MAE: {mae:.4f}] [CS: {cs:.4f}]] \n')

    best_mae = mae
    best_cs = cs
    best_mae_epoch = -1

    for epoch in range(0, cfg.epoch):
        net.train()
        train_dataset.get_pair_lists()
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

        train_loss = train(cfg, net, optimizer, train_loader, epoch)

        print('\nEpoch: %d, MSE Loss: %.4f'%(epoch+1, train_loss))

        if ((epoch + 1) == 1) | ((epoch + 1) >= cfg.start_eval):

            net.eval()
            mae, cs = test_random(cfg, net, test_ref_loader, test_loader)
            sys.stdout.write(f'Epoch: {epoch+1}, [MAE: {mae:.4f}] [CS: {cs:.4f}]] \n')

            if best_mae > mae:
                best_mae = mae
                best_cs = cs
                best_mae_epoch = epoch + 1
                best_total_results = [mae, cs]
                save_model(cfg, net, optimizer, epoch, best_total_results, criterion='mae')

        print('lr: %.6f' % (optimizer.param_groups[0]['lr']))
        if lr_scheduler:
            lr_scheduler.step()

    print('Training End')
    print('Epoch: %d\tMAE: %.4f\tCS: %.4f' % (best_mae_epoch, best_mae, best_cs))


def train(cfg, net, optimizer, data_loader, epoch):
    avg_loss = 0

    for idx, sample in enumerate(data_loader):

        base_min_image, base_max_image, test_image = sample['base_min_image'], sample['base_max_image'], sample['test_image']
        labels = sample['label']

        base_min_image, base_max_image, test_image = base_min_image.cuda(), base_max_image.cuda(), test_image.cuda()

        image_cat = torch.cat([base_min_image, base_max_image, test_image], dim=0)

        optimizer.zero_grad()
        base_min_f, base_max_f, test_f = net('extraction', {'img': image_cat}).split(len(base_min_image))
        out = net('comparison', {'base_min_f': base_min_f, 'base_max_f': base_max_f, 'test_f': test_f})
        out = out.squeeze()
        labels = labels.float().cuda()

        loss = torch.nn.MSELoss()(out, labels)

        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        sys.stdout.write(f'\r[Epoch {epoch + 1}/{cfg.epoch}] [Batch {idx + 1}/{len(data_loader)}] [Train Loss: {loss.item():.4f}]')
    return avg_loss/(idx+1)

def test_random(cfg, net, ref_data_loader, data_loader):
    net.eval()

    lower_bounds, upper_bounds = get_age_bounds(int(ref_data_loader.dataset.df_base['age'].max()), np.unique(ref_data_loader.dataset.df_base['age']), cfg.tau)
    ref_age_unique = np.unique(ref_data_loader.dataset.df_base['age'])

    y1_list = ref_data_loader.dataset.base_min_age_np
    y2_list = ref_data_loader.dataset.base_max_age_np

    pred_total = []

    with torch.no_grad():

        ############################### Extract Reference & Test Features ###############################

        base_f_torch = extract_features(net, ref_data_loader, data_type='ref')
        test_f_torch = extract_features(net, data_loader, data_type='test')

        ############################### Initial Prediction ###############################

        vals, inds = find_kNN(queries=test_f_torch, samples=base_f_torch, k=cfg.k_neighbors)
        #init_preds = ref_data_loader.dataset.df_base['age'].values[inds.flatten().cpu()].reshape(-1, cfg.k_neighbors).mean(axis=1)
        init_preds = ref_data_loader.dataset.base_age_np[inds.flatten().cpu()].reshape(-1, cfg.k_neighbors).mean(axis=1)
        init_preds = (init_preds + 0.5).astype('int')

        ############################### MWR Process ###############################

        for idx, test_f in enumerate(test_f_torch):
            if idx % 1 == 0:
                sys.stdout.write(f'\rTesting... [{idx + 1}/{len(test_f_torch)}]')

            pred = init_preds[idx]
            y1, y2, y1_age, y2_age = select_pair_geometric_random(cfg=cfg, pred=pred, y1_list=y1_list, y2_list=y2_list, lower_bounds=lower_bounds, upper_bounds=upper_bounds, ref_age_unique=ref_age_unique)

            for _ in range(cfg.max_mwr_iter):

                y1 = ref_data_loader.dataset.base_min_idx_np[y1]
                y2 = ref_data_loader.dataset.base_max_idx_np[y2]

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = net('comparison', {'base_min_f': base_f_torch[y1], 'base_max_f': base_f_torch[y2], 'test_f': test_f.expand_as(base_f_torch[y1])})
                    out = out.squeeze()

                pred_mwr = get_absolute_score(y1_age=y1_age, y2_age=y2_age, pred_rho=out)
                y1, y2, y1_age, y2_age = select_pair_geometric_random(cfg=cfg, pred=pred, y1_list=y1_list, y2_list=y2_list, lower_bounds=lower_bounds, upper_bounds=upper_bounds, ref_age_unique=ref_age_unique)

                if pred_mwr == pred:
                    pred = pred_mwr
                    break

                else:
                    pred = pred_mwr

            pred_total.append(pred)

    print('\nMWR result')
    mae, cs = get_results(cfg, data_loader.dataset.df_test, np.array(pred_total), is_return=True)
    return mae, cs




if __name__ == "__main__":
    cfg = Config()
    main(cfg)