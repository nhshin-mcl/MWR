from Utils import *
import Network as models
import os
import pandas as pd
import torch


def global_regression(arg, train_data, test_data, sampling=False, sample_rate=1):

    device = torch.device("cuda:%s" % (arg.gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    ### Create network ###
    model = models.create_model(arg, arg.backbone)
    model.to(device)

    initial_model = os.path.join(arg.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s.pth'%
                                 (str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))
    print(initial_model)

    ### Load network parameters ###
    checkpoint = torch.load(initial_model, map_location=device)
    model_dict = model.state_dict()

    model_dict.update(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '{}'".format(initial_model))

    model.eval()

    ### Designate save path ###
    if sampling is True:

        train_data_sampled_path = os.path.join(arg.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_global_sampled.csv'%
                                     (str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))

        if os.path.exists(train_data_sampled_path) is False:

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

            train_data_sampled.to_csv(train_data_sampled_path)
            print('Save sampled datalist to', train_data_sampled_path)

        else:
            train_data_sampled = pd.read_csv(train_data_sampled_path)
            print('Load sampled datalist: ', train_data_sampled_path)

        train_data = train_data_sampled
        train_data['age'] = train_data['age'].to_numpy().astype('int')

        save_path = os.path.join(arg.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_loss_total_sampled.npy'%
                                 (str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))

        save_results_path = os.path.join(arg.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_global_top1_results_sampled.txt'%
                                 (str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))

    else:
        save_path = os.path.join(arg.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_loss_total.npy' % (
        str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))

        save_results_path = os.path.join(arg.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_global_top1_results.txt' % (
        str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))

    ### Get features ###
    features = feature_extraction_global_regression(arg, train_data, test_data, model, device)

    ### Make initial prediction ###
    init_pred = initial_prediction(train_data, test_data, features, 5)

    ### Get regression error ###
    if os.path.isfile(save_path) == True:
        loss_total, pair_idx = get_best_pairs_global_regression(arg, train_data, train_data, features, model, device, True)
        loss_total = np.load(save_path, allow_pickle=True)
    else:
        loss_total, pair_idx = get_best_pairs_global_regression(arg, train_data, train_data, features, model, device, False)
        np.save(save_path, loss_total)
        print('Best pair indices saved', save_path)

    ### Get best reference pairs ###
    refer_idx, refer_idx_pair = select_reference_global_regression(train_data, pair_idx, np.array(loss_total), limit=1)

    ### MWR ###
    pred = MWR_global_regression(arg, train_data, test_data, features, refer_idx, refer_idx_pair, init_pred, model, device)

    if os.path.isfile(save_results_path) == False:
        np.savetxt(save_results_path, np.array(pred).reshape(-1, 1))
        print('Global regression results saved', save_results_path)

    ### Viz results ###
    get_results(arg, test_data, np.array(pred))

def local_regression(arg, train_data, test_data, pth_path, en_single_results, reg_bound, ref_num=1, sampling=False, sample_rate=1):
    device = torch.device("cuda:%s" % (arg.gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    ### Create network ###
    model = models.create_model(arg, arg.backbone)
    model.to(device)

    ### Load network parameters ###
    initial_model = os.path.join(pth_path, '%s_%s_local.pth'%(arg.dataset, arg.experiment_setting.lower()))

    checkpoint = torch.load(initial_model, map_location=device)
    model_dict = model.state_dict()

    model_dict.update(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '{}'".format(initial_model))

    model.eval()

    ### Designate save path ###
    if sampling is True:

        train_data_sampled_path = os.path.join(arg.ckpt_dir, 'local', '%s/%s/fold%d/%s_%s_sampled_5p.csv' % (
            str(arg.dataset).upper(), arg.experiment_setting, arg.fold, arg.dataset, arg.experiment_setting))

        if os.path.exists(train_data_sampled_path) is False:

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

            train_data_sampled.to_csv(train_data_sampled_path)
            print('Save sampled datalist to', train_data_sampled_path)

        else:
            train_data_sampled = pd.read_csv(train_data_sampled_path)
            print('Load sampled datalist: ', train_data_sampled_path)

        train_data = train_data_sampled
        train_data['age'] = train_data['age'].to_numpy().astype('int')

    ### Get features ###
    features = feature_extraction_local_regression(arg, train_data, test_data, model, device)

    ### Designate save path ###
    if sampling is False:
        save_path = os.path.join(arg.ckpt_dir, arg.experiment_title, 'fold%d' % (arg.fold), 'loss_total_5p.npy')
    else:
        save_path = os.path.join(arg.ckpt_dir, arg.experiment_title, 'fold%d' % (arg.fold), 'loss_total_sampled_5p.npy')
    print('save path: ', save_path)

    ### Get regression error ###
    if os.path.isfile(save_path) == True:
        loss_total, index_y2_total = get_best_pairs_local_regression(arg, train_data, train_data, features, reg_bound, model, device, True)
        loss_total = np.load(save_path, allow_pickle=True)
    else:
        loss_total, index_y2_total = get_best_pairs_local_regression(arg, train_data, train_data, features, reg_bound, model, device, False)
        np.save(save_path, loss_total)

    ### Get best reference pairs ###
    refer_idx, refer_idx_pair = select_reference_local_regression(arg, train_data, index_y2_total, np.array(loss_total), reg_bound, limit=ref_num)

    ### MWR ###
    pred = MWR_local_regression(arg, train_data, test_data, features, refer_idx, refer_idx_pair, en_single_results, reg_bound, model, device)

    ### Viz result ###
    get_results(arg, test_data, np.array(pred))
