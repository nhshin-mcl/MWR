from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import *
import os

######################### Feature Extraction #########################

def feature_extraction_global_regression(arg, train_data, test_data, model, device):

    features = {'train': [], 'test': []}

    batch_size = 50

    Images_train = ImageLoader(arg, train_data)
    dataloader_Images_train = DataLoader(Images_train, batch_size=batch_size, shuffle=False, num_workers=4)

    Images_test = ImageLoader(arg, test_data)
    dataloader_Images_test = DataLoader(Images_test, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_Images_train)):
            inputs = data
            inputs = inputs.to(device)

            outputs = torch.squeeze(model('extraction', x=inputs))

            outputs_numpy = outputs.cpu().detach().numpy().reshape(-1, 512)
            features['train'].extend(outputs_numpy)

        for i, data in enumerate(tqdm(dataloader_Images_test)):
            inputs = data
            inputs = inputs.to(device)

            outputs = torch.squeeze(model('extraction', x=inputs))

            outputs_numpy = outputs.cpu().detach().numpy().reshape(-1, 512)
            features['test'].extend(outputs_numpy)

    features = {'train': np.array(features['train']), 'test': np.array(features['test'])}

    return features

def feature_extraction_local_regression(arg, train_data, test_data, model, device):

    features = {'train': [[] for _ in range(arg.reg_num)], 'test': [[] for _ in range(arg.reg_num)]}

    batch_size = 50

    Images_train = ImageLoader(arg, train_data)
    dataloader_Images_train = DataLoader(Images_train, batch_size=batch_size, shuffle=False, num_workers=4)

    Images_test = ImageLoader(arg, test_data)
    dataloader_Images_test = DataLoader(Images_test, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for reg_idx in range(arg.reg_num):
            for i, data in enumerate(tqdm(dataloader_Images_train)):
                inputs = data
                inputs = inputs.to(device)

                outputs = torch.squeeze(model('extraction', x=inputs, idx=reg_idx))
                outputs_numpy = outputs.cpu().detach().numpy().reshape(-1, 512)
                features['train'][reg_idx].extend(outputs_numpy)

        for reg_idx in range(arg.reg_num):
            for i, data in enumerate(tqdm(dataloader_Images_test)):
                inputs = data
                inputs = inputs.to(device)

                outputs = torch.squeeze(model('extraction', x=inputs, idx=reg_idx))
                outputs_numpy = outputs.cpu().detach().numpy().reshape(-1, 512)
                features['test'][reg_idx].extend(outputs_numpy)

    features = {'train': np.array(features['train']), 'test': np.array(features['test'])}

    return features

######################### Global Reference Selection #########################
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

######################### Local Reference Selection #########################
def select_reference_local_regression(arg, train_data, pair_idx, loss, reg_bound, limit=5):
    # ref vector
    refer_idx, refer_pair_idx = [[] for _ in range(len(reg_bound))], [[] for _ in range(len(reg_bound))]

    data_age = np.array(train_data['age']).astype('int')
    data_age_max = data_age.max()
    lb_list, up_list = get_age_bounds(int(data_age_max), np.unique(data_age), arg.tau)

    for i in range(90):
        for r in range(len(reg_bound)):
            if (i >= lb_list[reg_bound[r][0]]) and (i <= reg_bound[r][1]):

                ref_image = train_data.loc[np.array(train_data['age']).astype('int') == i]
                idx = train_data.loc[np.array(train_data['age']).astype('int') == i].index.to_numpy()

                if len(ref_image) > 0:
                    loss_tmp = np.array(loss[r][idx].tolist()).flatten().tolist()

                    if len(loss_tmp) < limit:
                        idx_tmp = np.random.choice(np.arange(len(loss_tmp)), limit)
                    else:
                        idx_tmp = np.argsort(loss_tmp)[:limit]

                    row, column = idx_tmp // len(np.array(loss[r])[idx[0]]), idx_tmp % len(np.array(loss[r])[idx[0]])

                    y_1_idx_tmp, y_2_idx_tmp = idx[row], np.array(pair_idx[r])[idx[0]][column]

                    refer_idx[r].append(y_1_idx_tmp.tolist())
                    refer_pair_idx[r].append(y_2_idx_tmp.tolist())

                else:
                    refer_idx[r].append([])
                    refer_pair_idx[r].append([])

            else:
                refer_idx[r].append([])
                refer_pair_idx[r].append([])

    return refer_idx, refer_pair_idx

######################### Make best pairs #########################
def get_gt(age_y1, age_y2, age_x):
    mean = (age_y1 + age_y2) / 2
    tau = (mean - age_y1)

    if age_x <= age_y1:
        gt = -1.

    elif age_x >= age_y2:
        gt = 1.

    else:
        gt = (age_x - mean) / (tau + 1e-10)

    return gt

def get_best_pairs_global_regression(arg, train_data, test_data, features, model, device, load):

    data_age_max = int(train_data['age'].max())
    data_age = train_data['age'].to_numpy().astype('int')
    data_test_age = test_data['age'].to_numpy().astype('int')
    lb_list, up_list = get_age_bounds(int(data_age_max), np.unique(data_age), arg.tau)

    feat_train = torch.as_tensor(features['train'], dtype=torch.float32).to(device).squeeze()
    feat_test = torch.as_tensor(features['train'], dtype=torch.float32).to(device).squeeze()

    loss_total = []
    index_y2_total = []

    for i, age_y1 in enumerate(tqdm(data_age)):

        age_y2 = up_list[age_y1]
        index_y2 = np.where(data_age == age_y2)[0]
        index_y2_total.append(index_y2)
        index_test = np.where((data_test_age >= (age_y1 - 6)) & (data_test_age <= (age_y2 + 6)))[0]

        loss_tmp = []
        batch_size = 5000

        gt_total = []

        if load == False:

            for j in index_test:
                age_test = data_test_age[j]

                age_y1_log, age_y2_log, age_test_log = np.log(age_y1), np.log(age_y2), np.log(age_test)
                gt = get_gt(age_y1_log, age_y2_log, age_test_log)

                gt_total.append(gt)

            gt_total = np.array(gt_total).repeat(len(index_y2))

            feat_y1 = feat_train[i].repeat(len(index_y2), 1).reshape(-1, 512, 1, 1)
            feat_y2 = feat_train[index_y2].reshape(-1, 512, 1, 1)

            for index in index_test:

                feat_x = feat_test[index].repeat(1, len(index_y2)).reshape(-1, 512, 1, 1)

                for k in range(0, len(feat_y1), batch_size):
                    batch = min(batch_size, len(feat_y1))
                    outputs = model('test', x_1_1=feat_y1[k:k+batch], x_1_2=feat_y2[k:k+batch], x_2=feat_x[k:k+batch])
                    loss_tmp.extend(outputs.squeeze().cpu().detach().numpy().reshape(-1))

            outputs = np.array(loss_tmp)
            loss_total.append(np.power((outputs - gt_total), 2).reshape(len(index_test), len(index_y2)).mean(axis=0))

    return loss_total, index_y2_total

def get_best_pairs_local_regression(arg, train_data, test_data, features, reg_bound, model, device, load):

    data_age_max = int(train_data['age'].max())
    data_age = train_data['age'].to_numpy().astype('int')
    data_test_age = test_data['age'].to_numpy().astype('int')
    lb_list, up_list = get_age_bounds(int(data_age_max), np.unique(data_age), arg.tau)

    feat_train = torch.as_tensor(features['train'], dtype=torch.float32).to(device).squeeze()
    feat_test = torch.as_tensor(features['train'], dtype=torch.float32).to(device).squeeze()

    loss_total = [[] for _ in range(len(reg_bound))]
    index_y2_total = [[] for _ in range(len(reg_bound))]

    for i, age_y1 in enumerate(tqdm(data_age)):
        for r in range(len(reg_bound)):
            if (age_y1 >= lb_list[reg_bound[r][0]]) and (age_y1 <= reg_bound[r][1]):

                max_age = reg_bound[r][1] + 6
                min_age = reg_bound[r][0] - 6

                age_y2 = up_list[age_y1]
                index_y2 = np.where(data_age == age_y2)[0]
                index_y2_total[r].append(index_y2)
                index_test = np.where((data_test_age >= (max(age_y1 - 6, min_age))) & (data_test_age <= (min(age_y2 + 6, max_age))))[0]

                loss_tmp = []
                batch_size = 5000

                gt_total = []

                if load == False:

                    for j in index_test:
                        age_test = data_test_age[j]

                        age_y1_log, age_y2_log, age_test_log = np.log(age_y1), np.log(age_y2), np.log(age_test)
                        gt = get_gt(age_y1_log, age_y2_log, age_test_log)

                        gt_total.append(gt)

                    gt_total = np.array(gt_total).repeat(len(index_y2))

                    feat_y1 = feat_train[r][i].repeat(len(index_y2), 1).reshape(-1, 512, 1, 1)
                    feat_y2 = feat_train[r][index_y2].reshape(-1, 512, 1, 1)

                    for index in index_test:

                        feat_x = feat_test[r][index].repeat(1, len(index_y2)).reshape(-1, 512, 1, 1)

                        for k in range(0, len(feat_y1), batch_size):
                            batch = min(batch_size, len(feat_y1))
                            outputs = model('test', x_1_1=feat_y1[k:k+batch], x_1_2=feat_y2[k:k+batch], x_2=feat_x[k:k+batch], idx=r)
                            loss_tmp.extend(outputs.squeeze().cpu().detach().numpy().reshape(-1))

                    outputs = np.array(loss_tmp)
                    loss_total[r].append(np.power((outputs - gt_total), 2).reshape(len(index_test), len(index_y2)).mean(axis=0))

            else:
                loss_total[r].append([])
                index_y2_total[r].append([])

    return loss_total, index_y2_total

######################### MWR ########################
def initial_prediction(train_data, test_data, features, n_neighbors):

    f_tr, f_te = torch.as_tensor(features['train']), torch.as_tensor(features['test'])
    tr_age = torch.as_tensor(train_data['age'])

    cdist = torch.cdist(f_te, f_tr)
    distances, indices = torch.topk(cdist, n_neighbors, dim=1, largest=False)

    indices = indices.cpu().detach().numpy()
    init_pred = (tr_age[indices].float().mean(dim=1) + 0.5).int()

    return init_pred.cpu().detach().numpy()

def MWR_global_regression(arg, train_data, test_data, features, refer_idx, refer_idx_pair, initial_prediction, model, device):
    test_data['kmeans_age'] = initial_prediction

    lb_list, up_list = get_age_bounds(int(train_data['age'].max()), np.unique(train_data['age']), arg.tau)

    f_tr = torch.as_tensor(features['train']).to(device)
    f_te = torch.as_tensor(features['test']).to(device)

    pred_age = []
    max_iter = 10
    train_data_age_unique = np.unique(train_data['age'])
    memory=np.zeros(shape=(len(test_data), max_iter), dtype=np.int)

    with torch.no_grad():
        for i in tqdm(range(len(test_data)), "Test"):

            age = abs(int(test_data['kmeans_age'].iloc[i]))
            iteration = 0

            while True:
                lb_age = int(np.argsort(np.abs(np.log(age) - np.log(train_data_age_unique) - arg.tau/2))[0])
                lb_age = int(train_data_age_unique[lb_age])
                up_age = int(up_list[lb_age])

                lb_age, up_age = np.array([lb_age]), np.array([up_age])

                idx_1_final = [refer_idx[lb_age[tmp]] for tmp in range(len(lb_age))]
                idx_2_final = [refer_idx_pair[lb_age[tmp]] for tmp in range(len(lb_age))]

                idx_1_final = np.array(sum(idx_1_final, []))
                idx_2_final = np.array(sum(idx_2_final, []))

                idx_1_final = np.array(idx_1_final).reshape(-1)
                idx_2_final = np.array(idx_2_final).reshape(-1)

                test_duplicate = [i] * len(idx_1_final)
                feature_1, feature_2, feature_test = f_tr[idx_1_final].reshape(-1, 512, 1, 1), f_tr[idx_2_final].reshape(-1, 512, 1, 1), \
                                                     f_te[test_duplicate].reshape(-1, 512, 1, 1)

                outputs = model('test', x_1_1=feature_1, x_1_2=feature_2, x_2=feature_test)
                outputs = outputs.squeeze().cpu().detach().numpy().reshape(-1)

                up_age = np.array([up_age[tmp].repeat(len(refer_idx_pair[lb_age[tmp]])) for tmp in range(len(up_age))]).reshape(-1)
                lb_age = np.array([lb_age[tmp].repeat(len(refer_idx[lb_age[tmp]])) for tmp in range(len(lb_age))]).reshape(-1)

                mean = (np.log(up_age) + np.log(lb_age)) / 2
                tau = abs(mean - np.log(lb_age))

                refined_age = np.mean([np.exp(outputs[k] * tau[k] + mean[k]) for k in range(len(outputs))])

                if (max_iter == iteration) or (int(refined_age + 0.5) == age):
                    age = int(refined_age + 0.5)
                    memory[i, iteration:] = age
                    pred_age.append(age)
                    break

                else:
                    age = int(refined_age + 0.5)
                    memory[i, iteration] = age
                    iteration += 1

    return pred_age

def MWR_local_regression(arg, train_data, test_data, features, refer_idx, refer_idx_pair, global_prediction, reg_bound, model, device):
    pred_age_final = []
    train_data_age_unique = np.unique(train_data['age'])
    train_data_age_total = train_data['age'].to_numpy().reshape(-1)

    f_tr = torch.as_tensor(features['train']).to(device)
    f_te = torch.as_tensor(features['test']).to(device)

    lb_list_global, up_list_global = get_age_bounds(int(train_data['age'].max()), np.unique(train_data['age']), arg.tau)
    lb_list_total, up_list_total = [], []
    lb_list_half_total, up_list_half_total = [], []

    for reg_num_tmp in range(arg.reg_num):
        max_age = up_list_global[reg_bound[reg_num_tmp][1]]
        min_age = lb_list_global[reg_bound[reg_num_tmp][0]]
        train_age_list = np.where((train_data_age_total >= min_age) & (train_data_age_total <= max_age))[0]
        train_age_list = train_data_age_total[train_age_list]
        lb_list, up_list = get_age_bounds(int(train_age_list.max()), np.unique(train_age_list), arg.tau)
        lb_list_half, up_list_half = get_age_bounds(int(train_age_list.max()), np.unique(train_age_list), arg.tau / 2)

        lb_list_total.append(lb_list), up_list_total.append(up_list)
        lb_list_half_total.append(lb_list_half), up_list_half_total.append(up_list_half)

    with torch.no_grad():

        max_local_iter = 10
        memory = np.zeros(shape=(len(test_data), max_local_iter))

        lb_final_total, up_final_total, reg_idx = [], [], []

        for i in tqdm(range(0, len(test_data))):

            reg_num_list = []
            refine_list = []
            iteration = 0

            age = int(global_prediction[i])
            for tmp in range(arg.reg_num):
                if age in np.arange(reg_bound[tmp][0], reg_bound[tmp][1] + 1):
                    reg_num_list.append(tmp)

            while True:
                for tmp_idx, reg_num in enumerate(reg_num_list):

                    lb_age = int(np.argsort(np.abs(np.log(age) - np.log(train_data_age_unique) - arg.tau / 2))[0])
                    lb_age = int(train_data_age_unique[lb_age])
                    up_age = int(up_list_total[reg_num][lb_age])

                    if tmp_idx == 0:
                        lb_final = lb_age
                        up_final = up_age
                        reg_idx_final = reg_num

                    lb_age, up_age = np.array([lb_age]), np.array([up_age])

                    invalid_lb = np.setdiff1d(lb_age, (train_data_age_unique).astype('int'), True)
                    invalid_up = np.setdiff1d(up_age, (train_data_age_unique).astype('int'), True)

                    if len(invalid_lb) != 0 or len(invalid_up) != 0:
                        invalid_idx_lb, invalid_idx_up = [], []

                        for invalid_age in invalid_lb:
                            invalid_idx_lb.append(list(np.where(invalid_age == lb_age)[0]))
                        for invalid_age in invalid_up:
                            invalid_idx_up.append(list(np.where(invalid_age == up_age)[0]))

                        invalid_idx = sum(invalid_idx_lb, []) + sum(invalid_idx_up, [])

                        lb_age = np.delete(lb_age, invalid_idx)
                        up_age = np.delete(up_age, invalid_idx)

                    idx_1_final = [refer_idx[reg_num][lb_age[tmp]] for tmp in range(len(lb_age))]
                    idx_2_final = [refer_idx_pair[reg_num][lb_age[tmp]] for tmp in range(len(up_age))]

                    idx_1_final = np.array(sum(idx_1_final, []))
                    idx_2_final = np.array(sum(idx_2_final, []))

                    if len(idx_1_final) == 0:
                        idx_1_final = [refer_idx[reg_num][lb_age[tmp] - 1] for tmp in range(len(lb_age))]
                        lb_age = lb_age - 1
                    if len(idx_2_final) == 0:
                        idx_2_final = [refer_idx_pair[reg_num][lb_age[tmp] + 1] for tmp in range(len(up_age))]
                        up_age = up_age + 1

                    idx_1_final = np.array(idx_1_final).reshape(-1)
                    idx_2_final = np.array(idx_2_final).reshape(-1)

                    test_duplicate = [i] * len(idx_1_final)

                    feature_1, feature_2, feature_test = f_tr[reg_num][idx_1_final].reshape(-1, 512, 1, 1), f_tr[reg_num][idx_2_final].reshape(-1, 512, 1, 1), \
                                                         f_te[reg_num][test_duplicate].reshape(-1, 512, 1, 1)

                    outputs = model('test', x_1_1=feature_1, x_1_2=feature_2, x_2=feature_test, idx=reg_num)
                    outputs = outputs.squeeze().cpu().detach().numpy().reshape(-1)

                    up_age = np.array([up_age[tmp].repeat(len(refer_idx_pair[reg_num][lb_age[tmp]])) for tmp in range(len(up_age))]).reshape(-1)
                    lb_age = np.array([lb_age[tmp].repeat(len(refer_idx[reg_num][lb_age[tmp]])) for tmp in range(len(lb_age))]).reshape(-1)

                    mean = (np.log(up_age) + np.log(lb_age)) / 2
                    tau = abs(mean - np.log(lb_age))

                    refined_age_tmp = np.mean([np.exp(outputs[k] * (tau[k]) + mean[k]) for k in range(len(outputs))])

                    if sum(outputs == 1) == len(outputs):
                        refined_age_tmp = up_age.max()
                    if sum(outputs == -1) == len(outputs):
                        refined_age_tmp = lb_age.min()

                    refine_list.append(refined_age_tmp)

                refined_age = np.array(refine_list).mean()

                if (max_local_iter == (iteration + 1)) or (int(refined_age + 0.5) == age):
                    age = int(refined_age + 0.5)
                    memory[i, iteration:] = age
                    pred_age_final.append(age)

                    lb_final_total.append(lb_final)
                    up_final_total.append(up_final)
                    reg_idx.append(reg_idx_final)
                    break
                else:
                    age = int(refined_age + 0.5)
                    memory[i, iteration] = age
                    reg_num_list = []
                    refine_list = []

                    for tmp in range(arg.reg_num):
                        if age in np.arange(reg_bound[tmp][0], reg_bound[tmp][1] + 1):
                            reg_num_list.append(tmp)

                    iteration += 1

    return pred_age_final

######################################### Result Viz #########################################
def get_results(arg, test_data, pred_age):

    Error = np.array(abs(np.array(pred_age) - test_data['age']), dtype=np.float32)
    CS = np.array(abs(np.array(pred_age) - test_data['age']) <= 5, dtype=np.float32)

    if arg.dataset != 'clap2015':
        print('Dataset: %s, Setting: %s, MAE: %.2f, CS: %.3f'%(arg.dataset, arg.experiment_setting, Error.mean(), CS.mean()))
    else:
        Eps_clap = 1 - np.mean((1 / np.exp(np.square(np.subtract(pred_age, test_data['age'])) / (2 * np.square(test_data['stdv'])))))
        print('Dataset: %s, Setting: %s, MAE: %.2f, CS: %.3f, Eps: %.4f' % (arg.dataset, arg.experiment_setting, Error.mean(), CS.mean(), Eps_clap.mean()))


