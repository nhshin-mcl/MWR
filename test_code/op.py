import argparse
from Prediction import *
from DataSelector import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--regression', type=str)
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--tau', type=float, default=0.2)

    parser.add_argument('--experiment_setting', type=str)
    parser.add_argument('--experiment_title', type=str)

    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--data_path', type=str, default='datalist')
    parser.add_argument('--im_path', type=str)
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()

    ####################################################################
    # Morph Setting
    # A: RS_partial, B: 3_Fold_BW, C: SE, D: RS
    # Example) python op.py --dataset morph --regression global --experiment_setting RS_partial --im_path your_image_path

    # clap2015 Setting
    # validation split / test split
    # Example) python op.py --dataset clap2015 --regression global --experiment_setting val --im_path your_image_path

    # CACD Setting
    # train split / validation split
    # Example) python op.py --dataset cacd --regression global --experiment_setting train --im_path your_image_path

    # UTK Setting
    # coral
    # Example) python op.py --dataset utk --regression global --experiment_setting coral --im_path your_image_path
    ####################################################################

    if args.regression == 'global':
        args.backbone = 'Global_Regressor'
    elif args.regression == 'local':
        args.backbone = 'Local_Regressor'

    for fold in range(5):

        args.fold = fold

        args.experiment_title = os.path.join(args.regression, str(args.dataset), str(args.experiment_setting))
        train_data, test_data, reg_bound, sampling, sample_rate = data_select(args)

        if args.regression == 'global':
            print(args)
            print('Global Regression')
            print('Age range partition: ', reg_bound)

            global_regression(args, train_data, test_data, sampling=sampling, sample_rate=sample_rate)

        elif args.regression == 'local':
            args.reg_num = len(reg_bound)

            print(args)
            print('Local Regression')
            print('Age range partition: ', reg_bound)

            pth_group_path = os.path.join(args.ckpt_dir, args.experiment_title, 'fold%d' % (args.fold))

            if sampling is False:
                results_path = os.path.join(args.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_global_top1_results.txt'%
                                            (str(args.dataset).upper(), args.experiment_setting, args.fold, args.dataset, args.experiment_setting))
            else:
                results_path = os.path.join(args.ckpt_dir, 'global', '%s/%s/fold%d/%s_%s_global_top1_results_sampled.txt' %
                                                (str(args.dataset).upper(), args.experiment_setting, args.fold, args.dataset, args.experiment_setting))

            print('Global Result: ', results_path)
            results = pd.read_csv(results_path, header=None)
            results = np.array(results.values.flatten(), dtype=np.int32)

            local_regression(args, train_data, test_data, pth_group_path, results, reg_bound, 1, sampling, sample_rate)



