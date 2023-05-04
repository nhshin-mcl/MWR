import os
from utils.util import get_current_time


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


class ConfigV1:
    def __init__(self):

        # dataset
        self.dataset_name = 'morph'
        self.dataset_root = f'/hdd1/Dataset/AgeEstimation/'
        self.training_scheme = 'RS_partial'
        self.fold = 0

        # network
        self.backbone = 'vgg16bn'
        self.model_name = 'MWR'

        # exp description
        self.exp_name = f'Back_{self.backbone}_M_{self.model_name}'
        self.sampled_refs = ' '
        self.gamma_errors = ' '

        #self.image_size = 256
        #self.patch_size = 224
        self.batch_size = 16
        self.start_iter = 0
        self.start_eval = 80

        self.tau = 0.2
        self.k_neighbors = 5
        self.max_mwr_iter = 10
        self.geometric_tau = True # Using geometric scheme
        self.half = True # For half precision
        self.mwr_sampling = False # Sampling data from datalist to obtain a gamma error list

        # optimization
        self.optim = 'Adam'
        self.scheduler = 'none'
        self.lr = 1e-4
        self.weight_decay = 0
        self.epoch = 300

        # misc
        self.num_workers = 0
        self.gpu = '1'

        # logging
        self.save_folder = '/hdd1/2023/2022CVPR_code_publish/results/results_mwr'
        self.save_folder = f'{self.save_folder}/{self.dataset_name}/{self.model_name}/{self.exp_name}_T{self.tau:.2f}_{get_current_time()}'
        os.makedirs(self.save_folder, exist_ok=True)
        self.log_configs()

        # Path of imdb-wiki pretrained parameter
        self.imdb_wiki_pretrain = False
        if self.imdb_wiki_pretrain:
            self.init_model = '/hdd1/AgeEstimation_Pytorch/pretrained/bn_vgg16_multi_pairs_v2/lr_1e-04_fixed_geometric_0.2/Epoch_0035.pth'

        # Path of finetuned parameter for evaluation
        self.load = True
        if self.load:
            self.init_model = os.path.join('/hdd1/2023/2022CVPR_code_publish/ckpt/global', self.dataset_name, self.training_scheme, f'fold{self.fold}', f'{self.dataset_name}_{self.training_scheme.lower()}.pth')

            if self.mwr_sampling:
                self.sampled_refs = os.path.join('/hdd1/2023/2022CVPR_code_publish/ckpt/global', self.dataset_name, self.training_scheme, f'fold{self.fold}', f'{self.dataset_name}_{self.training_scheme.lower()}_global_sampled.csv')
                self.gamma_errors = os.path.join('/hdd1/2023/2022CVPR_code_publish/ckpt/global', self.dataset_name, self.training_scheme, f'fold{self.fold}', f'{self.dataset_name}_{self.training_scheme.lower()}_loss_total_sampled.npy')
            else:
                self.gamma_errors = os.path.join('/hdd1/2023/2022CVPR_code_publish/ckpt/global', self.dataset_name, self.training_scheme, f'fold{self.fold}', f'{self.dataset_name}_{self.training_scheme.lower()}_loss_total.npy')

    def log_configs(self, log_file='log.txt'):
        if os.path.exists(f'{self.save_folder}/{log_file}'):
            log_file = open(f'{self.save_folder}/{log_file}', 'a')
        else:
            log_file = open(f'{self.save_folder}/{log_file}', 'w')

        write_log(log_file, '------------ Options -------------')
        for k in vars(self):
            write_log(log_file, f'{str(k)}: {str(vars(self)[k])}')
        write_log(log_file, '-------------- End ----------------')

        log_file.close()
        return

if __name__=="__main__":
    c = ConfigV1()
    print('debug... ')


