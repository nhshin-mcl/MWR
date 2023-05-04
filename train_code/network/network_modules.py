import numpy as np
from utils.util import get_rho, get_age_bounds



class PairGenerator:
    def __init__(self,train_data, tau=0.2):
        self.tau = tau
        self.data = train_data

    def get_ternary_mwr_pairs(self, rank):

        rank = np.array(rank)

        sample_idx = [[] for _ in range(3)]
        sample_label = []

        rank_unique, rank_num = np.unique(rank, return_counts=True)
        rank_min, rank_max = rank_unique.min(), rank_unique.max()

        idx = np.arange(len(rank))
        np.random.shuffle(idx)

        delta = self.tau - 0.05

        lower_bounds, upper_bounds = get_age_bounds(int(rank_max), rank_unique, self.tau)
        lower_bounds_delta, upper_bounds_delta = get_age_bounds(int(rank_max), rank_unique, delta)

        for min_base_idx in idx:
            min_base_rank = rank[min_base_idx]

            #max_base_rank = rank_unique[np.abs(np.log(rank_unique) - (np.log(min_base_rank) + self.tau)).argmin()]
            max_base_rank = upper_bounds[min_base_rank]
            max_base_idx = np.where(max_base_rank == rank)[0]
            max_base_idx = np.random.choice(max_base_idx)

            while True:

                flag = np.random.rand()
                if flag < 0.1:
                    start = rank_min
                    end = max(rank_min, lower_bounds_delta[int(min_base_rank)])
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif (flag >= 0.1) & (flag < 0.3):
                    start = max(rank_min, lower_bounds_delta[int(min_base_rank)])
                    end = max(rank_min, upper_bounds_delta[int(min_base_rank)])
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif (flag >= 0.3) & (flag < 0.7):
                    start = max(rank_min, min_base_rank)
                    end = min(rank_max, max_base_rank)
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif (flag >= 0.7) & (flag < 0.9):
                    start = max(rank_min, lower_bounds_delta[int(max_base_rank)])
                    end = min(rank_max, upper_bounds_delta[int(max_base_rank)])
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif flag >= 0.9:
                    start = min(rank_max, upper_bounds_delta[int(max_base_rank)])
                    end = rank_max
                    test_rank = np.where((rank_unique > start) & (rank_unique <= end))[0]

                if len(test_rank) == 0:
                    continue

                else:
                    test_rank = np.random.choice(test_rank)
                    test_rank = rank_unique[test_rank]

                    test_idx = np.where(rank == test_rank)[0]

                    if len(test_idx) > 0:

                        label = get_rho(np.array([min_base_rank]), np.array([max_base_rank]), np.array([test_rank]))[0]
                        break

            test_idx = np.random.choice(test_idx)

            sample_idx[0].append(min_base_idx)
            sample_idx[1].append(max_base_idx)
            sample_idx[2].append(test_idx)
            sample_label.append(label)

        permute = np.arange(len(sample_idx[0]))
        np.random.shuffle(permute)
        sample_idx_min = np.array(sample_idx[0])[permute]
        sample_idx_max = np.array(sample_idx[1])[permute]
        sample_idx_test = np.array(sample_idx[2])[permute]
        sample_label = np.array(sample_label)[permute]

        return sample_idx_min, sample_idx_max, sample_idx_test, sample_label

