# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html
# hyperband method

import math
import numpy as np
from .bandit import Bandit
import sys
from tqdm import tqdm


class Hyperband(Bandit):
    def __init__(self, benchmark, params, max_iter, eta, seed, filename, save, visualize_lr_schedule):
        super(Hyperband, self).__init__(
            benchmark, params, max_iter, eta, seed, filename, save, visualize_lr_schedule)

    def tune(self):
        if self.save:
            self.save_meta()

        total_resources, configs, unique_configs = self.get_total_info(
            hb=True, reuse=False)
        print('\nTotal resources: %s , Configs to be trained: %s , Unique hyperparameter configs: %s \n' %
              (total_resources, configs, unique_configs))

        best_hyperparameters = {}
        best_loss = math.inf
        pbar = tqdm(total=configs, position=0, desc='Progress |')

        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = self.get_n(s)

            # initial number of iterations to run configurations for
            r = self.get_r(s)

            hs = self.create_hyperparameterspace(self.seed+s, self.params)
            T = [hs.sample_configuration() for i in range(n)]

            current_hyperparameters, current_loss = self.sh_loop(
                T, n, r, s, pbar=pbar)

            if current_loss < best_loss:
                best_hyperparameters = current_hyperparameters
                best_loss = current_loss

        # pbar.close()
        return (best_loss, best_hyperparameters)
