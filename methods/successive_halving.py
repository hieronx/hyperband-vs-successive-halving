# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html
# successive halving method for most exploratory bracket

import math
import numpy as np
from .bandit import Bandit
from tqdm import tqdm
import math


class Successive_halving(Bandit):
    def __init__(self, benchmark, params, max_iter, eta, seed, filename, save):
        super(Successive_halving, self).__init__(
            benchmark, params, max_iter, eta, seed, filename, save)

    def tune(self):
        if self.save:
            self.save_meta()

        total_resources, configs, unique_configs = self.get_total_info(
            hb=False, reuse=False)
        print('\nTotal resources: %s , Configs to be trained: %s , Unique hyperparameter configs: %s \n' %
              (total_resources, configs, unique_configs))

        best_global_hyperparameters = {}
        best_global_loss = math.inf
        s = self.s_max

        pbar = tqdm(total=configs, position=0, desc='Progress |')

        for x in range(s+1):
            best_hyperparameters = {}
            best_loss = math.inf

            # initial number of configurations
            n = self.get_n(s)
            r = self.get_r(s)

            hs = self.create_hyperparameterspace(self.seed+x, self.params)
            T = [hs.sample_configuration() for i in range(n)]

            current_hyperparameters, current_loss = self.sh_loop(
                T, n, r, s, pbar=pbar)

        pbar.close()
        return (best_global_loss, best_global_hyperparameters)
