# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html
# hyperband method

import math
import numpy as np
from .bandit import Bandit
import sys
from tqdm import tqdm


class Hyperband(Bandit):
    def __init__(self, benchmark, params, max_iter, eta, seed, filename, save):
        super(Hyperband, self).__init__(
            benchmark, params, max_iter, eta, seed, filename, save)

    # configs are not unique, these are th configs trained
    def get_total_info(self):
        resource = 0
        configs = 0
        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s))

            # initial number of iterations to run configurations for
            r = self.max_iter * self.eta**(-s)

            for i in range(s + 1):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta ** (i)

                resource += r_i * n_i
                configs += n_i

        return round(resource, 2), int(configs)

    def tune(self):
        if self.save:
            self.save_meta()

        total_resources, configs = self.get_total_info()
        print('Total resources: %s \n' % (total_resources))

        best_hyperparameters = {}
        best_loss = math.inf
        resource = 0
        pbar = tqdm(total=configs, position=0, desc='Progress |')

        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s))

            # initial number of iterations to run configurations for
            r = self.max_iter * self.eta**(-s)

            hs = self.create_hyperparameterspace(self.seed+s, self.params)
            T = [hs.sample_configuration() for i in range(n)]

            z = 0
            for i in range(s+1):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta ** (i)
                val_losses = []
                z += int(n_i * r_i)

                for t in T:
                    resource += r_i
                    train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.benchmark.run(
                        iterations=r_i, hyperparameters=t)

                    val_losses.append(val_loss)

                    if self.save:
                        self.save_results(t.get('lr'), s, n_i, r_i, train_loss,
                                          train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)
                    pbar.update(1)

                best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]

                if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                    best_loss = val_losses[best_ix[0]]
                    best_hyperparameters = T[best_ix[0]]

                T = [T[i] for i in best_ix]
            # print(z)
        pbar.close()
        return (best_loss, best_hyperparameters)
