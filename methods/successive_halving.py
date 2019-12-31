# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html
# successive halving method for most exploratory bracket

import math
import numpy as np
from .bandit import Bandit


class Successive_halving(Bandit):
    def __init__(self, benchmark, params, max_iter, eta, seed, filename):
        super(Successive_halving, self).__init__(
            benchmark, params, max_iter, eta, seed, filename)

    def get_total_resource(self):
        resource = 0
        s = self.s_max
        for _ in range(s+1):

            # initial number of configurations
            n = int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s))
            r = self.max_iter * self.eta ** (-s)

            T = n

            for i in range(self.s_max):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta**(i)

                for _ in range(T):
                    resource += r_i

                T = int(n_i / self.eta)

        return round(resource, 2)

    def tune(self):
        best_global_hyperparameters = {}
        best_global_loss = math.inf
        s = self.s_max
        p = 0
        for x in range(s):
            best_hyperparameters = {}
            best_loss = math.inf

            # initial number of configurations
            n = int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s))
            r = self.max_iter * self.eta ** (-s)

            hs = self.create_hyperparameterspace(self.seed+x, self.params)
            T = [hs.sample_configuration() for i in range(n)]

            for i in range(self.s_max+1):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta**(i)
                val_losses = []
                print(n_i, r_i)
                for t in T:
                    p += r_i
                    train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.benchmark.run(
                        iterations=r_i, hyperparameters=t)

                    val_losses.append(val_loss)

                    self.save_results(t.get('lr'), x, n_i, r_i, train_loss, train_accuracy,
                                      val_loss, val_accuracy, test_loss, test_accuracy)

                best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]

                if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                    best_loss = val_losses[best_ix[0]]
                    best_hyperparameters = T[best_ix[0]]

                T = [T[i] for i in best_ix]

            if best_loss < best_global_loss:
                best_global_hyperparameters = best_hyperparameters
                best_global_loss = best_loss
        print(p)
        print('Total resource : ', self.get_total_resource())
        return (best_global_loss, best_global_hyperparameters)
