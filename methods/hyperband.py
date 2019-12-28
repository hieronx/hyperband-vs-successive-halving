# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html
# hyperband method

import math
import numpy as np
from .bandit import Bandit


class Hyperband(Bandit):
    def __init__(self, benchmark, params, max_iter, eta, log_fn, args):
        super(Hyperband, self).__init__(benchmark, params, max_iter, eta, log_fn)

    def tune(self):
        best_hyperparameters = {}
        best_loss = math.inf

        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s))

            # initial number of iterations to run configurations for
            r = self.max_iter * self.eta**(-s)

            hs = self.create_hyperparameterspace(s, self.params)
            T = [hs.sample_configuration() for i in range(n)]

            for i in range(s+1):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta ** (i)
                val_losses = []

                for t in T:
                    train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.benchmark.run(
                        iterations=int(r_i), hyperparameters=t)

                    val_losses.append(train_loss)

                    self.save_results('hb', self.max_iter, self.eta, t.get('lr'), s, n_i, r_i, train_loss, train_accuracy,
                                      val_loss, val_accuracy, test_loss, test_accuracy)

                best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]

                if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                    best_loss = val_losses[best_ix[0]]
                    best_hyperparameters = T[best_ix[0]]

                T = [T[i] for i in best_ix]

        return (best_loss, best_hyperparameters)
