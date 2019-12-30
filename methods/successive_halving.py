# successive halving method

import math
import numpy as np
from .bandit import Bandit


class Successive_halving(Bandit):
    def __init__(self, benchmark, params, max_iter, eta, log_fn, args):
        super(Successive_halving, self).__init__(benchmark, params, max_iter, eta, log_fn)
        self.args = args


    def tune(self):
        cumulative = 0
        best_global_hyperparameters = {}
        best_global_loss = math.inf

        for i in range(self.s_max+1):
            best_hyperparameters = {}
            best_loss = math.inf

            # initial number of configurations
            n = int(math.ceil(int(self.B / self.max_iter / (self.s_max+1)) * self.eta**self.s_max))

            r = self.max_iter * self.eta**(-self.s_max)

            hs = self.create_hyperparameterspace(self.args.seed, self.params)
            T = [hs.sample_configuration() for i in range(n)]
            
            for i in range(self.s_max):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta**(i)
                cumulative += int(r_i * n_i)
                val_losses = []

                for t in T:
                    train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.benchmark.run(
                        iterations=int(r_i), hyperparameters=t)

                    val_losses.append(train_loss)

                    self.save_results('sh', self.max_iter, self.eta, t.get('lr'), self.s_max, n_i, r_i, train_loss, train_accuracy,
                                    val_loss, val_accuracy, test_loss, test_accuracy)

                best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]

                if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                    best_loss = val_losses[best_ix[0]]
                    best_hyperparameters = T[best_ix[0]]

                T = [T[i] for i in best_ix]
            
            if best_loss < best_global_loss:
                best_global_hyperparameters = best_hyperparameters
                best_global_loss = best_loss

        return (best_global_loss, best_global_hyperparameters)
