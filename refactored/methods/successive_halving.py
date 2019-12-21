# successive halving method

import math
import numpy as np
from .bandit import Bandit


class Successive_halving(Bandit):
    def __init__(self, benchmark, params, max_iter, eta):
        super(Successive_halving, self).__init__(
            benchmark, params, max_iter, eta)

    def tune(self):
        best_hyperparameters = {}
        best_loss = math.inf

        # initial number of configurations
        n = int(math.ceil(int(self.max_iter) * 2))
        r = self.max_iter  # initial number of iterations to run configurations for

        seed = 4
        hs = self.create_hyperparameterspace(seed, self.params)
        T = [hs.sample_configuration() for i in range(n)]

        for i in range(4):
            n_i = n * 2**(-i)
            r_i = r * 2**(i)
            val_losses = []

            for t in T:
                train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.benchmark.run(
                    iterations=int(r_i), hyperparameters=t)

                val_losses.append(train_loss)

                self.save_results(t.get('lr'), 4, n_i, r_i, train_loss, train_accuracy,
                                  val_loss, val_accuracy, test_loss, test_accuracy)

            best_ix = np.argsort(val_losses)[0:int(n_i / 2)]

            if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                best_loss = val_losses[best_ix[0]]
                best_hyperparameters = T[best_ix[0]]

            T = [T[i] for i in best_ix]

        return (best_loss, best_hyperparameters)
