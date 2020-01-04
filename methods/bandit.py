# base class for the hyperband and successive halving method

from tqdm import tqdm
import math
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import os.path
from os import path


class Bandit:

    def __init__(self, benchmark, params, max_iter=81, eta=3, seed=2020, filename='', save=True):
        self.benchmark = benchmark
        self.params = params
        self.filename = filename
        self.save = save
        self.max_iter = max_iter  # maximum iterations/epochs per configuration
        self.eta = eta  # defines downsampling rate (default=3)
        self.seed = seed

        # number of unique executions of Successive Halving (minus one)
        self.s_max = int(self.logeta(self.max_iter))

        # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
        self.B = (self.s_max+1)*self.max_iter

    def logeta(self, x):
        return math.log(x) / math.log(self.eta)

    def get_n(self, s):
        return int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s))

    def get_r(self, s):
        return self.max_iter * self.eta**(-s)

    def get_sh_loop_info(self, n, r, s, reuse=False):
        resource = 0
        configs = 0
        prev_ri = 0

        for i in range(s+1):
            n_i = n * self.eta**(-i)
            r_i = (r * self.eta ** (i)) - prev_ri

            if reuse:
                prev_ri = r_i

            resource += n_i * r_i
            configs += n_i

        print('Bracket %s : %s resources' % (s, int(resource)))
        return resource, configs

    # configs are not unique, these are the configs trained from scratch each time
    def get_total_info(self, hb=True, reuse=False):
        total_resource = 0
        total_configs = 0
        unique_configs = 0
        s = self.s_max

        # both sh and hb have an outerloop of s_max + 1
        outer_loop = self.s_max + 1

        for x in reversed(range(outer_loop)):
            if hb:
                s = x

            # initial number of configurations
            n = self.get_n(s)
            # initial number of iterations to run configurations for
            r = self.get_r(s)

            unique_configs += n
            resource, configs = self.get_sh_loop_info(n, r, s, reuse=reuse)

            total_resource += resource
            total_configs += configs

        return round(total_resource, 2), int(total_configs), int(unique_configs)

    # a run of successive halving for (n, r)
    def sh_loop(self, T, n, r, s, pbar=None):
        best_hyperparameters = {}
        best_loss = math.inf

        for i in range(s+1):
            n_i = n * self.eta**(-i)
            r_i = r * self.eta ** (i)
            val_losses = []

            for t in T:
                train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.benchmark.run(
                    iterations=r_i, hyperparameters=t)

                val_losses.append(val_loss)

                if self.save:
                    self.save_results(t.get('lr'), s, n_i, r_i, train_loss,
                                      train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy)

                if pbar:
                    pbar.update(1)

            best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]

            if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                best_loss = val_losses[best_ix[0]]
                best_hyperparameters = T[best_ix[0]]

            T = [T[i] for i in best_ix]

        return best_hyperparameters, best_loss

    # create a hyperparameter generator with a seed
    def create_hyperparameterspace(self, seed, params):
        cs = CS.ConfigurationSpace(seed=seed)

        for configs in params:
            if configs[1] is configs[2]:
                hp = CSH.CategoricalHyperparameter(
                    name=configs[0], choices=[configs[1]])
            else:
                hp = CSH.UniformFloatHyperparameter(
                    name=configs[0], lower=configs[1], upper=configs[2], log=configs[3])

            cs.add_hyperparameter(hp)

        return cs

    # save meta data
    def save_meta(self):
        bench_meta = self.benchmark.get_meta()
        name = "./results/" + self.filename + ".csv.meta"
        line = 'seed : ' + str(self.seed) + '\n' + 'methods : ' + self.__class__.__name__ + '\n' + 'eta : ' + str(self.eta) + '\n' + 'dataset : ' + \
            bench_meta['dataset'] + '\n' + 'data_shape (channel, height, width) : ' + ' '.join(map(str, list(bench_meta['tensor_shape']))) + '\n' + 'training_size : ' + \
            str(bench_meta['size_train']) + '\n' + \
            'validation_size : ' + \
            str(bench_meta['size_val']) + '\n' + 'test_size : ' + \
            str(bench_meta['size_test']) + '\n' + 'model : ' + bench_meta[
                'model'] + '\n' + 'params (name, lower, upper, logsampling): ' + str(self.params) + '\n' + \
            'batch_size : ' + \
            str(bench_meta['bs']) + '\n' 'mini_iterations : ' + \
            str(bench_meta['mini_iterations'])

        if not path.exists(name):
            with open(name, 'w') as f:
                f.write(line)

    # save results
    def save_results(self, lr, bracket, n_i, r_i, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        name = "./results/" + self.filename + ".csv"

        if not path.exists(name):
            header = 'max_iter,learning_rate,bracket,n_i,r_i,train_loss,train_accuracy,val_loss,val_accuracy,test_loss,test_accuracy\n'
            with open(name, 'w') as f:
                f.write(header)

        line = str(self.max_iter)+','+str(lr)+','+str(bracket) + ',' + str(n_i)+',' + str(r_i)+',' + str(train_loss) + ',' + str(train_accuracy) + ',' + str(val_loss) + \
            ',' + str(val_accuracy)+',' + str(test_loss) + \
            ',' + str(test_accuracy) + '\n'

        with open(name, 'a') as f:
            f.write(line)
