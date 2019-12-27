# base class for the hyperband and successive halving method

from tqdm import tqdm
import math
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import os.path
from os import path

class Bandit:

    def __init__(self, benchmark, params, max_iter=81, eta=3, filename=''):
        self.benchmark = benchmark
        self.params = params
        self.filename = filename

        self.max_iter = max_iter  # maximum iterations/epochs per configuration
        self.eta = eta  # defines downsampling rate (default=3)

        # number of unique executions of Successive Halving (minus one)
        def logeta(x): return math.log(x)/math.log(eta)
        self.s_max = int(logeta(max_iter))

        # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
        self.B = (self.s_max+1)*max_iter

    # create a hyperparameter generator
    def create_hyperparameterspace(self, seed, params):
        cs = CS.ConfigurationSpace(seed=seed)

        for configs in params:
            hp = CSH.UniformFloatHyperparameter(
                name=configs[0], lower=configs[1], upper=configs[2], log=configs[3])
            cs.add_hyperparameter(hp)

        return cs

    def save_results(self, optimizer, max_iter, eta, lr, bracket, n_i, r_i, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        name = "./results/" + self.filename + ".csv"

        if not path.exists(name):
            header = 'optimizer, max_iter, eta, learning_rate, bracket, n_i, r_i, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy\n'
            with open(name, 'w') as f:
                f.write(header)

        line = str(optimizer)+','+str(max_iter)+','+str(eta)+','+str(lr)+','+str(bracket) + ',' + str(n_i)+',' + str(r_i)+',' + str(train_loss) + ',' + str(train_accuracy) + ',' + str(val_loss) + \
            ',' + str(val_accuracy)+',' + str(test_loss) + \
            ',' + str(test_accuracy) + '\n'
            
        with open(name, 'a') as f:
            f.write(line)

        return None
