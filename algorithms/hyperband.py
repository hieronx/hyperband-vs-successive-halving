from tqdm import tqdm
from math import floor, ceil, log, inf
import numpy as np

# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html
class Hyperband:

    def __init__(self, model, max_iter = 81, eta = 3):
        self.model = model

        self.max_iter = max_iter  # maximum iterations/epochs per configuration
        self.eta = eta # defines downsampling rate (default=3)
        
        self.s_max = floor(log(max_iter, eta)) + 1  # number of unique executions of Successive Halving (minus one)
        self.B = (self.s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    def tune(self):
        best_hyperparameters = {}
        best_loss = inf
        
        for s in tqdm(reversed(range(self.s_max+1)), total=self.s_max+1):            
            n = int(ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s)) # initial number of configurations
            r = self.max_iter * self.eta**(-s) # initial number of iterations to run configurations for
            
            T = [self.model.sample_hyperparameters() for i in range(n)]
            
            for i in range(s+1):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta**(i)
                
                val_losses = [self.model.run(num_iters=r_i,hyperparameters=t) for t in T]
                best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]
                
                if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                    best_loss = val_losses[best_ix[0]]
                    best_hyperparameters = T[best_ix[0]]
                
                T = [T[i] for i in best_ix]
            
        return (best_loss, best_hyperparameters)