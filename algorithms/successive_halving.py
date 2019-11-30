from tqdm import tqdm
import math
import numpy as np

class SuccessiveHalving:

    def __init__(self, model, max_iter):
        self.model = model
        self.max_iter = max_iter

    def tune(self):
        best_hyperparameters = {}
        best_loss = math.inf
    
        n = int(math.ceil(int(self.max_iter) * 2)) # initial number of configurations
        r = self.max_iter # initial number of iterations to run configurations for
        
        T = [self.model.sample_hyperparameters() for i in range(n)]
        
        for i in range(4):
            n_i = n * 2**(-i)
            r_i = r * 2**(i)
            
            val_losses = [self.model.run(num_iters=r_i,hyperparameters=t) for t in T]
            best_ix = np.argsort(val_losses)[0:int(n_i / 2)]
            
            if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                best_loss = val_losses[best_ix[0]]
                best_hyperparameters = T[best_ix[0]]
            
            T = [T[i] for i in best_ix]
            
        return (best_loss, best_hyperparameters)