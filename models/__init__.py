from abc import ABC, abstractmethod
 
class Model(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample_hyperparameters(self):
        pass

    @abstractmethod
    def run(self, num_iters, hyperparameters):
        pass
