from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import random

from . import Model

class RidgeRegression(Model):

    def __init__(self):
        diabetes = datasets.load_diabetes()
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(diabetes.data, diabetes.target, test_size=0.20, random_state=42)

    def sample_hyperparameters(self):
        return { 'alpha': round(random.uniform(0, 1), 3) }

    def run(self, num_iters, hyperparameters):
        model = Ridge(max_iter=num_iters, alpha=hyperparameters['alpha']).fit(self.train_X, self.train_y)
        pred_y = model.predict(self.test_X)
        loss = mean_squared_error(self.test_y, pred_y)
        
        return loss