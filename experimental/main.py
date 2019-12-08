import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hyperband import Hyperband
from efficientnet import *
from simple_cnn import SimpleCNN

if __name__ == '__main__':
    params = [['momentum', 0.5, 1.5, False], [
        'weight_decay', 0.000005, 0.0005, False]]
    #hb = Hyperband(model=EfficientNetB0(), param=params, ds_name='CIFAR10', max_iter = 81, eta = 3)
    # hb.tune()

    #hb = Hyperband(model=EfficientNetB0(num_classes=10, color=False), param=params, ds_name='FashionMNIST', max_iter = 81, eta = 3)
    hb = Hyperband(model=SimpleCNN(num_classes=10, color=True),
                   hyperband=True, param=params, ds_name='CIFAR10', max_iter=81, eta=3)
    hb.tune()
