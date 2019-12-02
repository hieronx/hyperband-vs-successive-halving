'''
Source: https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import math
import random
from tqdm import tqdm

from . import Model, resnet

class CifarCnn(Model):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.bs = 32

        self.train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=self.bs, shuffle=True, num_workers=0)

        self.test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(self.test, batch_size=self.bs, shuffle=False, num_workers=0)

    def sample_hyperparameters(self):
        return { 'learning-rate': round(random.uniform(0, 1), 2) }

    def run(self, num_iters, hyperparameters):
        net = resnet.ResNet18()
        net = net.to(self.device)

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        optimizer = optim.SGD(net.parameters(), lr=hyperparameters['learning-rate'], momentum=0.9, weight_decay=5e-4)

        best_acc = -math.inf
        num_train_batches, num_test_batches = math.ceil(len(self.train) / self.bs), math.ceil(len(self.test) / self.bs)

        for epoch in range(math.floor(num_iters)):
            print('\nEpoch %d of %d' % ((epoch+1), (num_iters)))

            # Train
            net.train()
            for batch_idx, (inputs, targets) in tqdm(enumerate(self.train_loader), total=num_train_batches):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                pred = net(inputs)
                loss = F.nll_loss(pred, targets)
                loss.backward()

                optimizer.step()
            
            # Validate
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in tqdm(enumerate(self.test_loader), total=num_test_batches):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    pred = net(inputs)
                    loss = F.nll_loss(pred, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * (correct / total)
            if acc > best_acc:
                best_acc = acc

        return best_acc