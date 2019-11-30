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

from . import Model, efficientnet

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

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    def sample_hyperparameters(self):
        return { 'learning-rate': round(random.uniform(0, 1), 2) }

    def run(self, num_iters, hyperparameters):
        net = efficientnet.EfficientNetB0()
        net = net.to(self.device)

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=hyperparameters['learning-rate'], momentum=0.9, weight_decay=5e-4)

        best_acc = -math.inf

        for epoch in range(math.floor(num_iters)):
            print('\nEpoch: %d' % epoch)

            # Train
            net.train()
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                pred = net(inputs)
                loss = ce_loss(pred, targets)
                loss.backward()

                optimizer.step()
            
            # Validate
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    pred = net(inputs)
                    loss = ce_loss(pred, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * (correct / total)
            if acc > best_acc:
                best_acc = acc

        return best_acc