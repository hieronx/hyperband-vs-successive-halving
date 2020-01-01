# Benchmark runs a configuration given a model, iterations, dataset and hyperparameters

import copy
from tqdm import tqdm
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import os
import sys

from .models import *


class Benchmark:

    def __init__(self, model, dataset, batchsize, mini_iterations, seed, dry_run, val_size=0.1):
        self.model = model
        self.dataset = dataset
        self.bs = batchsize
        self.mini_iterations = mini_iterations
        self.seed = seed
        self.dry_run = dry_run
        self.val_size = val_size

        self.tensor_shape = None
        # number of examples in train, val and test set
        self.size_train = 0
        self.size_val = 0
        self.size_test = 0

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.trainloader, self.valloader, self.testloader = self.prepare_data()

    # save meta-info as dict
    def get_meta(self):
        meta = {'model': self.model, 'dataset': self.dataset, 'bs': self.bs, 'mini_iterations': self.mini_iterations,
                'size_train': self.size_train, 'size_val': self.size_val, 'size_test': self.size_test, 'tensor_shape': self.tensor_shape}
        return meta

    # this gets the lr from the optimizer, for when we wants to use lr decay e.g lr schedules
    # as indicated in the paper for diverse convergence rates
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # does transforms on data to be used with model
    def transform_data(self):
        # normalize to mean depending on dataset
        if self.dataset == 'CIFAR10':
            norm = transforms.Normalize(
                (0.49139968, 0.48215827,
                 0.44653124), (0.24703233, 0.24348505, 0.26158768))
        elif self.dataset == 'CIFAR100':
            norm = transforms.Normalize(
                (0.50707519, 0.48654887, 0.44091785), (0.26733428, 0.25643846, 0.27615049))
        elif self.dataset == 'FashionMNIST' or self.dataset == 'MNIST':
            norm = transforms.Normalize((0.1307,), (0.3081,))
        else:
            print('normalisation of data has not been implemented, pls do that')
            exit(1)

        # demeaning
        transform = transforms.Compose([
            # transforms.CenterCrop((28, 28)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm])

        return transform

    # get the data and split it into train, val and test sets
    def prepare_data(self):
        print('===> Preparing data')
        transform = self.transform_data()

        train_valset = getattr(torchvision.datasets, self.dataset)(
            root='./data', train=True, download=True, transform=transform)

        testset = getattr(torchvision.datasets, self.dataset)(
            root='./data', train=False, download=True, transform=transform)

        # it assumes the data is all same size!
        # [channel, width, height]
        self.tensor_shape = list(train_valset[0][0].size())

        num_train_val = len(train_valset)
        indices = list(range(num_train_val))

        # stratify to ensure balanced classes in train and val
        a, b = train_test_split(
            indices, stratify=train_valset.targets, test_size=self.val_size, random_state=self.seed)

        trainset = torch.utils.data.Subset(train_valset, a)
        valset = torch.utils.data.Subset(train_valset, b)

        self.size_train = len(trainset)
        self.size_val = len(valset)
        self.size_test = len(testset)

        # trainloader uses an iterative sampler to sample the data sequentially to get to the
        # number of iterations needed
        trainloader = torch.utils.data.DataLoader(
            trainset, sampler=torch.utils.data.sampler.SequentialSampler(trainset), shuffle=False, num_workers=1, batch_size=self.bs)
        valloader = torch.utils.data.DataLoader(
            valset, num_workers=1, batch_size=self.bs)
        testloader = torch.utils.data.DataLoader(
            testset, num_workers=1, batch_size=self.bs)

        print('===> Done preparing data')
        return trainloader, valloader, testloader

    # validate runs the val and test data and returns the loss and acc
    def validate(self, net, loader, test, hyperparameters):
        net = net.to(self.device)

        if self.device == 'cuda:0':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        loss = math.inf

        # start validating/testing mode
        net.eval()
        val_loss = 0
        correct = 0
        total = 0

        t = tqdm(enumerate(loader), total=len(loader), ncols=145,
                 position=1, bar_format="{desc:<55}{percentage:3.0f}%|{bar}{r_bar}", leave=False)

        if test:
            message = 'Testing... | lr=%0.04f | loss=%0.3f | acc=%0.3f%% | %g/%g |'
        else:
            message = 'Validate.. | lr=%0.04f | loss=%0.3f | acc=%0.3f%% | %g/%g |'

        with torch.no_grad():
            for batch_idx, (inputs, targets) in t:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_description(message %
                                  (hyperparameters.get('lr'), val_loss/(batch_idx+1), (100.*correct/total), correct, total))

        return (val_loss/(batch_idx+1)), (100.*correct/total)

    # train runs the train iterator and returns the running loss, acc and trained modelcd
    def train(self, trainloader, iterations, hyperparameters):
        #set seed again, so that each model will have the same init weights
        torch.manual_seed(self.seed)

        if not hasattr(sys.modules[__name__], self.model):
            print('====> Model doesn\'t exist!')
            exit(1)

        net = getattr(sys.modules[__name__], self.model)(
            tensor_shape=self.tensor_shape)

        net = net.to(self.device)

        if self.device == 'cuda:0':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(net.parameters(), **
                              hyperparameters.get_dictionary())
        # optimizer = optim.Adam(params=net.parameters(),lr = hyperparameters.get('lr'))
        #lr = self.get_lr(optimizer)

        loss = math.inf

        # start train mode
        net.train()
        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        total_resource = int(iterations * self.mini_iterations)

        t = tqdm(range(total_resource), total=total_resource, ncols=145,
                 position=1, bar_format="{desc:<55}{percentage:3.0f}%|{bar}{r_bar}", leave=False)

        dataloader_iterator = iter(trainloader)

        for i in t:
            # check if it reached the end of the data, in that case re-init the iterator from start
            try:
                inputs, targets = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(trainloader)
                inputs, targets = next(dataloader_iterator)

            inputs, targets = inputs.to(
                self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description('Training.. | lr=%0.4f | loss=%0.3f | acc=%0.3f%% | %g/%g |' %
                              (hyperparameters.get('lr'), train_loss/(i+1), (100.*correct/total), correct, total))

            running_loss = train_loss / (i + 1)

        return running_loss, (100.*correct/total), net

    # runs the training, validation and testing; returns val loss
    def run(self, iterations, hyperparameters):
        if self.dry_run:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            # train, validate and test
            train_loss, train_accuracy, net = self.train(
                self.trainloader, iterations, hyperparameters)

            val_loss, val_accuracy = self.validate(
                net, self.valloader, False, hyperparameters)

            test_loss, test_accuracy = self.validate(
                net, self.testloader, True, hyperparameters)

            return train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy
