from tqdm import tqdm
import math
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from utils import progress_bar
# Based on pseudocode from https://homes.cs.washington.edu/~jamieson/hyperband.html

class Hyperband:

    def __init__(self, model, param, ds_name, max_iter = 81, eta = 3, seed = 1234):
        self.model = model
        self.param = param
        self.ds_name = ds_name

        self.max_iter = max_iter  # maximum iterations/epochs per configuration
        self.eta = eta # defines downsampling rate (default=3)
        self.seed = seed
        
        logeta = lambda x: math.log(x)/math.log(eta)
        self.s_max = int(logeta(max_iter))

        self.B = (self.s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

        self.bs = 128
    
    def create_hyperparameterspace(self):
        cs = CS.ConfigurationSpace(seed=self.seed)

        for configs in self.param:
            hp = CSH.UniformFloatHyperparameter(name=configs[0], lower=configs[1], upper=configs[2], log=configs[3])
            cs.add_hyperparameter(hp)
        return cs
        
    def tune(self):
        best_hyperparameters = {}
        best_loss = math.inf
        cs = self.create_hyperparameterspace()

        for s in tqdm(reversed(range(self.s_max+1)), total=self.s_max+1):            
            n = int(math.ceil(int(self.B / self.max_iter / (s+1)) * self.eta**s)) # initial number of configurations
            r = self.max_iter * self.eta**(-s) # initial number of iterations to run configurations for

            T = [cs.sample_configuration() for i in range(n)]

            for i in range(s+1):
                n_i = n * self.eta**(-i)
                r_i = r * self.eta**(i)
                
                val_losses = [self.train(num_iters=r_i, hyperparameters=t, conf='b_'+str(s)+'_ni_'+str(i)) for t in T]
                best_ix = np.argsort(val_losses)[0:int(n_i / self.eta)]
                
                if len(best_ix) > 0 and val_losses[best_ix[0]] < best_loss:
                    best_loss = val_losses[best_ix[0]]
                    best_hyperparameters = T[best_ix[0]]
                
                T = [T[i] for i in best_ix]
            
        return (best_loss, best_hyperparameters)

    def prepare_data(self):

        if self.ds_name == 'CIFAR10':
            transform_train = transforms.Compose([
            #    transforms.RandomCrop(32, padding=4),
            #    transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

        elif self.ds_name == 'FashionMNIST':
            transform_train = transforms.Compose([
            #    transforms.RandomCrop(32, padding=4),
            #    transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914,), (0.2023,)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, ), (0.2023,)),
            ])
            trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.bs, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.bs, shuffle=False, num_workers=2)

        return trainloader, testloader

    def train(self, num_iters, hyperparameters, conf):

        # Data
        print('\n==> Preparing data..')
        trainloader, testloader = self.prepare_data()
        print('==> Done preparing data..')
        # Model
        print('===> Building model..')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net = self.model
        net = net.to(device)

        if device == 'cuda:0':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        momentum = hyperparameters.get('momentum')
        weight_decay  = hyperparameters.get('weight_decay')
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=momentum , weight_decay=weight_decay)

        print('====> Done building model..')
        start_epoch = 0
        loss = math.inf

        for epoch in range(start_epoch, start_epoch+int(num_iters)):
            print('\nEpoch: %d of %d' % (epoch, int(num_iters)))
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            endloss = 0

            #progress_bar(0, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #        % (0, 0, 0, 0))
            
            t = tqdm(enumerate(trainloader),total=len(trainloader), ncols=130, position=0)
            for batch_idx, (inputs, targets) in t:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                t.set_description('| loss=%0.3f | acc=%0.3f%% | %g/%g |' % (train_loss/(batch_idx+1), (100.*correct/total), correct, total))

                endloss = train_loss/(batch_idx+1)
                #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            #acc = 100.*correct/total
            #if acc > best_acc:
            print('Saving..')
            state = {
            'net': net.state_dict(),
            'acc': 100.*correct/total,
            'loss': endloss,
            'epoch': epoch,
            'num_iters': num_iters,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'conf': conf
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            
            torch.save(state, './checkpoint/'+conf+'_'+ str(momentum)+'_'+str(weight_decay)+'.pth')
            
        #self.test(self, testloader=testloader, checkpoint='./checkpoint/'+conf+'_'+ str(momentum)+'_'+str(weight_decay)+'.pth')
        self.test(testloader, './checkpoint/'+conf+'_'+ str(momentum)+'_'+str(weight_decay)+'.pth')

        return loss
     
    def test(self, testloader, checkpoint):
        # Load checkpoint.
        print('=====> Testing from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net = self.model
        net = net.to(device)

        net.load_state_dict(checkpoint['net'])
        print('accuracy %f%%, loss %f, momentum %f, weight_decay %f' % (checkpoint['acc'], checkpoint['loss'],checkpoint['momentum'], checkpoint['weight_decay']))

        if device == 'cuda:0':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        momentum = checkpoint['momentum']
        weight_decay  = checkpoint['weight_decay']
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=momentum , weight_decay=weight_decay)

        #net.eval()
        test_loss = 0
        correct = 0
        total = 0
        t = tqdm(enumerate(testloader),total=len(testloader), ncols=130, position=0)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in t:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_description('| loss=%0.3f | acc=%0.3f%% | %g/%g |' % (test_loss/(batch_idx+1), (100.*correct/total), correct, total))

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
