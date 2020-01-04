# ultilities
import argparse
import sys


def parse(args):
    parser = argparse.ArgumentParser(
        description="Hyperband vs Successive Halving", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('script', nargs='?', default='')

    parser.add_argument('--hyperband', action='store_true',
                        help='If added, runs Hyperband')
    parser.add_argument('--successive_halving', action='store_true',
                        help='If added, runs Successive Halving')

    parser.add_argument('--iterations', type=int, default=10,
                        help='Set number of Resource, R, one unit of R == (batchsize x mini-iterations)')
    parser.add_argument('--eta', type=int, default=3, help='Set the eta')

    parser.add_argument('--model_name', default='SmallCNN',
                        help='Set the modelname (default=SmallCNN)')
    parser.add_argument('--dataset', default='FashionMNIST',
                        help='Set the dataset (default=FashionMNIST)')

    parser.add_argument('--batch_size', type=int,
                        default=64, help='Set the batch size (default=64)')
    parser.add_argument('--mini_iterations', type=int,
                        default=100, help='Set the number of mini-iterations (default=100)')

    parser.add_argument('--lr_schedule', type=str, choices=['Linear', 'LambdaLR', 'StepLR', 'ExponentialLR', 'CyclicLR'], default='Linear',
                        help='Set the learning rate schedule')
    parser.add_argument('--visualize_lr_schedule', action='store_true',
                        help='If added, plots the learning rate schedule to an image file')

    parser.add_argument('--mult_r', type=int, default=1,
                        help='Set the maximum budget, running the experiments every step_r from 1 till mult_r, with each time (iterations x step_r) more resources (default=1)')
    parser.add_argument('--step_r', type=int, default=1,
                        help='Set the coarsity the resourses, runs an experiment every step_r from 1 till mult_r (default=1)')

    parser.add_argument('--seed', type=int, default=2020,
                        help='Set the seed for experimental consistency (default=2020)')

    parser.add_argument('--dry_run', action='store_true',
                        help='Runs the whole experiment without training the models')

    parser.add_argument('--save', action='store_true',
                        help='If added stores the train, val, test, lr, brackets etc.. results of all models')

    arg = parser.parse_args(args)

    if (arg.mult_r % arg.step_r) is not 0:
        print('Ensure that mult_r MOD step_r == 0')
        exit(1)

    if not arg.hyperband and not arg.successive_halving and not arg.script:
        print('Ensure that you run at least Hyperband or Successive Halving or both')
        exit(1)

    return arg
