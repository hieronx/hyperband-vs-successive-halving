# ultilities
import argparse
import sys


def parse(args):
    parser = argparse.ArgumentParser(
        description="Hyperband vs Successive Halving", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--hyperband', action='store_true',
                        help='If added, runs Hyperband')
    parser.add_argument('--successive_halving', action='store_true',
                        help='If added, runs Successive Halving')

    parser.add_argument('--iterations', type=int, default=10,
                        help='Set number of Resource, R, one unit of R == (batchsize x mini-iterations)')
    parser.add_argument('--eta', type=int, default=3, help='Set the eta')

    parser.add_argument('--model_name', default='SmallCNN',
                        help='Set the modelname')
    parser.add_argument('--dataset', default='FashionMNIST',
                        help='Set the dataset')

    parser.add_argument('--batch_size', type=int,
                        default=64, help='Set the batch size')
    parser.add_argument('--mini_iterations', type=int,
                        default=100, help='Set the number of mini-iterations')

    parser.add_argument('--mult_r', type=int, default=1,
                        help='Set the maximum budget, running the experiments every step_r from 1 till mult_r, with each time (iterations x step_r) more resources')
    parser.add_argument('--step_r', type=int, default=1,
                        help='Set the coarsity the resourses, runs an experiment every step_r from 1 till mult_r')

    parser.add_argument('--seed', type=int, default=2020,
                        help='Set the seed for experimental consistency')

    parser.add_argument('--dry_run', action='store_true',
                        help='Runs the whole experiment without training the models')

    parser.add_argument('--save', action='store_true',
                        help='If added stores the train, val, test, lr, brackets etc.. results of all models')

    arg = parser.parse_args(args)

    if (arg.mult_r % arg.step_r) is not 0:
        print('Ensure that mult_r MOD step_r == 0')
        exit(1)

    if not arg.hyperband and not arg.successive_halving:
        print('Ensure that you run at least Hyperband or Successive Halving or both')
        exit(1)

    return arg
