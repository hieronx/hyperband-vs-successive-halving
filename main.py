import argparse
import time

from methods.hyperband import Hyperband
from methods.successive_halving import Successive_halving
from wrapper.benchmark import Benchmark

def run(args):
    # hyperparameters space
    # watch out for loss nan's that means gradient is vanishing or exploding!
    params = [['lr', 0.001, 0.05, True], ['momentum', 0.89, 0.9, False]]

    # set up benchmark environment for testing configs, model, dataset, batchsize
    # color defines if input of cnn has 1 or 3 channels
    # args** model, datasetname, batchsize, mini-iterations
    # MNIST, FashionMNIST, CIFAR10
    benchmark = Benchmark(args.model_name, args.dataset, args.batch_size, args.mini_iterations)

    run_datetime = time.strftime("%Y%m%d-%H%M%S")

    if args.hyperband:
        hb = Hyperband(benchmark, params, args.iterations, args.eta, run_datetime)
    else:
        hb = Successive_halving(benchmark, params, args.iterations, args.eta, run_datetime)

    hb.tune()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperband vs Successive Halving", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--hyperband', dest='hyperband', action='store_true')
    parser.add_argument('--successive_halving', dest='hyperband', action='store_false')
    parser.set_defaults(hyperband=True)

    parser.add_argument('--iterations', type=int, default=81)
    parser.add_argument('--eta', type=int, default=3)

    parser.add_argument('--model_name', default='SimpleCNN')
    parser.add_argument('--dataset', default='MNIST')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--mini_iterations', type=int, default=100)

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    run(args)