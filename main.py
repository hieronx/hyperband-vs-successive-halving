from methods.hyperband import Hyperband
from methods.successive_halving import Successive_halving
from wrapper.benchmark import Benchmark
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hyperband", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hyperband', dest='hyperband', action='store_true')
    parser.add_argument('--halving', dest='hyperband',
                        action='store_false')
    parser.set_defaults(hyperband=True)

    parser = parser.parse_args()

    # hyperparameters space
    # watch out for loss nan's that means gradient is vanishing or exploding!
    params = [['lr', 0.001, 0.05, True], ['momentum', 0.89, 0.9, False]]

    # set up benchmark environment for testing configs, model, dataset, batchsize
    # color defines if input of cnn has 1 or 3 channels
    # args** model, datasetname, batchsize, mini-iterations
    # MNIST, FashionMNIST, CIFAR10
    batchsize = 100
    mini_iterations = 100
    benchmark = Benchmark('SimpleCNN',
                          'MNIST', batchsize, mini_iterations)

    # 1 iteration has (batchsize * mini-iteration) examples
    #eta is size
    iterations = 81
    eta = 3

    if parser.hyperband:
        hb = Hyperband(benchmark, params, iterations, eta)
    else:
        hb = Successive_halving(benchmark, params, iterations, eta)

    hb.tune()
