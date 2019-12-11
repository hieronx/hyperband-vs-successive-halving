from methods.hyperband import Hyperband
from methods.successive_halving import SuccessiveHalving
from wrapper.benchmark import Benchmark

if __name__ == '__main__':

    # hyperparameters space
    params = [['lr', 0.001, 0.3, True]]

    # set up benchmark environment for testing configs, model, dataset, batchsize
    # color defines if input of cnn has 1 or 3 channels
    # args** model, datasetname, batchsize, mini-iterations
    # MNIST, FashionMNIST, CIFAR10
    batchsize = 128
    mini_iterations = 100
    benchmark = Benchmark('SimpleCNN',
                          'FashionMNIST', batchsize, mini_iterations)

    # 1 iteration has (batchsize * mini-iteration) examples
    #eta is size
    iterations = 81
    eta = 3

    hb = Hyperband(benchmark, params, iterations, eta)
    #hb = SuccessiveHalving(benchmark, params, iterations, eta)
    hb.tune()
