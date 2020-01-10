from wrapper.benchmark import Benchmark
from matplotlib import pyplot as plt


if __name__ == '__main__':
    d = ['Linear', 'LambdaLR', 'StepLR', 'ExponentialLR', 'CyclicLR']
    benchmark = Benchmark('SmallCNN', 'FashionMNIST', 64,
                          100, 'LambdaLR', 2020, True)
    s = benchmark.test_schedule(100, 0.25)
    plt.plot(s)
    plt.savefig('./lr-schedule.png')
