Project work codebase for comparing Hyperband and Successive Halving as hyperparameter selection methods for deep neural networks.

## == running the program ==

To run the benchmark use the command:
python3 main.py

## == hyperparameters ==

In your main.py file, please define:
params = [['lr', 0.001, 0.3, True]]

where params is a list of hyperparameters following the structure
hyperparametername, lowerbound, upperbound, log-random sampling
lr, 0.001, 03, True

## == benmark ==

Benchmark creates an class that will run eahc individual config when called
-The first arg is the model you want to use, color indicates the input channel
of the model, True for color and false for gray.
-The second arg is the datasetname in torchvision
Third is batchsize, fourth is mini iterations

benchmark = Benchmark(SimpleCNN, 'FashionMNIST', 128, 100)

1 iteration has (batchsize \* mini-iteration) examples eta is size

## == hyperband or successive halving ==

iterations = 81
eta = 3

hb = Hyperband(benchmark, params, iterations, eta)
or
hb = SuccessiveHalving(benchmark, params, iterations, eta)

## to run hyperband or successive halving

hb.tune()
