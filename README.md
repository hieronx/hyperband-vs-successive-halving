# Hyperband vs Successive Halving

Project work codebase for comparing Hyperband and Successive Halving as hyperparameter selection methods for deep neural networks.

## Setup

Make sure [Conda](https://www.anaconda.com/) is installed, and then run:

```
conda env create -f environment.yml || conda env update -f environment.yml || exit
conda activate hb-vs-sh
```

## Usage

To run the baseline model use the command:

```
python main.py baseline
```

To train other models, use this to print all command-line options:

```
python main.py -h
```

### Modifying hyperparameters

In your main.py file, please define:
```
params = [['lr', 0.001, 0.3, True]]
```

where params is a list of hyperparameters following the structure `hyperparameter name, lower bound, upper bound, log-random sampling`.

### Benchmark

Benchmark creates an class that will run each individual config when called, where
- The first arg is the model you want to use, color indicates the input channel of the model, True for color and false for gray;
- The second arg is the datasetname in torchvision;
- Third is batchsize, fourth is mini iterations;

```
benchmark = Benchmark(SimpleCNN, 'FashionMNIST', 128, 100)
```

One iteration has `batchsize * mini-iteration` examples.

### Hyperband or Successive Halving example


```
iterations = 81
eta = 3

hb = Hyperband(benchmark, params, iterations, eta)
hb = SuccessiveHalving(benchmark, params, iterations, eta)

hb.tune()
```