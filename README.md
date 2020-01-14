# Hyperband vs Successive Halving

Project work codebase for comparing Hyperband and Successive Halving as hyperparameter selection methods for deep neural networks given varying learning rate schedules.

## Setup

Make sure [Conda](https://www.anaconda.com/) is installed, and then run:

```
conda env create -f environment.yml || conda env update -f environment.yml || exit
conda activate hb-vs-sh
```

## Usage

To run the baseline model benchmark, use the command:

```
python main.py baseline
```

Add the save command to save the results

```
python main.py baseline --save
```

Use this to print all command-line options:

```
python main.py -h
```

### Modifying hyperparameters

In your main.py file, please define:

```
params = [['lr', 0.001, 0.25, False], ['momentum', 0.9, 0.9, False]]
```

Where params is a list of hyperparameters following the structure `hyperparameter name, lower bound, upper bound, log-random sampling`.
If the upper- and lower bound is the same, then it will use the value of the lower bound.

### Benchmark

Benchmark creates an class that will run each individual config when called, where

- The first arg is the model you want to use
- The second arg is the datasetname in torchvision
- Third is batchsize
- fourth is mini-terations
- Fifth is the learning schedule
- Sixth is the seed, all functions except for the weight initalization of the models uses this seed
- Seventh is the dry_run option, if in the command --dry_run is added, it will then run the benchmark without training the models

```
Benchmark(args.model_name, args.dataset, args.batch_size, args.mini_iterations, args.lr_schedule, args.seed, args.dry_run)
```

One iteration has `batchsize * mini-iteration` examples.

### Hyperband or Successive Halving example

```
hb = Hyperband(benchmark, params, max_iter=R, eta=args.eta,
                           seed=args.seed, filename=hb_filename, save=args.save, visualize_lr_schedule=args.visualize_lr_schedule)
hb.tune()

sh = Successive_halving(
                benchmark, params, max_iter=R, eta=args.eta, seed=args.seed, filename=sh_filename, save=args.save, visualize_lr_schedule=args.visualize_lr_schedule)
sh.tune()
```

## Sources

- Hyperband and Successive Halving implementations are based on https://homes.cs.washington.edu/~jamieson/hyperband.html.
- The SmallCNN model is based on code from https://nextjournal.com/gkoehler/pytorch-mnist.
