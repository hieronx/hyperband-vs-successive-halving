import argparse
import time
import torch
import numpy as np
import sys

from methods.hyperband import Hyperband
from methods.successive_halving import Successive_halving
from methods.ultils import parse

from wrapper.benchmark import Benchmark

if __name__ == '__main__':

    args = parse(sys.argv[1:])

    # Set the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # params , name; lowerbound; upperbound; logsampling, if lower==upper then it is a static choice
    params = [['lr', 0.001, 0.25, False], ['momentum', 0.9, 0.9, False]]

    benchmark = Benchmark(args.model_name, args.dataset,
                          args.batch_size, args.mini_iterations, args.seed, args.dry_run)

    postfix = str(time.strftime("%Y%m%d-%H%M%S"))

    hb_filename = 'seed' + str(args.seed) + '-hb-R' + str(args.iterations) + '-to-R' + \
        str(args.mult_r*args.iterations) + '-' + postfix
    sh_filename = 'seed' + str(args.seed) + '-sh-R' + str(args.iterations) + '-to-R' + \
        str(args.mult_r*args.iterations) + '-' + postfix

    for R_multiple in range(args.step_r, args.mult_r+1, args.step_r):
        R = args.iterations * R_multiple

        if args.hyperband:
            print('\n\n-------------------------------------------------------')
            print('Running Hyperband with R = %d and η = %d\n' %
                  (R, args.eta))

            hb = Hyperband(benchmark, params, max_iter=R, eta=args.eta,
                           seed=args.seed, filename=hb_filename, save=args.save)
            hb.tune()

        if args.successive_halving:
            print('\n\n-------------------------------------------------------')
            print('Running Successive Halving with R = %d and η = %d\n' %
                  (R, args.eta))
            sh = Successive_halving(
                benchmark, params, max_iter=R, eta=args.eta, seed=args.seed, filename=sh_filename, save=args.save)
            sh.tune()
