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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # params , name; lowerbound; upperbound; logsampling, if lower==upper then it is a static choice
    params = [['lr', 0.003, 0.3, False], ['momentum', 0.9, 0.9, False]]

    benchmark = Benchmark(args.model_name, args.dataset,
                          args.batch_size, args.mini_iterations, args.seed, args.dry_run)

    run_datetime = time.strftime("%Y%m%d-%H%M%S")
    postfix = 'seed' + str(args.seed) + '-' + str(run_datetime)

    hb_filename = 'hb-R' + str(args.iterations) + '-to-R' + \
        str(args.mult_r*args.iterations) + '-' + postfix
    sh_filename = 'sh-R' + str(args.iterations) + '-to-R' + \
        str(args.mult_r*args.iterations) + '-' + postfix

    for R_multiple in range(1, args.mult_r+1, args.step_r):
        R = args.iterations * R_multiple

        if args.hyperband:
            print('Running Hyperband with R = %d and η = %d' % (R, args.eta))

            hb = Hyperband(benchmark, params, max_iter=R, eta=args.eta,
                           seed=args.seed, filename=hb_filename)
            hb.tune()

        if args.successive_halving:
            print('Running Successive Halving with R = %d and η = %d' %
                  (R, args.eta))
            sh = Successive_halving(
                benchmark, params, max_iter=R, eta=args.eta, seed=args.seed, filename=sh_filename)
            sh.tune()
