import argparse
import time
import torch
import numpy as np
import sys

from methods.hyperband import Hyperband
from methods.successive_halving import Successive_halving
from methods.utils import parse

from wrapper.benchmark import Benchmark


def run(args):
    # params , name; lowerbound; upperbound; logsampling, if lower==upper then it is a static choice
    params = [['lr', 0.001, 0.25, False], ['momentum', 0.9, 0.9, False]]

    benchmark = Benchmark(args.model_name, args.dataset,
                          args.batch_size, args.mini_iterations, args.lr_schedule, args.seed, args.dry_run)

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
                           seed=args.seed, filename=hb_filename, save=args.save, visualize_lr_schedule=args.visualize_lr_schedule)
            hb.tune()

        if args.successive_halving:
            print('\n\n-------------------------------------------------------')
            print('Running Successive Halving with R = %d and η = %d\n' %
                  (R, args.eta))
            sh = Successive_halving(
                benchmark, params, max_iter=R, eta=args.eta, seed=args.seed, filename=sh_filename, save=args.save, visualize_lr_schedule=args.visualize_lr_schedule)
            sh.tune()


def start_script(args):
    if args.script == 'baseline':
        seeds = [2020, 4040, 6060]

        for seed in seeds:
            print('Using seed %d' % seed)
            args.seed = seed

            # Set the seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            schedules = ['Linear', 'LambdaLR',
                         'StepLR', 'ExponentialLR', 'CyclicLR']

            for schedule in schedules:
                print('Using schedule %s' % schedule)
                args.lr_schedule = schedule

                # Call run for R = 20,40,60,80,100 for both Hyperband and Successive Halving
                args.step_r = 2
                args.mult_r = 10

                args.hyperband = True
                run(args)

                args.successive_halving = True
                args.hyperband = False
                run(args)


if __name__ == '__main__':
    args = parse(sys.argv[1:])

    # Set the seed for an individual run
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.script:
        start_script(args)
    else:
        run(args)
