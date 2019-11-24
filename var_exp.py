import argparse
import multiprocessing
from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import os

from scripts import configs_cv as C
from scripts import ranges_cv as R
from scripts.rl_exp_cv_var import main as main_func
import itertools
import copy
from rl.tools.utils.misc_utils import zipsame


def func(tp):
    print(tp['general']['exp_name'], tp['general']['seed'])


def get_valcombs_and_keys(ranges):
    keys = []
    values = []
    for r in ranges:
        keys += r[::2]
    values = [list(zipsame(*r[1::2])) for r in ranges]
    cs = itertools.product(*values)
    combs = []
    for c in cs:
        comb = []
        for x in c:
            comb += x
        print(comb)
        combs.append(comb)
    return combs, keys


def main(env, configs_name, range_name, base_algorithms, args=None):
    # Set to the number of workers you want (it defaults to the cpu count of your machine)
    print('# of CPU (threads): {}'.format(multiprocessing.cpu_count()))
    configs = getattr(C, 'configs_' + configs_name)
    
    ranges = R.get_ranges(env, range_name, [base_algorithms])
    combs, keys = get_valcombs_and_keys(ranges)
    print('Total number of combinations: {}'.format(len(combs)))
    
    comb = combs[0]
    
    tp = copy.deepcopy(configs)
    value_strs = [tp['general']['exp_name']]  # the description string start from the the exp name
    for (value, key) in zip(comb, keys):
        entry = tp
        for k in key[:-1]:  # walk down the configs tree
            entry = entry[k]
        # Make sure the key is indeed included in the configs, so that we set the desired flag.
        assert key[-1] in entry, 'newly added key: {}'.format(key[-1])
        entry[key[-1]] = value
        # We do not include seed number.
        if len(key) == 2 and key[0] == 'general' and key[1] == 'seed':
            continue
        else:
            if value is True:
                value = 'T'
            if value is False:
                value = 'F'
            value_strs.append(str(value).split('/')[0])  # in case of experts/cartpole/final.ckpt....
    tp['general']['exp_name'] = '-'.join(value_strs)
    tp['experimenter']['exp_type'] = args.type
    tp['experimenter']['ro'] = args.ro

    if not os.path.exists('./Model'):
        os.mkdir('./Model')
    files = os.listdir('./Model')
    func = filter(lambda x: "sim" in x, files)
    
    prefixes = [f.split('sim')[0] for f in func]
    prefixes = list(set(prefixes))

    all_pairs = []
    for prefix in prefixes:
        addon = int(prefix.split('_')[1]) 
        
        if addon >= args.end or addon < args.start:
            continue

        if addon % args.show_freq != 0:
            continue

        all_pairs.append((addon, prefix))

    all_pairs.sort()
    tp['all_pairs'] = all_pairs
    tp['general']['seed'] = args.seed
    main_func(tp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Change this to 'cp', 'hopper', snake', 'walker3d', or 'default', to use the stepsize setting for your env.
    parser.add_argument('env')
    parser.add_argument('configs_name')
    parser.add_argument('-r', '--range_name', default='st')
    parser.add_argument('-a', '--base_algorithms', default='rnatgrad')
    parser.add_argument('--type', type=str, default='train', choices=['train', 'gen-ro', 'est-mean', 'cal-var'])
    parser.add_argument('--seed', type=int, default=1000, help='the random seed for generating')
    parser.add_argument('--ro', type=int, default=500, help='the number of rollouts to generate')
    
    parser.add_argument('--start', type=int, default=0, help='the start index of the models to compare')
    parser.add_argument('--end', type=int, default=25, help='the end index of the models to compare')

    parser.add_argument('--show-freq', type=int, default=5, help='frequency to show the results')
    args = parser.parse_args()

    assert args.ro % 5 == 0

    if args.seed < 0:
        args.seed = np.random.randint(10000, 90000)
    
    main(args.env, args.configs_name, 
            args.range_name, args.base_algorithms,
            args)
