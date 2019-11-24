import argparse
import multiprocessing
from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool as Pool

from scripts import configs_cv as C
from scripts import ranges_cv as R
from scripts.rl_exp_cv_opt import main as main_func
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


def main(env, configs_name, range_names, base_algorithms, n_processes, args=None):
    # Set to the number of workers you want (it defaults to the cpu count of your machine)
    if n_processes == -1:
        n_processes = None
    print('# of CPU (threads): {}'.format(multiprocessing.cpu_count()))
    configs = getattr(C, 'configs_' + configs_name)
    tps = []
    assert len(range_names) == 1
    for range_name in range_names:
        ranges = R.get_ranges(env, range_name, base_algorithms)
        combs, keys = get_valcombs_and_keys(ranges)
        print('Total number of combinations: {}'.format(len(combs)))
        for i, comb in enumerate(combs):
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
            tp['experimenter']['data_gen_type'] = args.self
            tps.append(tp)
    '''
    assert len(tps) == 1
    tps[0]['general']['seed'] = args.seed
    main_func(tps[0])
    '''
    with Pool(processes=n_processes, maxtasksperchild=1) as p:  # None for using all the cpus available
        p.map(main_func, tps, chunksize=1)
        p.map(func, tps, chunksize=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Change this to 'cp', 'hopper', snake', 'walker3d', or 'default', to use the stepsize setting for your env.
    parser.add_argument('env')
    parser.add_argument('configs_name')
    parser.add_argument('-r', '--range_names', nargs='+')
    parser.add_argument('-a', '--base_algorithms', nargs='+', default=['rnatgrad'])
    parser.add_argument('--n_processes', type=int, default=-1)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--self', type=int, default=0)
    args = parser.parse_args()

    main(args.env, args.configs_name, 
            args.range_names, args.base_algorithms, 
            args.n_processes, args)
