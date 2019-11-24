import math
STEPSIZE = {
    'cp': {'etas': {'adam': 0.005, 'natgrad': 0.05, 'rnatgrad': 0.05, 'trpo': 0.002},
           'c': 0.01},
}

# Fixed.
ranges_common = [
    [['general', 'seed'], [x * 10000 + 5000 for x in range(10)]],
    [['experimenter', 'pretrain_kwargs', 'rollout_kwargs', 'min_n_samples'], [5000]],
    [['experimenter', 'pretrain_kwargs', 'pretrain'], [True]],
    [['experimenter', 'pretrain_kwargs', 'n_vf_updates'], [1]],
    [['experimenter', 'pretrain_kwargs', 'n_dyn_updates'], [1]],
    [['experimenter', 'pretrain_kwargs', 'update_pol_nor'], [True]],
    [['advantage_estimator', 'gamma'], [1.0]],
    [['advantage_estimator', 'v_target'], [1.0]],
    [['advantage_estimator', 'lambd'], [1.0]],
    [['algorithm', 'alg_kwargs', 'update_pol_nor'], [False]],
    [['oracle', 'dyn', 'fun_kwargs', 'max_n_samples'], [200000]],
    [['policy', 'nor_kwargs', 'clip_thre'], [None]],
]

# Can tune.
ranges_option = [
    [['experimenter', 'run_alg_kwargs', 'n_itrs'], [50]],  
    # [['general', 'max_episode_steps'], [1000, 2000, 4000, 8000, 16000]],
    [['general', 'max_episode_steps'], [1000]],
    [['advantage_estimator', 'delta'], [0.999]],
    [['advantage_estimator', 'data_aggregation'], [False]],
    [['advantage_estimator', 'max_n_rollouts'], [30]],
    [['advantage_estimator', 'vfn_params', 'fun_kwargs', 'n_epochs'], [500]],
    [['algorithm', 'rollout_kwargs', 'min_n_samples'], [5000]],
    [['algorithm', 'rollout_kwargs', 'max_n_rollouts'],[50]],
    [['oracle', 'or_kwargs', 'n_ac_samples'], [1000]],
    [['oracle', 'or_kwargs', 'switch_at_itr'], [None]],
    [['oracle', 'or_kwargs', 'cv_onestep_weighting'], [False]],
]

ranges_option_cv = ranges_option +[
    [['algorithm', 'env_type'], [0.1]],  # env for training vf    
    [['algorithm', 'train_vf_with_sim'], [True]],
    [['oracle', 'env_type'], ['dyn']],  # env for model, dyn
]

nros = 5


ranges_sigmas = ranges_option + [
    [['oracle', 'or_kwargs', 'cv_type'], ['state']],
    [['oracle', 'env_type'], ['true']],  # env for model, dyn
    [['algorithm', 'alg_kwargs', 'log_sigmas_freq'], [3]],
    [['algorithm', 'alg_kwargs', 'log_sigmas_kwargs', 'idx'], [100]],
    [['algorithm', 'alg_kwargs', 'log_sigmas_kwargs', 'n_ros'], [30]],
    [['algorithm', 'alg_kwargs', 'log_sigmas_kwargs', 'n_acs'], [30]],
    [['algorithm', 'alg_kwargs', 'log_sigmas_kwargs', 'n_taus'], [30]],
    [['algorithm', 'alg_kwargs', 'log_sigmas_kwargs', 'n_steps'], [None]],
    [['algorithm', 'alg_kwargs', 'log_sigmas_kwargs', 'use_vf'], [True, False]],
    [['policy', 'pol_kwargs', 'init_std'], [0.1, 0.3, 1.0, 3.0]],
]

ranges_upper = ranges_option + [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [100000]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [None]],
    [['oracle', 'or_kwargs', 'cv_type'], ['state']],
]

ranges_nocv = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['nocv']],
    [['oracle', 'or_cls', ], ['tfDoublyRobustPG']],
    [['cv_type_name'], ['nocv']],
]

ranges_st = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['state']],
    [['oracle', 'or_cls', ], ['tfDoublyRobustPG']],
    [['cv_type_name'], ['state']],
]

ranges_sa = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['new']],
    [['oracle', 'or_cls', ], ['tfDoublyRobustPG']],
    [['oracle', 'or_kwargs', 'stop_cv_step'], [1]],
    [['cv_type_name'], ['sa']],
]

ranges_traj = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['new']],
    [['oracle', 'or_kwargs', 'stop_cv_step'], [None]],
    [['oracle', 'or_cls', ], ['tfDoublyRobustPG']],
    [['oracle', 'or_kwargs', 'theta'], [0.9]],
    [['cv_type_name'], ['trajwise']],
]

ranges_dr = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['dr']],
    [['oracle', 'or_kwargs', 'stop_cv_step'], [None]],
    [['oracle', 'or_kwargs', 'theta'], [0.9]],
    [['oracle', 'or_kwargs', 'gamma2'], [0.9]],
    [['oracle', 'or_cls', ], ['tfDoublyRobustPG']],
    [['cv_type_name'], ['dr']],
]

ranges_sa_quad = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['quad']],
    [['oracle', 'or_kwargs', 'stop_cv_step'], [1]],
    [['oracle', 'or_kwargs', 'quad_style'], ['diff', 'next', 'diff-gn', 'next-gn']],
]


ranges_quad = ranges_option_cv + [
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [nros]],
    [['oracle', 'or_kwargs', 'cv_type'], ['quad']],
    [['oracle', 'or_kwargs', 'stop_cv_step'], [None]],
    [['oracle', 'or_kwargs', 'quad_style'], ['diff', 'next', 'diff-gn', 'next-gn']],
    [['oracle', 'or_kwargs', 'theta'], [1.0]],
]


def get_ranges(env, ranges_name, base_algorithms):
    """Combines ranges_common, ranges_env_specific, and ranges_algorithm_specific.

    When ranges_name ends with a positive integer, it specifies the number of iterations, e.g. mf_300.
    """
    ranges_lr = []
    ranges_env_specific = []
    ranges_exp_specific = globals()['ranges_' + ranges_name]

    # Take in account the case where we want to try an array of stepsize!
    # Give the option of specifying a range of etas for ONE alg.
    use_default_lr = True
    for r in ranges_exp_specific:
        if r[0] == ['algorithm', 'learning_rate', 'eta'] or r[0] == ['algorithm', 'base_alg']:
            use_default_lr = False
            break
    if use_default_lr:
        etas = STEPSIZE[env]['etas']
        c = STEPSIZE[env]['c']
        etas = [etas[alg] for alg in base_algorithms]  # get the etas that are needed
        ranges_lr = [
            [['algorithm', 'base_alg'], base_algorithms, ['algorithm', 'learning_rate', 'eta'], etas],
            [['algorithm', 'learning_rate', 'c'], [c]],
        ]
    # Number of iterations.
    split = ranges_name.split('_')
    if split[-1].isdigit():
        ranges_env_specific = [
            [['experimenter', 'run_alg_kwargs', 'n_itrs'], [int(split[-1])]],
        ]
        ranges_name = '_'.join(split[: -1])
    ranges = ranges_common + ranges_env_specific + ranges_lr + ranges_exp_specific

    return ranges
