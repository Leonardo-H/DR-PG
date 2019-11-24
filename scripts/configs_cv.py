import copy
from rl.tools.utils.misc_utils import dict_update


configs = {
    'general': {
        'top_log_dir': 'log',
        'envid': 'DartCartPole-v1',
        'seed': 0,
        'exp_name': 'cp',
        # 'on', 'off', 'selective', whether stop for pdb.set_trace()
        # when it's 'selective', only stops for pdb.set_trace_force()
        'pdb_mode': 'off',
        'max_episode_steps': None,
    },
    'experimenter': {
        'pretrain_kwargs': {
            'pretrain': True,
            'update_pol_nor': True,
            'n_vf_updates': 1,
            'n_dyn_updates': 1,
            'n_rw_updates': 1,
            'rollout_kwargs': {
                'min_n_samples': None,
                'max_n_rollouts': None,
                'max_rollout_len': None,
            },
        },
        'run_alg_kwargs': {
            'n_itrs': 50,
            'save_policy': False,
            'save_freq': None,
            # used when pretrain is True. if all None, will be set to the same as rollout_kwargs.
        },
        'rollout_kwargs': {
            'max_n_rollouts': None,
            'min_n_samples': None,
            'max_rollout_len': None,  # the max length of rollouts in training
        },
    },
    'algorithm': {
        'alg_cls': 'SimpleCVRL',  # 'SimpleRL', 
        'base_alg': 'trpo',  # 'natgrad', 'adam', 'adagrad', 'trpo'
        'train_vf_with_sim': False,  # for SimpleCVRL, train vf using simulation data
        # used for model_based, dyna of SimpleRL, or for SimpleCVRL, env for training vf
        'env_type': 'true',
        # used for model_based, dyna of SimpleRL, or for SimpleCVRL, for training vf
        'rollout_kwargs': {
            'min_n_samples': None,
            'max_n_rollouts': None,
            'max_rollout_len': None,  # the max length of rollouts in training
        },
        'damping': 0.1,  # used for second-order base_alg
        'adam_beta1': 0.9,  # used for adam base_alg
        'adagrad_rate': 0.5,  # used for adagrad base_alg
        'alg_kwargs': {
            'update_rule': 'model_free',  # used for SimpleRL. 'model_based', 'model_free', 'dyna'
            'update_pol_nor': True,  
            'log_sigmas_freq': None,  # the iteration number when variance will be logged
            'log_sigmas_kwargs': {
                'idx': None,  # G_idx
                'n_ros': 30,
                'n_acs': 30,
                'n_taus': 30,
                'n_steps': None,
                'use_vf': False,
            },  # for SimpleCVRL, and cv_type 'new'
        },
        'learning_rate': {
            'p': 0,
            'eta': 0.01,
            'c': 0.01,
            'limit': None,
        },
    },
    'oracle': {
        'or_cls': 'tfDoublyRobustPG',
        # used for tfPolicyGradientWithCV, and tfPolicyGradientSimpleCV, 'true', 'dyn' for learning dynamics, or 'dyn-rw' for learning both dynamics and reward function.
        'env_type': 'dyn-rw',
        # For tfPolicyGradientSimpleCV, since just need to predict next state, regressor is enough
        'or_kwargs': {
            'cv_type': 'nocv',   # for SimpleCVRL, 'nocv', 'state', 'new'
            'switch_at_itr': None,
            'cv_onestep_weighting': False,
            'stop_cv_step': 1,  # for cv_type 'new'
            'theta': 1.0,  # for cv_type 'new'
            'gamma2':0.9,
            'quad_style': 'diff',  # how the quadratic approximation is constructed
            'dyn_update_weights_type': 'one',  # for tfPolicyGradientSimpleCV
            'rw_update_weights_type': 'one',  # for tfPolicyGradientSimpleCV
            'n_ac_samples': 100,  # for cv_type 'new'
            'avg_type': 'sum',  # 'avg' or 'sum'
            'correlated': True,  # False, # update adv nor before computing adv
            'use_log_loss': True,
            'normalize_weighting': False,
            'onestep_weighting': False,  # XX turned off
        },
        'nor_kwargs': {  # simple baseline substraction
            'rate': .0,
            'momentum': 0,  # 0 for instant, None for moving average
            'clip_thre': None,
            'unscale': True,
            'unbias': False,
        },
        # For learning the reward function.
        'rw': {
            'nor_cls': 'tfNormalizerMax',
            'name': 'or_rw',
            'nor_kwargs': {
                'rate': 0.0,
                'momentum': None,
                'clip_thre': 5.0,
            },
            'fun_cls': 'tfMLPSupervisedLearner',
            'fun_kwargs': {
                'use_aggregation': True,  # aggregate all of the previous data
                'max_n_samples': 200000,
                'learning_rate': 1e-3,
                'batch_size': 128,  # small batch size
                'n_batches': 1024,  # 625,
                'n_epochs': 500,
                'batch_size_for_prediction': 100000,
                'size': 64,
                'n_layers': 2,
            },
        },
        'dyn': {
            'nor_cls': 'tfNormalizerMax',
            'name': 'or_dyn',
            'nor_kwargs': {
                'rate': 0.0,
                'momentum': None,
                'clip_thre': 5.0,
            },
            'fun_cls': 'tfMLPSupervisedLearner',
            'fun_kwargs': {
                'pred_residue': True,  # XXX predict residue!!!
                'use_aggregation': True,
                'max_n_samples': 200000,
                'learning_rate': 1e-3,
                'batch_size': 128,  # small batch size
                'n_batches': 1024,  # 625,
                'n_epochs': 500,
                'batch_size_for_prediction': 100000,
                'size': 64,
                'n_layers': 2,
            },
        },
    },
    'policy': {
        'restore_path': None,
        'nor_cls': 'tfNormalizerMax',
        'nor_kwargs': {
            'rate': 0.0,
            'momentum': None,
            'clip_thre': 5.0,
        },
        'policy_cls': 'tfGaussianMLPPolicy',
        'pol_kwargs': {
            'size': 32,  # 64,
            'n_layers': 1,  # 2
            'init_std': 0.37,
            'max_to_keep': 100,
        },
    },
    'advantage_estimator': {
        'gamma': 1.0,  # discount in the problem definition, 1.0 for undiscounted problem
        'delta': 0.99,  # discount to make MDP well-behave, or variance reduction parameter
        'lambd': 0.98,   # 0, 0.98, GAE parameter, in the estimation of on-policy value function
        'default_v': 0.0,  # value of the absorbing state
        # 'monte-carlo' (lambda = 1), 'td' (lambda = 0), 'same' (as GAE lambd),
        # or a float number that specifies the lambda in TD(lambda) for learning vf.
        # XX None to turn off the usage of value function.
        'v_target': 0.98,
        'onestep_weighting': False,  # whether to use one-step importance weight (only for value function learning)
        'multistep_weighting': False,  # whether to use multi-step importance weight
        'data_aggregation': False,
        'max_n_rollouts': None,  # for data aggregation
        'n_updates': 1,
        'vfn_params': {
            'nor_cls': 'tfNormalizerMax',
            'nor_kwargs': {
                'rate': 0.0,
                'momentum': None,
                'clip_thre': 5.0,
            },
            'fun_class': 'tfMLPSupervisedLearner',
            'fun_kwargs': {
                'use_aggregation': False,
                'max_n_update_samples': None,
                'learning_rate': 1e-3,
                'batch_size': 128,  # small batch size
                'n_batches': 1024,
                'n_epochs': 500,
                'batch_size_for_prediction': 100000,
                'size': 64,
                'n_layers': 2,
            },
        },
    },
    'cv_type_name':{
        'dr',
    },
}


# Cartpole.
configs_cp = configs

# Hopper.
t = {
    'general': {
        'exp_name': 'hp',
        'envid': 'DartHopper-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 500},
    },
}
configs_hopper = copy.deepcopy(configs)
dict_update(configs_hopper, t)

# Reacher.
t = {
    'general': {
        'exp_name': 'reacher',
        'envid': 'DartReacher3d-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 200},
    },
}
configs_reacher = copy.deepcopy(configs)
dict_update(configs_reacher, t)


# Dog.
t = {
    'general': {
        'exp_name': 'dog',
        'envid': 'DartDog-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 500},
    },
}
configs_dog = copy.deepcopy(configs)
dict_update(configs_dog, t)

# Walker2d.
t = {
    'general': {
        'exp_name': 'walker2d',
        'envid': 'DartWalker2d-v1',
    },
    'experimenter': {
        'run_alg_kwargs': {'n_itrs': 500},
    }
}
configs_walker2d = copy.deepcopy(configs)
dict_update(configs_walker2d, t)


# Test the new control variate.
t = {
    'general': {
        'seed': 200,
        'exp_name': 'cp',
    },
    'experimenter': {
        'rollout_kwargs': {'max_n_rollouts': 10},
        'pretrain_kwargs': {
            'pretrain': True,
            'rollout_kwargs': {'min_n_samples': 500},
            'update_pol_nor': True,
        },
        'run_alg_kwargs': {
            'n_itrs': 50,
        },
    },
    'advantage_estimator': {
        'gamma': 1.0,
        'delta': 0.98,
        'v_target': 1.0,
        'lambd': 1.0,
        'data_aggregation': False,
    },
    'algorithm': {
        'alg_cls': 'SimpleCVRL',
        'base_alg': 'rnatgrad', 
        'alg_kwargs': {
            'update_pol_nor': False,
            'stop_std_grad': False,
        },
        'learning_rate': {
            'eta': 0.1,
            'c': 0.01,
        },
        'train_vf_with_sim': False,
        'rollout_kwargs': {
            'min_n_samples': 100
        },
    },
    'oracle': {
        'or_cls': 'tfDoublyRobustPG',
        'env_type': 'dyn',
        # 'env_type': 'dyn',  # learn both dyn and rw
        'or_kwargs': {
            'n_ac_samples': 200,
            'cv_type': 'dr',  # 'state', 'nocv', 'new', 'quad', 'dr'
            'stop_cv_step': None,  # for cv_type 'new'
            'theta': 0.98,  # for cv_type 'new'
            'gamma2':0.9,
            'quad_style': 'next-gn',
            'switch_at_itr': None,
            'cv_onestep_weighting': True,
        },
        'dyn': {
            'fun_kwargs': {
                'max_n_samples': 50000,
            }
        },
    },
    'policy': {
        'nor_kwargs': {'clip_thre': 5.0},
    },
    'cv_type_name':{
        'dr',
    },
}
configs_newcvtest = copy.deepcopy(configs)
dict_update(configs_newcvtest, t)
