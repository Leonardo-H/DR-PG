import numpy as np
import tensorflow as tf
import git
import copy
import functools
import time
import os
import pdb

from rl.tools.utils import logz
from rl.tools.online_learners import base_algorithms as bAlg
from rl.tools.online_learners.scheduler import PowerScheduler
from rl.tools import supervised_learners as Sup
from rl.tools import normalizers as Nor
from rl.online_learners import online_optimizer as OO
from rl.experimenter.generate_rollouts import generate_rollout
from rl.adv_estimators import AdvantageEstimator
from rl import algorithms as Alg
from rl import oracles as Or
from rl import envs as Env
from rl import experimenter as Exp
from rl import policies as Pol


def configure_log(configs, unique_log_dir=False):
    """ Configure output directory for logging. """

    # parse configs to get log_dir
    c = configs['general']
    top_log_dir = c['top_log_dir']
    log_dir = c['exp_name']
    seed = c['seed']

    # create dirs
    os.makedirs(top_log_dir, exist_ok=True)
    if unique_log_dir:
        log_dir += '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(top_log_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, '{}'.format(seed))
    os.makedirs(log_dir, exist_ok=True)

    # Log commit number.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    configs['git_commit_sha'] = sha

    # save configs
    logz.configure_output_dir(log_dir)
    logz.save_params(configs)
    return log_dir


def general_setup(c):
    envid, seed = c['envid'], c['seed'],
    env = Env.create_env(envid, seed)
    # pdb.set_trace()
    if c['max_episode_steps'] is not None:
        env._max_episode_steps = c['max_episode_steps']
    # fix randomness
    tf.set_random_seed(seed)  # graph-level seed
    np.random.seed(seed)
    return env, seed


def create_policy(env, seed, c, name='learner_policy'):
    pol_cls = getattr(Pol, c['policy_cls'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    build_nor = Nor.create_build_nor_from_str(c['nor_cls'], c['nor_kwargs'])
    policy = pol_cls(ob_dim, ac_dim, name=name, seed=seed,
                     build_nor=build_nor, **c['pol_kwargs'])
    return policy


def create_advantage_estimator(policy, seed, adv_configs, name='value_function_approximator'):
    adv_configs = copy.deepcopy(adv_configs)
    # create the value function SupervisedLearner
    c = adv_configs['vfn_params']
    build_nor = Nor.create_build_nor_from_str(c['nor_cls'], c['nor_kwargs'])
    vfn_cls = getattr(Sup, c['fun_class'])
    vfn = vfn_cls(policy.ob_dim, 1, name=name, seed=seed, build_nor=build_nor, **c['fun_kwargs'])
    # create the adv object
    adv_configs.pop('vfn_params')
    adv_configs['vfn'] = vfn

    ae = AdvantageEstimator(policy, **adv_configs)
    return ae


def create_cv_oracle(policy, ae, c, env, seed):
    nor = Nor.NormalizerStd(None, **c['nor_kwargs'])

    var_env = create_env_from_env(env, 'true', seed*2)
    sim_env = create_env_from_env(env, c['env_type'], seed, c['dyn'], c['rw'])
    oracle = Or.tfDoublyRobustPG(policy, ae, nor,
                                            sim_env=sim_env, var_env=var_env,
                                            **c['or_kwargs'])
    return oracle

def create_experimenter_for_opt_exp(alg, env, ro_kwargs):
    # For interacting with the true env.
    gen_ro = functools.partial(generate_rollout, env=env, **ro_kwargs)
    return Exp.Experimenter(env, alg, gen_ro)

def create_experimenter_for_var_exp(alg, env, ro_kwargs):
    # For interacting with the true env.
    gen_ro = functools.partial(generate_rollout, env=env, **ro_kwargs)
    return Exp.Experimenter_for_Var_Exp(env, alg, gen_ro)

# dc is short for dynamic config (the transition model)
# rc is short for reward config (the reward model)
def create_env_from_env(env, env_type, seed, dc=None, rc=None):
    if env_type == 'true':
        new_env = Env.create_env(env.env.spec.id, seed)
        new_env._max_episode_steps = env._max_episode_steps  # set the horizon
        return new_env
    elif isinstance(env_type, float) and env_type >= 0.0 and env_type < 1.0:
        # the larger inaccuracy is, the larger randomness will be added. 
        return Env.create_sim_env(env, seed, inaccuracy=env_type)
    elif env_type == 'dyn':
        assert dc is not None
        return Env.create_sim_env(env, seed, dc=dc)
    elif env_type == 'dyn-rw':
        assert dc is not None and rc is not None
        return Env.create_sim_env(env, seed, dc=dc, rc=rc)
    else:
        raise ValueError('Unknown env_type {}'.format(env_type))


def _create_base_alg(policy, c):
    scheduler = PowerScheduler(**c['learning_rate'])
    x0 = policy.variable
    # create base_alg
    if c['base_alg'] == 'adam':
        base_alg = bAlg.Adam(x0, scheduler, beta1=c['adam_beta1'])
    elif c['base_alg'] == 'adagrad':
        base_alg = bAlg.Adagrad(x0, scheduler, rate=c['adagrad_rate'])
    elif c['base_alg'] == 'natgrad':
        base_alg = bAlg.AdaptiveSecondOrderUpdate(x0, scheduler)
    elif c['base_alg'] == 'rnatgrad':
        base_alg = bAlg.RobustAdaptiveSecondOrderUpdate(x0, scheduler)
    elif c['base_alg'] == 'trpo':
        base_alg = bAlg.TrustRegionSecondOrderUpdate(x0, scheduler)
    else:
        raise ValueError('Unknown base_alg')
    return base_alg


def create_cv_algorithm_for_opt_exp(policy, oracle, env, seed, c):
    base_alg = _create_base_alg(policy, c)
    online_optimizer = OO.BasicOnlineOptimizer(policy, base_alg, p=c['learning_rate']['p'],
                                               damping=c['damping'])        
    if c['train_vf_with_sim']:
        sim_env = create_env_from_env(env, c['env_type'], seed)
        gen_ro = functools.partial(generate_rollout,
                                    env=sim_env, pi=policy.pi, logp=None,
                                    **c['rollout_kwargs'])
    else:
        gen_ro = None
    alg = Alg.SimpleCVRL(online_optimizer, oracle, policy, gen_sim_ro=gen_ro, **c['alg_kwargs'])

    return alg


def create_cv_algorithm_for_var_exp(policy, oracle, env, seed, exp_type, c):
    base_alg = _create_base_alg(policy, c)
    online_optimizer = OO.BasicOnlineOptimizer(policy, base_alg, p=c['learning_rate']['p'],
                                               damping=c['damping'])            
    if exp_type == 'cal-var' or exp_type == 'est-mean' or exp_type == 'gen-ro':
        alg = Alg.ComputeGrad(None, oracle, policy, gen_sim_ro=None, **c['alg_kwargs'])
    else:
        if c['train_vf_with_sim']:
            sim_env = create_env_from_env(env, c['env_type'], seed)
            gen_ro = functools.partial(generate_rollout,
                                        env=sim_env, pi=policy.pi, logp=None,
                                        **c['rollout_kwargs'])
        else:
            gen_ro = None
        alg = Alg.SimpleCVRL(online_optimizer, oracle, policy, gen_sim_ro=gen_ro, **c['alg_kwargs'])

    return alg
