import gym
import os
import pdb

from rl import envs
from rl.tools import supervised_learners as Sup
from rl.tools import normalizers as Nor
# from pybullet_envs.minitaur.agents.tools import BatchEnv
# from pybullet_envs.minitaur.agents.tools.wrappers import ExternalProcess


def create_batch_env(envid, seed, n_envs, render=False, use_ext_proc=True):
    def cons(): return create_env(envid, seed, render=False)
    func = cons if not use_ext_proc else lambda: ExternalProcess(cons)
    envs = [func() for _ in range(n_envs)]
    env = BatchEnv(envs, blocking=not use_ext_proc)
    return env


def create_env(envid, seed, render=False):
    """Create minitaur or other standard gym environment."""
    # if 'minitaur' in envid:
    #     from pybullet_envs.minitaur.agents.scripts import utility
    #     from pybullet_envs.minitaur.agents import tools
    #     config = utility.load_config(os.path.expanduser('minitaur_config'))
    #     if 'bad' in envid:
    #         with config.unlocked:
    #             config.env.keywords['accurate_motor_model_enabled'] = False
    #             config.env.keywords['control_latency'] = .0
    #             config.env.keywords['pd_latency'] = .0
    #             config.env.keywords['urdf_version'] = None
    #     env = config.env(render=render)
    #     if config.max_length:
    #         env = tools.wrappers.LimitDuration(env, config.max_length)
    #     env = tools.wrappers.RangeNormalize(env)
    #     env = tools.wrappers.ClipAction(env)
    #     env = tools.wrappers.ConvertTo32Bit(env)

    #     class MySpec(object):
    #         def __init__(self, max_episode_steps):
    #             self.max_episode_steps = max_episode_steps

    #     env.spec = MySpec(1000)
    # else:
    #     env = gym.make(envid)
    env = gym.make(envid)    

    # Set up seed.
    env.seed(seed)
    return env


# Env.
ENVID2MODELENV = {
    'DartCartPole-v1': envs.Cartpole,
    'DartHopper-v1': envs.Hopper,
    'DartSnake7Link-v1': envs.Snake,
    'DartWalker3d-v1': envs.Walker3d,
    'DartDog-v1': envs.Dog,
    # 'DartReacher-v1': envs.Reacher2D,
    'DartReacher3d-v1': envs.Reacher,
}


def create_sim_env(env, seed, inaccuracy=None, dc=None, rc=None):
    """ Create an EnvWithModel object as a model of env."""
    # dc: config for dynamics.
    def create_predict(config, out=None, seed=None):
        # Create predict of a supervised learner.
        # out_dim defaulted to be st_dim
        assert seed is not None
        st = env.reset()
        st_dim, ac_dim = len(st), env.action_space.shape[0]
        cls = getattr(Sup, config['fun_cls'])
        build_nor = Nor.create_build_nor_from_str(config['nor_cls'], config['nor_kwargs'])
        if out == 'dyn':
            out_dim = st_dim
        elif out == 'rw':
            out_dim = 1
        else:
            raise ValueError('Invalid out: {}'.format(out))

        # XXX learn residue for dynamics!
        svl = cls(st_dim + ac_dim, out_dim, 
                  name=config['name'],
                  seed=seed, build_nor=build_nor, 
                  **config['fun_kwargs'])
        return svl.predict, svl
    predict, predict_model = create_predict(dc, 'dyn', seed=seed+5) if dc is not None else (None, None)
    rw_fun, rw_model = None, None
    envid = env.env.spec.id
    
    sim_env = ENVID2MODELENV[envid](env, predict=predict, rw_fun=rw_fun,
                                model_inacc=inaccuracy, seed=seed, 
                                predict_model=predict_model, rew_model=rw_model)
                                
    return sim_env