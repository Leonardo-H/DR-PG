import os
import argparse
import tensorflow as tf
import pdb

from scripts import configs_cv as C
from rl.configs import parser as ps
from rl.tools.utils import tf_utils as U


def setup_pdb(mode):
    if mode == 'on':
        pdb.set_trace_force = pdb.set_trace
        return
    elif mode == 'off':
        pdb.set_trace_force = lambda: 1
        pdb.set_trace = lambda: 1
    elif mode == 'selective':
        pdb.set_trace_force = pdb.set_trace
        pdb.set_trace = lambda: 1

    else:
        raise ValueError('Unknown pdb_mode: {}'.format(mode))


def main(c):

    # Setup logz and save c
    log_dir = ps.configure_log(c)

    # Create env and fix randomness
    # Assume that all envs will created from it.
    env, seed = ps.general_setup(c['general'])

    # Setup pdb mode.
    setup_pdb(c['general']['pdb_mode'])

    # Create objects for defining the algorithm
    policy = ps.create_policy(env, seed, c['policy'])
    ae = ps.create_advantage_estimator(policy, seed+1, c['advantage_estimator'])
    oracle = ps.create_cv_oracle(policy, ae, c['oracle'], env, seed+2)

    # Enter session.
    U.single_threaded_session().__enter__()
    tf.global_variables_initializer().run()

    # Restore policy if necessary.
    if c['policy']['restore_path'] is not None:
        policy.restore(c['policy']['restore_path'] + '_pol.ckpt')
        policy._nor._tf_params.restore(c['policy']['restore_path'] + '_polnor.ckpt')

    # Create algorithm after initializing tensors, since it needs to read the
    # initial value of the policy.
    alg = ps.create_cv_algorithm_for_opt_exp(policy, oracle, env, seed+3, c['algorithm'])

    # Let's do some experiments!
    exp = ps.create_experimenter_for_opt_exp(alg, env, c['experimenter']['rollout_kwargs'])

    def save_policy_fun(name):
        policy.save(path=os.path.join(log_dir, name + '_pol.ckpt'))
        policy._nor._tf_params.save(path=os.path.join(log_dir, name + '_polnor.ckpt'))

    exp.run_alg(save_policy_fun=save_policy_fun, **c['experimenter']['run_alg_kwargs'],
                **c['experimenter']['pretrain_kwargs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs_name', type=str)
    args = parser.parse_args()
    configs = getattr(C, 'configs_' + args.configs_name)
    main(configs)