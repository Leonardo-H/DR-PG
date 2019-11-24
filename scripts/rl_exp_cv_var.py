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
    exp_type = c['experimenter']['exp_type']
    print('exp_type is ', exp_type)

    if exp_type == 'train':
        c['experimenter']['run_alg_kwargs']['save_policy'] = True
        c['experimenter']['run_alg_kwargs']['save_freq'] = 1
    else:
        c['experimenter']['run_alg_kwargs']['save_policy'] = False
        c['experimenter']['run_alg_kwargs']['save_freq'] = None

    # Create env and fix randomness
    # Assume that all envs will created from it.
    env, seed = ps.general_setup(c['general'])

    # Setup pdb mode.
    setup_pdb(c['general']['pdb_mode'])

    # Create objects for defining the algorithm
    policy = ps.create_policy(env, seed, c['policy'])
    ae = ps.create_advantage_estimator(policy, seed+1, c['advantage_estimator'])

    c['oracle']['or_kwargs']['exp_type'] = exp_type
    oracle = ps.create_cv_oracle(policy, ae, c['oracle'], env, seed+2)

    # Enter session.
    sess = U.single_threaded_session()
    sess.__enter__()
    tf.global_variables_initializer().run()

    alg = ps.create_cv_algorithm_for_var_exp(policy, oracle, env, seed+3, exp_type, c['algorithm'])
    exp = ps.create_experimenter_for_var_exp(alg, env, c['experimenter']['rollout_kwargs'])

    if exp_type == 'train': 
        c['general']['seed'] += 100
    elif exp_type == 'gen-ro':
        c['general']['seed'] += 200
    elif exp_type == 'cal-var':
        c['general']['seed'] += 300
    elif exp_type == 'est-mean':
        c['general']['seed'] += 400
    else:
        raise NotImplementedError

    if exp_type == 'train':
        log_dir = ps.configure_log(c)
        save_dir = './Model'

        def save_policy_fun(name):
            policy.save(path=os.path.join(save_dir, name + '_pol.ckpt'))
            policy._nor._tf_params.save(path=os.path.join(save_dir, name + '_polnor.ckpt'))

        def save_value_fun(name):
            ae.save_vfn(save_dir, name)

        def save_sim_fun(name):
            oracle.save_sim_env(save_dir, name)

        exp.run_alg(save_policy_fun=save_policy_fun, 
                    save_value_fun=save_value_fun, 
                    save_sim_fun=save_sim_fun,
                    **c['experimenter']['run_alg_kwargs'],
                    **c['experimenter']['pretrain_kwargs'])  

    else:
        if not os.path.exists('./Rollouts'):
            os.makedirs('./Rollouts')
        all_pairs = c['all_pairs']
        for _, prefix in all_pairs:

            prefix = prefix
            # Setup logz and save c
            log_dir = ps.configure_log(c) + prefix

            ro_file = './Rollouts/' + 'rollouts_' + prefix + '.pickle'

            if exp_type in ['gen-ro', 'est-mean', 'cal-var']:
                name = './Model/' + prefix
                restore_path_prefix = name

                restore_dyn_path_prefix = restore_path_prefix
                vfn_path_prefix = restore_path_prefix + '_vfn'
                
                if restore_path_prefix is not None:
                    policy_path_prefix = restore_path_prefix + '_pi'
                    policy.restore(policy_path_prefix + '_pol.ckpt')
                    policy._nor._tf_params.restore(policy_path_prefix + '_polnor.ckpt')

                    ae.restore_vfn(prefix=vfn_path_prefix)

                    dyn_path_prefix = restore_dyn_path_prefix + 'sim_dyn'
                    oracle.restore_sim_env(dyn_path_prefix=dyn_path_prefix, rw_path_prefix=None)
                
            np_file_name = prefix + '_seed_' + str(c['general']['seed']) + '_' + c['cv_type_name']
            if c['cv_type_name'] == 'trajwise' or c['cv_type_name'] == 'dr':
                np_file_name = np_file_name + '_' + str(c['oracle']['or_kwargs']['theta']) + '_' + str(c['oracle']['or_kwargs']['gamma2'])
            dir_name = os.path.join('./data/', c['cv_type_name'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                                
            if exp_type == 'gen-ro':
                print('Generate Rollouts for Model ' + prefix)
                exp.collect_ro(save_policy_fun=None, 
                            save_value_fun=None, 
                            save_sim_fun=None,
                            ro_num=c['experimenter']['ro'],
                            ro_file=ro_file,
                            prefix=prefix,
                            **c['experimenter']['run_alg_kwargs'],
                            **c['experimenter']['pretrain_kwargs'])                 
            elif exp_type == 'est-mean':
                save_np_file_path = os.path.join('./data/', np_file_name + '_st_mean_.npy')
                exp.est_mean(save_policy_fun=None, 
                            save_value_fun=None, 
                            save_sim_fun=None,
                            save_np_file_path=save_np_file_path,
                            **c['experimenter']['run_alg_kwargs'],
                            **c['experimenter']['pretrain_kwargs'])
            elif exp_type == 'cal-var':
                save_np_file_path = os.path.join(dir_name, np_file_name + '.npy')
                exp.cal_variance(save_policy_fun=None, 
                            save_value_fun=None, 
                            save_sim_fun=None,
                            ro_file=ro_file,
                            prefix=prefix,
                            save_np_file_path=save_np_file_path,
                            **c['experimenter']['run_alg_kwargs'],
                            **c['experimenter']['pretrain_kwargs'])
            else:
                raise NotImplementedError
