import pdb
import functools
import time
import numpy as np
import pickle
import os
from rl.algorithms import Algorithm
from rl.tools.utils.misc_utils import safe_assign, timed, cprint
from rl.tools.utils import logz
from rl.experimenter.generate_rollouts import generate_rollout
from rl.experimenter.rollout import RO


class Experimenter_for_Var_Exp:
    def __init__(self, env, alg, gen_ro):
        self._env = env
        self._alg = safe_assign(alg, Algorithm)
        self._gen_ro_raw = gen_ro
        self._gen_ro = functools.partial(gen_ro, pi=self._alg.pi_ro, logp=self._alg.logp)
        self._ndata = 0  # number of data points seen

    def set_seed(self, seed):
        self.seed = seed

    def set_name(self, cv_type_name):
        self.cv_type_name = cv_type_name

    def gen_ro(self, log_prefix='', to_log=False):
        ro = self._gen_ro()
        self._ndata += ro.n_samples
        if to_log:
            log_rollout_info(ro, prefix=log_prefix)
            logz.log_tabular(log_prefix + 'NumberOfDataPoints', self._ndata)
        return ro

    def run_alg(self, n_itrs, save_policy=True, save_policy_fun=None, save_freq=3,
                save_value_fun=None, save_sim_fun=None,
                pretrain=True, rollout_kwargs=None, **other_pretrain_kwargs):
        start_time = time.time()
        if pretrain:  # algorithm-specific
            if rollout_kwargs is None:
                gr = self._gen_ro_raw
            elif (rollout_kwargs['max_n_rollouts'] is None and
                  rollout_kwargs['min_n_samples'] is None):
                gr = self._gen_ro_raw
            else:
                gr = functools.partial(generate_rollout, env=self._env, **rollout_kwargs)
            self._alg.pretrain(gr, **other_pretrain_kwargs)

        # Main loop
        for itr in range(n_itrs):
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)
            with timed('Generate env rollouts'):
                ro = self.gen_ro(to_log=True)
            
            # algorithm-specific
            if save_policy and isinstance(save_freq, int) and itr % save_freq == 0:
                mean_val = logz.get_val_from_LOG('MeanSumOfRewards')
                prefix = 'iter_{}_eval_'.format(itr) + '%.0f' % mean_val
                save_policy_fun(prefix + '_pi')
                save_value_fun(prefix + '_vfn')
                save_sim_fun(prefix + 'sim')
            self._alg.update(ro, gen_env_ro=self._gen_ro) 
            logz.dump_tabular()  # dump log

        # Save the final policy.
        if save_policy:
            save_policy_fun('final')
            cprint('Final policy has been saved.')

    def est_mean(self, n_itrs, save_policy=True, save_value_fun=None, 
                save_policy_fun=None, save_freq=3,
                save_sim_fun=None,
                save_gradient=None,
                save_np_file_path=None,
                pretrain=True, rollout_kwargs=None, **other_pretrain_kwargs):
        start_time = time.time()        

        # Main loop
        # default estimator number is 10000
        for itr in range(2000):
            ro = self._gen_ro()       

            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)

            # algorithm-specific
            self._alg.compute_grad(ro, gen_env_ro=self._gen_ro)
            logz.dump_tabular()  # dump log

        mean_st = self._alg.gradients       
        est_mean = np.mean(mean_st, axis=0, keepdims=True)     
        np.save(save_np_file_path, est_mean)

    def cal_variance(self, n_itrs, save_policy=True, save_value_fun=None, 
                save_policy_fun=None, save_freq=3,
                save_sim_fun=None,
                ro_file=None,
                save_np_file_path=None,
                prefix=None,
                save_gradient=None,
                pretrain=True, rollout_kwargs=None, **other_pretrain_kwargs):
        start_time = time.time()
        
        assert prefix is not None
        assert ro_file is not None
        assert save_np_file_path is not None

        save_grad_frequency = 1

        ro_path = ro_file

        with open(ro_path, 'rb') as f:
            ros = pickle.load(f)

        # Main loop        
        itr = 0
        self._alg.reset_grads()
        while True:
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)
            
            self._alg.compute_grad(ros.pop(0), gen_env_ro=self._gen_ro)

            logz.dump_tabular()  # dump log
            
            if itr % save_grad_frequency == 0:
                np.save(save_np_file_path, self._alg.gradients)

            if ros == []:
                break
            
            itr += 1

    def collect_ro(self, n_itrs, save_policy=True, save_value_fun=None, 
                save_policy_fun=None, save_freq=3,
                save_sim_fun=None,
                save_gradient=None,
                ro_file=None,
                prefix=None,
                ro_num=500,
                pretrain=True, rollout_kwargs=None, **other_pretrain_kwargs):

        assert prefix is not None
        assert ro_file is not None
        
        all_rollouts = []
        iters = ro_num // 5
        for _ in range(iters):
            ro = self._gen_ro()            
            all_rollouts += [ro]

        with open(ro_file, 'wb') as f:
            pickle.dump(all_rollouts, f)

def log_rollout_info(ro, prefix=''):
    # print('Logging rollout info')
    if not hasattr(log_rollout_info, "total_n_samples"):
        log_rollout_info.total_n_samples = {}  # static variable
    if prefix not in log_rollout_info.total_n_samples:
        log_rollout_info.total_n_samples[prefix] = 0
    sum_of_rewards = [rollout.rws.sum() for rollout in ro.rollouts]
    rollout_lens = [len(rollout) for rollout in ro.rollouts]
    n_samples = sum(rollout_lens)
    log_rollout_info.total_n_samples[prefix] += n_samples
    logz.log_tabular(prefix + "NumSamplesThisBatch", n_samples)
    logz.log_tabular(prefix + "NumberOfRollouts", len(ro))
    logz.log_tabular(prefix + "TotalNumSamples", log_rollout_info.total_n_samples[prefix])
    logz.log_tabular(prefix + "MeanSumOfRewards", np.mean(sum_of_rewards))
    logz.log_tabular(prefix + "StdSumOfRewards", np.std(sum_of_rewards))
    logz.log_tabular(prefix + "MaxSumOfRewards", np.max(sum_of_rewards))
    logz.log_tabular(prefix + "MinSumOfRewards", np.min(sum_of_rewards))
    logz.log_tabular(prefix + "MeanRolloutLens", np.mean(rollout_lens))
    logz.log_tabular(prefix + "StdRolloutLens", np.std(rollout_lens))
    logz.log_tabular(prefix + "MeanOfRewards", np.sum(sum_of_rewards) / (n_samples + len(sum_of_rewards)))
