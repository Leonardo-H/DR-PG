import pdb
import numpy as np
import os
from numpy import linalg as la
from rl.algorithms.algorithm import Algorithm
from rl.policies import Policy
from rl import oracles as Or
from rl.tools.utils.misc_utils import timed, safe_assign
from rl.tools.utils import logz

class SimpleCVRL(Algorithm):
    def __init__(self, learner, oracle, policy, update_pol_nor=True,
                 grad_std_n=10, grad_std_freq=None, log_sigmas_freq=None, log_sigmas_kwargs=None,
                 gen_sim_ro=None, ** kwargs):
        '''
        learner is online_optimizer
        '''
        self._learner = learner
        self._or = safe_assign(oracle, Or.rlOracle)
        self._policy = safe_assign(policy, Policy)
        self._itr = 0
        self._update_pol_nor = update_pol_nor
        self.gen_sim_ro = gen_sim_ro
        # self.log_sigmas_freq = log_sigmas_freq
        # self.log_sigmas_kwargs = log_sigmas_kwargs
        
    def pi(self, ob):
        return self._policy.pi(ob, stochastic=True)

    def pi_ro(self, ob):
        return self._policy.pi(ob, stochastic=True)

    def logp(self, obs, acs):
        return self._policy.logp(obs, acs)

    def pretrain(self, gen_ro, n_vf_updates=1, n_dyn_updates=1, n_rw_updates=1,
                 update_pol_nor=True, **kwargs):

        with timed('Pretraining'):
            ro = gen_ro(self.pi, logp=self.logp)
            if update_pol_nor:
                self._policy.prepare_for_update(ro.obs)  # update nor of policy
            ro = gen_ro(self.pi, logp=self.logp)
            for _ in range(n_vf_updates):
                self._or.update_ae(ro)
            for _ in range(n_dyn_updates):
                self._or.update_dyn(ro)
            for _ in range(n_rw_updates):
                self._or.update_rw(ro)

    def _update(self, env_ro, gen_env_ro):
        # gen_env_ro is just used for computing gradient std.
        assert gen_env_ro is not None

        # XXX If using simulation to train vf, vf should be updated after policy nor is updated.
        if self.gen_sim_ro is not None:
            with timed('Generate sim data'):
                sim_ro = self.gen_sim_ro()
            with timed('Update ae'):
                self._or.update_ae(sim_ro, to_log=True)  # update value function

        # if self.log_sigmas_freq is not None and self._itr % self.log_sigmas_freq == 0:
        #     with timed('Compute Sigmas'):
        #         self._or.log_sigmas(**self.log_sigmas_kwargs)

        with timed('Update Oracle'):
            self._or.update(env_ro, update_nor=True, to_log=True, itr=self._itr)
        with timed('Compute Grad'):
            grads = self._or.compute_grad(ret_comps=True)
            grad = grads[0]
            names = ['g', 'mc_g', 'ac_os', 'tau_os', 'dr_grad_os']
            
            for g, name in zip(grads, names):
                logz.log_tabular('norm_{}'.format(name), la.norm(g))

        with timed('Take Gradient Step'):
            self._learner.update(grad, self._or.ro)  # take the grad with the env_ro
        if self.gen_sim_ro is None:
            with timed('Update ae'):
                self._or.update_ae(env_ro, to_log=True)  # update value function
        # Always update dynamics using true data.
        with timed('Update dyn'):
            self._or.update_dyn(env_ro, to_log=True)  # update dynamics
        with timed('Update rw'):
            self._or.update_rw(env_ro, to_log=True)
        self._itr += 1
        logz.log_tabular('online_learner_stepsize', self._learner.stepsize)
        logz.log_tabular('std', np.mean(self._policy.std))

    def update(self, ro, gen_env_ro):
        if self._update_pol_nor:
            self._policy.prepare_for_update(ro.obs)
        self._update(ro, gen_env_ro)



class ComputeGrad(Algorithm):
    def __init__(self, learner, oracle, policy, update_pol_nor=True,
                 grad_std_n=10, grad_std_freq=None, log_sigmas_freq=None, log_sigmas_kwargs=None,
                 gen_sim_ro=None, ** kwargs):
        '''
        learner is online_optimizer
        '''
        self._learner = learner
        self._or = safe_assign(oracle, Or.rlOracle)
        self._policy = safe_assign(policy, Policy)
        self._itr = 0
        self._update_pol_nor = update_pol_nor
        # self.log_sigmas_freq = log_sigmas_freq
        # self.log_sigmas_kwargs = log_sigmas_kwargs
        
        self.gradients = None


        self.accum_ac = .0
        self.accum_func = .0
        self.accum_tau = .0

    def pi(self, ob):
        return self._policy.pi(ob, stochastic=True)

    def pi_ro(self, ob):
        return self._policy.pi(ob, stochastic=True)

    def logp(self, obs, acs):
        return self._policy.logp(obs, acs)
    
    def pretrain(self, gen_ro, n_vf_updates=1, n_dyn_updates=1, n_rw_updates=1,
                 update_pol_nor=True, **kwargs):
        pass

    def reset_grads(self,):
        self.gradients = None

    def _update(self, env_ro, gen_env_ro):
        # gen_env_ro is just used for computing gradient std.
        assert gen_env_ro is not None

        '''Set _ro to self._or '''
        with timed('Update Oracle'):
            # self.set_ro(env_ro)
            self._or.update(env_ro, update_nor=True, to_log=True, itr=self._itr)
        with timed('Compute Grad'):
            grads = self._or.compute_grad(ret_comps=True)
            grad = grads[0]
            if self.gradients is None:
                self.gradients = grad
            else:
                self.gradients = np.concatenate([self.gradients, grad], axis=0)

            names = ['g', 'mc_g', 'ac_os', 'tau_os', 'dr_grad_os']
            for g, name in zip(grads, names):
                logz.log_tabular('norm_{}'.format(name), la.norm(g))

            self.accum_ac += grads[2]
            self.accum_tau += grads[3]
            self.accum_func += grads[4]
            logz.log_tabular('norm_accum_ac_os', la.norm(self.accum_ac / self.gradients.shape[0]))
            logz.log_tabular('norm_accum_tau_os', la.norm(self.accum_tau / self.gradients.shape[0]))
            logz.log_tabular('norm_accum_func_os', la.norm(self.accum_func / self.gradients.shape[0]))

        self._itr += 1
        logz.log_tabular('std', np.mean(self._policy.std))

    def compute_grad(self, ro, gen_env_ro):
        # if self._update_pol_nor:
        #     self._policy.prepare_for_update(ro.obs)
        self._update(ro, gen_env_ro)

    def update(self, **kwargs):
        pass
