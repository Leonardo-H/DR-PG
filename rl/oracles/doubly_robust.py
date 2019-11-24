import pdb
import math
import numpy as np
import functools
import tensorflow as tf
import os
import time

from scipy import linalg as la
from rl.tools.oracles import tfLikelihoodRatioOracle
from rl.oracles.oracle import rlOracle
from rl.policies import tfPolicy, tfGaussianPolicy
from rl.tools.normalizers import OnlineNormalizer
from rl.tools.utils.tf_utils import tfObject
from rl.tools.utils import logz
from rl.tools.utils import tf_utils as U
from rl.experimenter.rollout import RO
from rl.experimenter.generate_rollouts import generate_rollout
from rl.tools.function_approximators import online_compatible
from rl.tools.utils.misc_utils import timed, unflatten, cprint
from rl.oracles.reinforcement_oracles import tfPolicyGradient

class tfDoublyRobustPG(tfPolicyGradient):
    
    # Natural ordering CV.
    @tfObject.save_init_args()
    def __init__(self, policy, ae, nor,
                 correlated=True, use_log_loss=False, normalize_weighting=False, onestep_weighting=True, avg_type='avg',
                 sim_env=None, n_ac_samples=0, cv_type='nocv', stop_cv_step=1, theta=1.0, gamma2=1.0,
                 quad_style='diff',
                 dyn_update_weights_type='one',
                 rw_update_weights_type='one',
                 var_env=None,
                 switch_at_itr=None,
                 cv_onestep_weighting=False,
                 traj_len=30,
                 exp_type=None,
                 num_traj_for_grad_q=20,
                 num_grad_q_for_grad_v=20,
                 random_pairs=False,        # whether to choose random (s,a) pair, instead of all (s,a)
                 **kwargs):
        # var_env: env for computing variance.
        # Only implemented this version for now.
        assert correlated is True  # update adv nor before normalizing adv, adv nor NOT used actually
        assert normalize_weighting is False
        assert use_log_loss is True
        assert avg_type == 'sum'
        assert onestep_weighting is False
        assert np.isclose(ae._pe.gamma, 1.0)  # undiscounted problem
        assert np.isclose(ae._pe.lambd, 1.0)  # telescoping sum, no GAE
        assert ae._v_target is not None  # vf should be on
        assert sim_env is not None  # current way of computing q
        assert nor is not None


        tfPolicyGradient.__init__(self, policy, ae, nor, correlated, use_log_loss,
                                  normalize_weighting, onestep_weighting, avg_type)
        self.sim_env = sim_env
        self.adv_nor = nor  # not used yet
        self.ac_dim = policy.y_dim
        self.ob_dim = policy.x_dim
        self.n_ac_samples = n_ac_samples
        self.delta = ae._pe.delta  # the discount factor used in vf definition
        self.ae = ae 

        # sa is implemented as a special case of traj
        assert cv_type in ['nocv', 'state', 'new', 'dr']  

        self.cv_type = cv_type
        self.stop_cv_step = stop_cv_step
        self.dyn_update_weights_type = dyn_update_weights_type
        self.rw_update_weights_type = rw_update_weights_type
        self.gen_ro = functools.partial(generate_rollout, env=var_env,
                                        pi=self.policy.pi, logp=None, min_n_samples=None)
        # extra decay
        self.theta = theta
        self.gamma2 = gamma2
        self.quad_style = quad_style
        self.cv_onestep_weighting = cv_onestep_weighting
        # For traj cv, first do several steps of state cv to warm up.
        self.switch_at_itr = switch_at_itr
        self.switched = False
        if self.switch_at_itr is not None:
            self.saved_cv_type = self.cv_type
            self.cv_type = 'state'  # switch back at iteration switch_at_itr

        self.traj_len = traj_len
        self.num_traj_for_grad_q = num_traj_for_grad_q
        self.num_grad_q_for_grad_v = num_grad_q_for_grad_v

        self.exp_type = exp_type

        if self.cv_type == 'dr':
            self.build_approx_grad_q_network()
                        

    def save_sim_env(self, log_dir, name):
        if hasattr(self.sim_env, 'get_predict_model') and self.sim_env.get_predict_model() is not None:
            self.sim_env.get_predict_model().save(path=os.path.join(log_dir, name + '_dyn_pol.ckpt'))
            self.sim_env.get_predict_model()._nor._tf_params.save(path=os.path.join(log_dir, name + '_dyn_polnor.ckpt'))

        if hasattr(self.sim_env, 'get_rew_model') and self.sim_env.get_rew_model() is not None:
            self.sim_env.get_rew_model().save(path=os.path.join(log_dir, name + '_rw_pol.ckpt'))
            self.sim_env.get_rew_model()._nor._tf_params.save(path=os.path.join(log_dir, name + '_rw_polnor.ckpt'))
        
    def restore_sim_env(self, dyn_path_prefix, rw_path_prefix):
        if self.sim_env.predict_model is not None:
            self.sim_env.predict_model.restore(dyn_path_prefix + '_pol.ckpt')
            self.sim_env.predict_model._nor._tf_params.restore(dyn_path_prefix + '_polnor.ckpt')

        if self.sim_env.rew_model is not None:
            self.sim_env.rew_model.restore(rw_path_prefix + '_pol.ckpt')
            self.sim_env.rew_model._nor._tf_params.restore(rw_path_prefix + '_polnor.ckpt')

    def update(self, ro, update_nor=False, to_log=False, log_prefix='', itr=None, **kwargs):
        if (itr is not None and self.switch_at_itr is not None and
                itr >= self.switch_at_itr and not self.switched):
            cprint('Switch to fancy cv: {} from {}'.format(self.saved_cv_type, self.cv_type))
            self.cv_type = self.saved_cv_type
            self.switched = True
        self._ro = ro

    def set_ro(self, ro_):
        self._ro = ro_

    def compute_grad(self, ret_comps=False):
        mc, ac_os, tau_os, func_os = .0, .0, .0, .0
        grads_list = []

        # pdb.set_trace()
        if self.cv_onestep_weighting:
            onestep_ws = self._ae.weights(self._ro, policy=self.policy)
        else:
            onestep_ws = np.ones(len(self._ro))

        for i, r in enumerate(self._ro.rollouts):
            cur_mc, cur_ac_os, cur_tau_os, cur_func_os = .0, .0, .0, .0
            decay = self.ae._pe.gamma * self.delta
            ws = decay ** np.arange(len(r))
            Ws = np.triu(la.circulant(ws).T, k=0)
            qs = np.ravel(np.matmul(Ws, r.rws[:, None]))
            gd = self.prepare_grad_data(r)
            cur_mc = self.policy.nabla_logp_f(r.obs_short, r.acs, qs)      # gradient estimated via MC;
            mc += cur_mc
            
            # CV for the first action, state (action) dependent CV.
            # for state baseline / MC, this term should be 0
            if self.cv_type != 'nocv':
                cur_ac_os = self.policy.nabla_logp_f(r.obs_short, r.acs,
                                            gd.qs * onestep_ws[i]) - gd.grad_exp_qs
                ac_os += cur_ac_os

            # CV for the future trajectory (for each t: \delta Q_{t+1} + ... + \delta^{step} Q_{t+step})
            # Note that it starts from t+1.
            # for sa/MC, those term should be 0
            if not (self.cv_type == 'nocv' or \
                    self.cv_type == 'new' and self.stop_cv_step == 1):
                if len(np.array(gd.Ws).shape) == 0:
                    tau_cvs = gd.Ws*(gd.qs * onestep_ws[i] - gd.exp_qs)
                else:
                    tau_cvs = np.ravel(np.matmul(gd.Ws, (gd.qs * onestep_ws[i]-gd.exp_qs)[:, None]))
                cur_tau_os = self.policy.nabla_logp_f(r.obs_short, r.acs, tau_cvs)
                tau_os += cur_tau_os

            if self.cv_type == 'dr':
                cur_func_os = gd.dr_grad_q - gd.dr_exp_grad_q
                func_os += cur_func_os

            cur_grad = - (cur_mc - (cur_ac_os + cur_tau_os + cur_func_os))
            grads_list.append(cur_grad.reshape([1, 194]))

        # Average.
        mc /= len(self._ro)
        ac_os /= len(self._ro)
        tau_os /= len(self._ro)
        func_os /= len(self._ro)

        g = - (mc - (ac_os + tau_os + func_os))  # gradient ascent
        if ret_comps:
            if self.exp_type in ['train', 'gen-ro', None]:
                return g, mc, ac_os, tau_os, func_os
            else:
                return np.concatenate(grads_list, axis=0), mc, ac_os, tau_os, func_os
        else:
            return g

    def prepare_grad_data(self, r):
        # r: a rollout object
        class GradDataDR(object):
            def __init__(self, qs, exp_qs, grad_exp_qs, *, 
                        dr_grad_q=None, dr_exp_grad_q=None, 
                        dr_count_q=None, dr_count_v=None,
                        decay=None, stop_cv_step=None):
                self.qs = qs  # T
                self.exp_qs = exp_qs  # T
                self.grad_exp_qs = grad_exp_qs  # d (already sum over the trajectory)

                '''
                The gradient of the q function, 
                    consider q as a function, 
                    rather than a deterministic value given (st, at)
                '''
                self.dr_grad_q = dr_grad_q
                self.dr_exp_grad_q = dr_exp_grad_q
                self.dr_count_q = dr_count_q
                self.dr_count_v = dr_count_v

                if decay is not None:
                    ws = decay ** np.arange(len(r))
                    if stop_cv_step is not None:
                        ws[min(stop_cv_step, len(r)):] = 0
                    if stop_cv_step == 1:       # sa case, the we do not need to calculate this one
                        Ws = None
                    else:
                        Ws = np.triu(la.circulant(ws).T, k=1)  # XXX WITHOUT the diagonal terms!!!!
                else:
                    Ws = 1.0
                self.Ws = Ws  # T * T

        if self.cv_type == 'nocv':
            qs = exp_qs = np.zeros(len(r))
            grad_exp_qs = 0.
            grad_data = GradDataDR(qs, exp_qs, grad_exp_qs)

        elif self.cv_type == 'state':
            qs = exp_qs = np.ravel(self.ae._vfn.predict(r.obs_short))
            grad_exp_qs = 0.
            grad_data = GradDataDR(qs, exp_qs, grad_exp_qs)

        elif self.cv_type == 'new':
            '''
            Use reparameterization (trick) to calculate the expectation;
                First sample multiple random values to get multiple random actions, 
                and compute q/v/nabla_logp_f for each action, then calculate the mean value.
            '''
            qs = self.compute_q(r.obs_short, r.acs, r.sts_short)
            # Sample extra actions for approximating the required expectations.
            # (repeat the same obs for many times consecutively)
            obs_exp = np.repeat(r.obs_short, self.n_ac_samples, axis=0)
            sts_exp = np.repeat(r.sts_short, self.n_ac_samples, axis=0)
            # sample the same randomness for all steps
            rand = np.random.normal(size=[self.n_ac_samples, self.ac_dim])
            rand = np.tile(rand, [len(r), 1])
            acs_exp = self.policy.pi_given_r(obs_exp, rand)
            qs_exp = self.compute_q(obs_exp, acs_exp, sts_exp)
            # Compute exp_qs
            exp_qs = np.reshape(qs_exp, [len(r), self.n_ac_samples])
            exp_qs = np.mean(exp_qs, axis=1)
            # Compute grad_exp_qs
            vs = np.ravel(self.ae._vfn.predict(r.obs_short))
            vs_exp = np.repeat(vs, self.n_ac_samples, axis=0)
            grad_exp_qs = self.policy.nabla_logp_f(obs_exp, acs_exp, qs_exp-vs_exp)
            grad_exp_qs /= self.n_ac_samples  # sum over problem horizon but average over actions
            grad_data = GradDataDR(qs, exp_qs, grad_exp_qs, 
                                    decay=self.delta*self.theta, stop_cv_step=self.stop_cv_step)

        elif self.cv_type == 'dr':            
            qs = self.compute_q(r.obs_short, r.acs, r.sts_short)
            # Sample extra actions for approximating the required expectations.
            # (repeat the same obs for many times consecutively)
            obs_exp = np.repeat(r.obs_short, self.n_ac_samples, axis=0)
            sts_exp = np.repeat(r.sts_short, self.n_ac_samples, axis=0)
            # sample the same randomness for all steps
            rand = np.random.normal(size=[self.n_ac_samples, self.ac_dim])
            rand = np.tile(rand, [len(r), 1])
            acs_exp = self.policy.pi_given_r(obs_exp, rand)
            qs_exp = self.compute_q(obs_exp, acs_exp, sts_exp)
            # Compute exp_qs
            exp_qs = np.reshape(qs_exp, [len(r), self.n_ac_samples])
            exp_qs = np.mean(exp_qs, axis=1)
            # Compute grad_exp_qs
            vs = np.ravel(self.ae._vfn.predict(r.obs_short))
            vs_exp = np.repeat(vs, self.n_ac_samples, axis=0)
            grad_exp_qs = self.policy.nabla_logp_f(obs_exp, acs_exp, qs_exp-vs_exp)
            grad_exp_qs /= self.n_ac_samples  # sum over problem horizon but average over actions
          
            # DR parts
            dr_grad_qs = self.approx_grad_q_given_ro(r)
            exp_dr_grad_qs = self.approx_grad_v(r)

            grad_data = GradDataDR(qs, exp_qs, grad_exp_qs, 
                                    dr_grad_q=dr_grad_qs, 
                                    dr_exp_grad_q=exp_dr_grad_qs,
                                    decay=self.delta*self.theta, 
                                    stop_cv_step=self.stop_cv_step)

        else:
            raise ValueError('Unknown cv_type.')

        return grad_data


    def approx_grad_q_given_ro(self, ro):
        this_not_done = np.ones([ro.acs.shape[0]])
        this_obs = ro.obs_short
        this_acts = ro.acs

        this_not_done = np.tile(this_not_done, [self.num_traj_for_grad_q])
        this_obs = np.tile(this_obs, [self.num_traj_for_grad_q, 1])
        this_acts = np.tile(this_acts, [self.num_traj_for_grad_q, 1])

        grad_q = 0
        for i in range(self.traj_len):
            next_obs, next_acts, next_not_done, grad = tf.get_default_session().run(
                    [self.next_obs, self.next_acts, self.next_not_done, self.grad_log_pi], 
                    feed_dict={
                        self.this_obs: this_obs,
                        self.this_acts: this_acts,
                        self.this_not_done: this_not_done,
                    }
                )
            this_obs = next_obs
            this_acts = next_acts
            this_not_done = next_not_done

            # omit the first grad
            if i > 0:
                grad_q += grad * self.dr_decay ** i
        return grad_q

    
    def approx_grad_q_given_obs_acts(self, this_obs, this_acts):
        this_not_done = np.tile(np.ones([this_acts.shape[0]]), [self.num_traj_for_grad_q])
        this_obs = np.tile(this_obs, [self.num_traj_for_grad_q, 1])
        this_acts = np.tile(this_acts, [self.num_traj_for_grad_q, 1])

        grad_q = 0
        for i in range(self.traj_len):
            next_obs, next_acts, next_not_done, grad = tf.get_default_session().run(
                    [self.next_obs, self.next_acts, self.next_not_done, self.grad_log_pi], 
                    feed_dict={
                        self.this_obs: this_obs,
                        self.this_acts: this_acts,
                        self.this_not_done: this_not_done,
                    }
                )
            this_obs = next_obs
            this_acts = next_acts
            this_not_done = next_not_done

            # omit the first grad
            if i > 0:
                grad_q += grad * self.dr_decay ** i
        return grad_q

    def approx_grad_v(self, ro):
        obs = ro.obs_short
        this_not_done = np.ones([ro.acs.shape[0]])

        this_obs = obs
        grad_v = 0
        for i in range(self.num_grad_q_for_grad_v):
            this_acts = self.policy.pi(this_obs)
            assert this_not_done.shape[0] == this_obs.shape[0] == this_acts.shape[0]
            assert len(this_not_done.shape) == 1
            grad_v += self.approx_grad_q_given_obs_acts(this_obs, this_acts)
        return grad_v / self.num_grad_q_for_grad_v

    def build_approx_grad_q_network(self):
        self.rew_const = 1
        self.dr_decay = self.gamma2

        self.this_obs = tf.placeholder(dtype=tf.float32, shape=[None, self.ae._ob_dim])
        self.this_acts = tf.placeholder(dtype=tf.float32, shape=[None, self.ae._ac_dim])
        self.this_not_done = tf.placeholder(dtype=tf.float32, shape=[None])

        cur_vfn = self.ae._vfn._rebuild_func_apprx_with_raw(self.this_obs)
        cur_policy_mean = self.policy._rebuild_cls_func_apprx_with_raw(self.this_obs, add=None)
        self.cur_policy_mean = cur_policy_mean
        self.cur_vfn = cur_vfn
        
        ts_logstd = tf.maximum(tf.to_float(np.log(self.policy._min_std)), self.policy._ts_logstd)
        ts_std = tf.exp(ts_logstd)

        # calculate the next obs and vfns
        ''' Next Obs '''
        next_obs_gq = self.sim_env.predict_model._rebuild_func_apprx_with_raw(
            tf.concat([self.this_obs, self.this_acts], axis=1)
        )
        next_obs_gq = next_obs_gq + self.this_obs
        self.next_obs_gq = next_obs_gq
        ''' Next Done '''
        # calculate the next done, done condition only for cartpole
        is_inf = tf.reduce_any(tf.math.is_inf(next_obs_gq), axis=1)
        is_exceed = next_obs_gq[:, 1] > .2

        assert len(is_inf.get_shape()) == len(is_exceed.get_shape())
        batch_is_done = tf.logical_or(is_inf, is_exceed)
        batch_is_not_done = tf.logical_not(batch_is_done)
        next_not_done_ = tf.cast(batch_is_not_done, tf.float32)
        next_not_done = next_not_done_ * self.this_not_done

        next_vfn = tf.squeeze(self.ae._vfn._rebuild_func_apprx_with_raw(next_obs_gq)) * next_not_done

        # HACK, we simply use rew_const here, it's correct for CartPole but maybe not for others
        cur_adv = self.rew_const + self.delta * next_vfn - tf.squeeze(cur_vfn)
        ts_logp = self.policy._build_logp(
                        self.policy.y_dim, self.this_acts, cur_policy_mean, ts_logstd)
        
        # mask first
        ts_loss = tf.reduce_sum(tf.stop_gradient(cur_adv * self.this_not_done) * ts_logp)

        # calculate the next action
        next_policy_mean = self.policy._rebuild_cls_func_apprx_with_raw(next_obs_gq, add=None)
        rand = tf.random_normal(tf.shape(next_policy_mean), seed=self.policy.seed + 100)
        noise = ts_std * rand
        next_acts = noise + next_policy_mean
        
        
        # print(cur_accum_grad.get_shape)
        total_grad = U.tf_flatten(tf.gradients(ts_loss, self.policy.ts_vars))
        assert total_grad.get_shape()[0] == 194

        self.grad_log_pi = total_grad / self.num_traj_for_grad_q
        self.next_not_done = next_not_done
        self.next_obs = next_obs_gq
        self.next_acts = next_acts
        
    @online_compatible
    def compute_v(self, obs, dones=None):
        # V that considers padding
        vfns = np.ravel(self.ae._vfn.predict(obs))
        if dones is not None:
            vfns[dones] = self.ae._pe.default_v
        return vfns

    @online_compatible
    def compute_q(self, obs, acs, sts):
        # compute q values given obs, and acs.
        # obs do not include the final obs.
        assert sts is not None
        assert len(sts) == len(obs)
        assert len(sts) == len(acs)

        if hasattr(self.sim_env, '_predict'):
            # XXX Clipping.
            acs = np.clip(acs, *self.sim_env._action_clip)
            next_obs = self.sim_env._predict(np.hstack([obs, acs]))  # XXX use ob instead of st
            rws = np.ones(len(obs))
            next_dones = self.sim_env._batch_is_done(next_obs)
        else:
            raise NotImplementedError
            
        vfns = self.compute_v(next_obs, next_dones)
        qfns = rws + self.delta * vfns
        return qfns

    def compute_update_args(self, ro, weights_type, tar=None):
        if weights_type == 'T-t':
            def weight(l): return np.arange(l, 0.0, -1.0)
        elif weights_type == 'one':
            def weight(l): return np.ones(l)
        assert self.sim_env._action_clip is not None

        def clip(acs): return np.clip(acs, *self.sim_env._action_clip)  # low and high limits
        inputs = np.concatenate([np.hstack([r.obs[:-1], clip(r.acs)]) for r in ro.rollouts])
        if tar == 'dyn':
            targets = np.concatenate([r.obs[1:] for r in ro.rollouts])
        elif tar == 'rw':
            targets = np.expand_dims(ro.rws, axis=1)  # n x 1, unsqueeze
        else:
            raise ValueError('Unknow tar: {}'.format(tar))
        weights = np.concatenate([weight(len(r.acs)) for r in ro.rollouts])
        return inputs, targets, weights

    def update_dyn(self, ro, to_log=False):
        if (hasattr(self.sim_env, '_predict') and self.sim_env._predict is not None):
            inputs, targets, weights = self.compute_update_args(ro, self.dyn_update_weights_type,
                                                                tar='dyn')
            self.sim_env._predict.__self__.update(inputs, targets, weights, to_log=to_log)

    def update_rw(self, ro, to_log=False):
        if (hasattr(self.sim_env, '_rw_fun') and self.sim_env._rw_fun is not None):
            inputs, targets, weights = self.compute_update_args(ro, self.rw_update_weights_type,
                                                                tar='rw')
            self.sim_env._rw_fun.__self__.update(inputs, targets, weights, to_log=to_log)

    def log_sigmas(self, idx=100, n_ros=30, n_acs=30, n_taus=30, n_steps=None,
                               use_vf=False):                              
        # XXX
        # Use state baseline to reduce the variance of the estimates.
        ro = self.gen_ro(max_n_rollouts=n_ros, max_rollout_len=idx+1)
        sts = np.array([r.obs[idx] for r in ro.rollouts if len(r) > idx])
        n_sts = len(sts)

        if n_sts == 0:
            log = {
                'sigma_s_mc': .0,
                'sigma_a_mc': .0,
                'sigma_tau_mc': .0,
                'n_ros_in_total': n_sts * n_acs * n_taus,
                'n_sts': n_sts,
            }
        else:
            acs = self.policy.pi(np.repeat(sts, n_acs, axis=0))
            acs = np.reshape(acs, [n_sts, n_acs, -1])
            Q = np.zeros((n_ros, n_acs, n_taus))
            N_dim = len(self.policy.logp_grad(ro.obs[0], ro.acs[0]))
            N = np.zeros((n_ros, n_acs, N_dim))
            decay = self.ae._pe.gamma * self.delta
            for i, s in enumerate(sts):
                for j, a in enumerate(acs[i]):
                    # This should be the bottleneck!!
                    ro = self.gen_ro(max_n_rollouts=n_taus, max_rollout_len=n_steps,
                                     start_state=s, start_action=a)
                    N[i, j] = self.policy.logp_grad(s, a)
                    for k, r in enumerate(ro.rollouts):
                        q0 = ((decay ** np.arange(len(r))) * r.rws).sum()
                        Q[i, j, k] = q0

            # Fill the rest with zeros.
            if use_vf:
                V = np.zeros((n_ros))
                for i, s in enumerate(sts):
                    V[i] = self.ae._vfn.predict(s[None])[0]

            def compute_sigma_s(Q):
                E_tau_Q = np.mean(Q, axis=2)  # s x a
                if use_vf:
                    E_tau_Q -= np.expand_dims(V, axis=-1)  # s x 1
                E_tau_Q = np.expand_dims(E_tau_Q, axis=-1)  # s x a x 1
                E_a_tau_NQ = np.mean(E_tau_Q * N, axis=1)  # s x N
                E_s_a_tau_NQ = np.mean(E_a_tau_NQ, axis=0)  # N
                E_s_a_tau_NQ = np.expand_dims(E_s_a_tau_NQ, axis=0)  # 1 x N
                Var = np.mean(np.square(E_a_tau_NQ - E_s_a_tau_NQ), axis=0)  # N
                sigma = np.sqrt(np.sum(Var))

                return sigma

            def compute_sigma_a(Q):
                E_tau_Q = np.mean(Q, axis=2)  # s x a
                E_tau_Q = np.expand_dims(E_tau_Q, axis=-1)  # s x a x 1
                N_E_tau_Q = N * E_tau_Q  # s x a x N
                if use_vf:
                    N_E_tau_Q_for_E_a = N * (E_tau_Q - np.reshape(V, V.shape+(1, 1)))
                else:
                    N_E_tau_Q_for_E_a = N_E_tau_Q
                E_a_N_E_tau_Q = np.mean(N_E_tau_Q_for_E_a, axis=1)  # s x N
                E_a_N_E_tau_Q = np.expand_dims(E_a_N_E_tau_Q, axis=1)  # s x 1 x N
                Var = np.mean(np.square(N_E_tau_Q - E_a_N_E_tau_Q), axis=1)  # s x N
                sigma = np.sqrt(np.sum(np.mean(Var, axis=0)))

                return sigma

            def compute_sigma_tau(Q):
                E_tau_Q = np.mean(Q, axis=2)  # s x a
                E_tau_Q = np.expand_dims(E_tau_Q, axis=-1)  # s x a x 1
                Var = np.mean(np.square(Q - E_tau_Q), axis=2)  # s x a
                Var = np.expand_dims(Var, axis=-1)  # s x a x 1
                sigma = np.sqrt(np.sum(np.mean(np.square(N) * Var, axis=(0, 1))))
                return sigma

            log = {
                'sigma_s_mc': compute_sigma_s(Q),
                'sigma_a_mc': compute_sigma_a(Q),
                'sigma_tau_mc': compute_sigma_tau(Q),
                'n_ros_in_total': n_sts * n_acs * n_taus,
                'n_sts': n_sts,
            }

        for k, v in log.items():
            logz.log_tabular(k, v)
