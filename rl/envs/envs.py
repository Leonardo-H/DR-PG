from abc import ABC, abstractmethod
import functools
import numpy as np
from numpy import linalg as la
import os
import pdb

import pydart2 as pydart
pydart.init()


class EnvWithModel(ABC):
    def __init__(self, horizon, predict=None, rw_fun=None, seed=None, action_clip=None, state_clip=None):
        # action_clip, state_clip are lists of of upper/lower bounds, if not None
        self._state_raw = None
        self._predict = predict
        self._rw_fun = rw_fun
        self._reward = self._rw_fun if self._rw_fun is not None else self._default_reward
        self._horizon = horizon  # task horizon
        self._np_rand = np.random.RandomState(seed)
        self._i_step = 0  # a counter that records the current number of steps in episode
        self._action_clip = action_clip
        self._state_clip = state_clip

        # XXX max_episode_steps maybe needed.
        class MySpec(object):
            def __init__(self, max_episode_steps):
                self.max_episode_steps = max_episode_steps
        self.spec = MySpec(horizon)

    @abstractmethod
    def _default_advance(self, a):
        pass

    @abstractmethod
    def _default_reward(self, prev_state, a):
        pass

    @abstractmethod
    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        # if acs is None:
        # return (A, b, c) such that r(s,a) = 0.5*a'*A*a + b'*a + c
        # A: batch * dim_y * dim_y, or batch * dim_y
        # b: batch * dim_y, or batch * dim_y
        # c: batch
        #
        # else:
        # return r(s,a)

        pass

    @property
    @abstractmethod
    def _is_done(self):
        pass

    @property
    def _obs(self):
        # Generate observation from state.
        # Default to simply be states.
        return self._state

    @abstractmethod
    def _reset(self):
        # Reset _state
        pass

    def reset(self):
        self._reset()
        self._i_step = 0
        return self._obs

    @property
    def _state(self):
        # Ensure the state is constrained no matter how _reset is implemented.
        assert self._state_raw is not None, "Need to reset first"
        return self._state_raw.copy() if self._state_clip is None else np.clip(self._state_raw, *self._state_clip)

    @_state.setter
    def _state(self, val):
        self._state_raw = val

    @property
    def state(self):
        """For compatibility, can only be read."""
        return self._state

    @abstractmethod
    def _update_simulator_state(self):
        '''Update the state maintained by the simulator, e.g. DartWorld for Dart.'''
        pass

    def _advance(self, a):
        if self._predict:
            return self._predict(np.hstack([self._state, a]).reshape([-1,5])).reshape([-1])
        else:
            return self._default_advance(a)

    def step(self, a):
        # Advance.
        prev_state = self._state  # maybe used by reward computation
        # Action clipping is part of the dynamics.
        a = a if self._action_clip is None else np.clip(a, *self._action_clip)
        if not self.action_space.contains(a):
            message = 'Invalid action in step {}: {}'.format(self._i_step, a)
            raise ValueError(message)

        self._state = self._advance(a)
        self._i_step += 1
        self._update_simulator_state()

        # Reward.
        # XXX Should be the reward associated with the state before step.
        reward = 1.0

        # Is done.
        done = self._is_done or self._i_step >= self._horizon

        return self._obs, reward, done, {}

    def render(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass


class DartEnvWithModel(EnvWithModel):
    '''XXX state is the state of the MDP, which is [q, dq], the same order as pydart skeleton.'''

    def __init__(self, env, model_path, predict=None, rw_fun=None, model_inacc=None, seed=None):

        self._max_episode_steps = env._max_episode_steps
        env = env.env  # env is TimeLimit object..
        action_clip = [env.action_space.low[0], env.action_space.high[0]]  # the same for all dims
        state_clip = None  # XX no state clip!
        self._ob_sp = env.observation_space
        self._ac_sp = env.action_space
        self._action_scale = env.action_scale
        self.dt = env.dt
        self._frame_skip = env.frame_skip

        # Dart objects.
        self._dart_world, self._robot = self._create_dart_objs(model_path, env)

        # Other stuff.
        super().__init__(env.spec.max_episode_steps, predict, rw_fun, seed, action_clip, state_clip)

        # Make the model inaccurate.
        self._perturb_physcial_params(model_inacc)

    class _require_robot_x_update_to_date(object):
        def __call__(self, func):
            @functools.wraps(func)
            def wrapper(env, *args, **kwargs):
                assert np.allclose(env._robot.x, env._state)
                return func(env, *args, **kwargs)
            return wrapper

    @property
    def observation_space(self):
        return self._ob_sp

    @property
    def action_space(self):
        return self._ac_sp

    @staticmethod
    def _create_dart_objs(model_path, env):
        full_path = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(full_path):
            raise IOError("File {} does not exist".format(full_path))
        dart_world = pydart.World(env.dart_world.dt, full_path)
        robot = dart_world.skeletons[-1]  # assume the skeleton of interest is always the last one!
        for jt in range(0, len(robot.joints)):
            for dof in range(len(robot.joints[jt].dofs)):
                if robot.joints[jt].has_position_limit(dof):
                    robot.joints[jt].set_position_limit_enforced(True)
        return dart_world, robot

    def _rand_ratio(self, inacc, np_rand):
        """Helper function to be used in _perturb_physcial_params."""
        assert inacc >= 0.0 and inacc < 1.0
        return 1.0 + inacc * (np_rand.choice(2) * 2.0 - 1.0)

    def _perturb_physcial_params(self, inacc):
        if inacc is None:
            return
        # Mass.
        for body in self._robot.bodynodes:
            body.set_mass(body.m * self._rand_ratio(inacc, self._np_rand))
        # Damping coeff for revolute joints.
        for j in self._robot.joints:
            if isinstance(j, pydart.joint.RevoluteJoint):
                j.set_damping_coefficient(
                    0, j.damping_coefficient(0) * self._rand_ratio(inacc, self._np_rand))

    def _update_simulator_state(self):
        self._robot.x = self._state

    def set_state(self, s):
        # Reset the robot to any state.
        self._robot.x = s
        self._state = s
        return self._obs

    def _a2tau(self, a):
        # Convert a into tau format (i.e. input to dartworld).
        return a

    def _add_external_forces(self):
        pass

    @_require_robot_x_update_to_date()
    def _default_advance(self, a):
        # Ensures that dart robot agree with self._state (i.e. update-to-date).
        assert np.allclose(self._robot.x, self._state)
        tau = self._a2tau(a * self._action_scale)
        for _ in range(self._frame_skip):
            self._add_external_forces()
            self._robot.set_forces(tau)
            self._dart_world.step()
        return self._robot.x

    @abstractmethod
    def _sample_initial_state(self):
        pass

    def _reset(self):
        self._dart_world.reset()
        self._state = self._robot.x
        self.set_state(self._sample_initial_state())



class Cartpole(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None, predict_model=None, rew_model=None):
        model_path = 'cartpole.skel'
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)
        self.predict_model = predict_model
        self.rew_model = rew_model

    def get_predict_model(self):
        return self.predict_model
    
    def get_rew_model(self):
        return self.rew_model

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        return 1.0

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = len(obs)
        ac_dim = len(self.action_space.high)
        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            rew = np.ones(n)

        if acs is None:
            return np.zeros((n, ac_dim)), np.zeros((n, ac_dim)), rew
        else:
            return rew

    def _batch_is_done(self, obs):
        sts = obs
        sts_ok = np.abs(sts[:, 1]) <= .2
        finite_ok = np.isfinite(sts).all(axis=1)

        return np.logical_not(np.logical_and.reduce([sts_ok, finite_ok]))


    '''
    Receive a batch of (s,a) pairs,
    return the inference next states
    '''
    def _batch_dynamic(self, obs, acts):
        assert len(obs.shape) == len(acts.shape)

        inputs = np.concatenate([obs, acts], axis=1)
        self.predict_model._predict(inputs)

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        notdone = np.isfinite(self._state).all() and (np.abs(self._state[1]) <= .2)
        done = not notdone
        return done

    def _sample_initial_state(self):
        return self._np_rand.uniform(low=-.01, high=.01, size=self._robot.ndofs * 2)  # *2 for both q and dq

    def _a2tau(self, a):
        tau = np.zeros(self._robot.ndofs)
        tau[0] = a
        return tau


class Hopper(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None):
        model_path = 'hopper_capsule.skel'
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)
        try:
            self._dart_world.set_collision_detector(3)
        except Exception:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self._dart_world.set_collision_detector(2)

    def _a2tau(self, a):
        tau = np.zeros(self._robot.ndofs)
        tau[3:] = a
        return tau

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        idx = -2
        # if (self._robot.q[idx] < self._robot.q_lower[idx] + 0.05 or
        #         self._robot.q[idx] > self._robot.q_upper[idx] - 0.05):
        q = prev_state[:self._robot.ndofs]
        if (q[idx] < self._robot.q_lower[idx] + 0.05 or q[idx] > self._robot.q_upper[idx] - 0.05):
            joint_limit_pen = 1.5
        else:
            joint_limit_pen = 0.0
        rew = 1.0  # alive bonus
        # rew += (self._state[0] - prev_state[0]) / (self._dart_world.dt * self._frame_skip)
        rew += prev_state[self._robot.ndofs]
        rew -= 0.001 * np.square(a).sum()
        rew -= 0.5 * joint_limit_pen

        return rew

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = len(obs)
        ac_dim = len(self.action_space.high)
        idx = -2
        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            lower = self._robot.q_lower[idx] + 0.05
            upper = self._robot.q_upper[idx] - 0.05
            assert sts.shape[1] == self._robot.ndofs * 2
            q, dq = sts[:, :self._robot.ndofs], sts[:, self._robot.ndofs:]  # state is [q, dq]
            pen = np.array(np.logical_or(q[:, idx] < lower, q[:, idx] > upper), dtype=float) * 1.5
            rew = 1.0
            rew += dq[:, 0]
            rew -= 0.5 * pen

        if acs is None:
            return -0.002 * np.ones((n, ac_dim)), np.zeros((n, ac_dim)), rew
        else:
            acs = np.clip(acs, *self._action_clip)
            return rew - 0.001 * np.sum(np.square(acs), axis=1)

    def _batch_is_done(self, obs):
        sts = obs[:, :-1]
        # Assume the hieghts is the last in obs.
        heights = obs[:, -1]
        angles = sts[:, 2]
        sts_ok = np.all(np.abs(sts[:, 2:]) < 100.0, axis=1)
        heights_ok = np.logical_and(heights > .7, heights < 1.8)
        angles_ok = abs(angles) < .4
        finite_ok = np.isfinite(sts).all(axis=1)
        horizon_ok = np.array([self._i_step < self._horizon] * len(obs), dtype=bool)

        return np.logical_not(np.logical_and.reduce([heights_ok, sts_ok, angles_ok,
                                                     finite_ok, horizon_ok]))

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        height = self._robot.bodynodes[2].com()[1]
        angle = self._state[2]

        return not (np.isfinite(self._state).all() and (np.abs(self._state[2:]) < 100.0).all() and
                    height > .7 and height < 1.8 and abs(angle) < .4)

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _obs(self):
        # obs = np.concatenate([self._robot.q[1:], np.clip(self._robot.dq, -10.0, 10.0)])
        # obs[0] = self._robot.bodynodes[2].com()[1]
        obs = np.concatenate([self._robot.q, self._robot.dq])
        obs = np.append(obs, self._robot.bodynodes[2].com()[1])

        return obs

    def _sample_initial_state(self):
        # Assumes that dart_world reset has been called.
        return self._robot.x + self._np_rand.uniform(low=-.005, high=.005, size=2 * self._robot.ndofs)


class Snake(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None):
        model_path = 'snake_7link.skel'
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)

        self._dart_world.set_collision_detector(3)
        for body in self._robot.bodynodes:
            body.set_friction_coeff(0.0)

    def _add_external_forces(self):
        for body in self._robot.bodynodes:
            v = body.com_spatial_velocity()
            d = body.to_world([0.0, 0.0, 1.0]) - body.to_world([0.0, 0.0, 0.0])
            v_positive = v[3:] + np.cross(v[:3], d) * 0.05
            v_negative = v[3:] - np.cross(v[:3], d) * 0.05
            if np.dot(v_positive, d) > 0.0:
                fluid_force = -50.0 * np.dot(v_positive, d) * d
            elif np.dot(v_negative, d) < 0.0:
                fluid_force = -50.0 * np.dot(v_negative, d) * d
            else:
                fluid_force = [0.0, 0.0, 0.0]
            body.add_ext_force(fluid_force)

    def _a2tau(self, a):
        tau = np.zeros(self._robot.ndofs)
        tau[3:] = a
        return tau

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        q = prev_state[:self._robot.ndofs]
        qd = prev_state[self._robot.ndofs:]

        rew = 0.1  # alive bonus
        rew += qd[0]  # (self._state[0] - prev_state[0]) / (self._dart_world.dt * self._frame_skip)
        rew -= 0.001 * np.square(a).sum()
        rew -= abs(prev_state[2]) * 0.1  # abs(self._state[2]) * 0.1
        return rew

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = obs.shape[0]
        ac_dim = len(self.action_space.high)

        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            assert sts.shape[1] == self._robot.ndofs * 2
            q, qd = sts[:, :self._robot.ndofs], sts[:, self._robot.ndofs:]  # state is [q, dq]
            rew = 0.1  # alive bonus
            rew += qd[:, 0]  # (self._state[0] - prev_state[0]) / (self._dart_world.dt * self._frame_skip)
            rew -= abs(sts[:, 2]) * 0.1  # abs(self._state[2]) * 0.1

        if acs is None:
            return -np.ones((n, ac_dim))*0.002, np.zeros((n, ac_dim)), rew
        else:
            rew -= 0.001 * np.square(a).sum(axis=1)
            return rew

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        return not (np.isfinite(self._state).all() and
                    (np.abs(self._state[2:]) < 100.0).all() and
                    abs(self._state[2]) < 1.5)

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _obs(self):
        return self._state[1:]

    def _sample_initial_state(self):
        return self._robot.x + self._np_rand.uniform(low=-.005, high=.005, size=2 * self._robot.ndofs)


class Dog(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None):
        model_path = 'dog.skel'
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)

    def _a2tau(self, a):
        tau = np.zeros(self._robot.ndofs)
        tau[6:] = a
        return tau

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        robot_x = self._robot.x
        self._robot.x = prev_state
        prev_dist = self._robot.bodynodes[0].com()[0]
        self._robot.x = robot_x  # self._state
        curr_dist, height, _ = self._robot.bodynodes[0].com()

        alive_bonus = 1.0
        vel_rew = 0.6 * (curr_dist - prev_dist) / self.dt
        act_pen = 1e-3 * np.square(a).sum()
        rew = alive_bonus + vel_rew - act_pen

        return rew

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = obs.shape[0]
        ac_dim = len(self.action_space.high)

        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            rew = np.zeros(n)
            for i, (prev_state, a) in enumerate(zip(sts, acs)):
                # it cannot be batched....
                robot_x = self._robot.x
                state = self._state

                self._robot.x = prev_state
                self._state = prev_state
                prev_dist = self._robot.bodynodes[0].com()[0]
                self._robot.x = self._advance(a)  # ignore the nonlinearity of action here...
                curr_dist, height, _ = self._robot.bodynodes[0].com()
                alive_bonus = 1.0
                vel_rew = 0.6 * (curr_dist - prev_dist) / self.dt
                rew = alive_bonus + vel_rew

                # set it back
                self._robot.x = robot_x
                self._state = state

        if acs is None:
            return -np.ones((n, ac_dim))*2e-3, np.zeros((n, ac_dim)), rew
        else:
            act_pen = 1e-3 * np.square(acs).sum(axis=1)
            rew -= act_pen
            return rew

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        _, height, side_deviation = self._robot.bodynodes[0].com()
        return not (np.isfinite(self._state).all() and (np.abs(self._state[2:]) < 100.0).all() and
                    height > .7 and height < 1.8 and abs(side_deviation) < .4)

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _obs(self):
        return np.concatenate([self._robot.q[1:], np.clip(self._robot.dq, -10.0, 10.0)])

    def _sample_initial_state(self):
        return self._robot.x + self._np_rand.uniform(low=-0.005, high=0.005, size=len(self._robot.x))


class Walker3d(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None):
        model_path = 'walker3d_waist.skel'
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)

        try:
            self._dart_world.set_collision_detector(3)
        except Exception:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self._dart_world.set_collision_detector(2)

        self._robot.set_self_collision_check(True)
        ground = self._dart_world.skeletons[0]
        for i in range(1, len(ground.bodynodes)):
            ground.bodynodes[i].set_friction_coeff(0)

    def _a2tau(self, a):
        tau = np.zeros(self._robot.ndofs)
        tau[6:] = a
        return tau

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        robot_x = self._robot.x
        self._robot.x = prev_state
        prev_dist = self._robot.bodynodes[0].com()[0]
        self._robot.x = robot_x  # self._state  # set it back!
        curr_dist, _, side_deviation = self._robot.bodynodes[0].com()

        contacts = self._dart_world.collision_result.contacts
        total_force_mag = 0.0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_pen = 0.0
        for j in [-3, -9]:
            if self._robot.q_lower[j] - self._robot.q[j] > -0.05:
                joint_limit_pen += 1.5
            if self._robot.q_upper[j] - self._robot.q[j] < 0.05:
                joint_limit_pen += 1.5

        alive_bonus = 1.0
        vel_rew = (curr_dist - prev_dist) / self.dt
        action_pen = 1e-3 * np.square(a).sum()
        joint_pen = 2e-1 * joint_limit_pen
        deviation_pen = 1e-3 * abs(side_deviation)
        rew = vel_rew + alive_bonus - action_pen - joint_pen - deviation_pen

        if self._is_done:
            rew = 0.0

        return rew

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = obs.shape[0]
        ac_dim = len(self.action_space.high)

        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            rew = np.zeros(n)
            for i, (prev_state, a) in enumerate(zip(sts, acs)):
                # it cannot be batched....
                robot_x = self._robot.x
                state = self._state

                self._robot.x = prev_state
                self._state = prev_state
                prev_dist = self._robot.bodynodes[0].com()[0]
                self._robot.x = self._advance(a)
                curr_dist, _, side_deviation = self._robot.bodynodes[0].com()

                contacts = self._dart_world.collision_result.contacts
                total_force_mag = 0.0
                for contact in contacts:
                    total_force_mag += np.square(contact.force).sum()

                joint_limit_pen = 0.0
                for j in [-3, -9]:
                    if self._robot.q_lower[j] - self._robot.q[j] > -0.05:
                        joint_limit_pen += 1.5
                    if self._robot.q_upper[j] - self._robot.q[j] < 0.05:
                        joint_limit_pen += 1.5

                alive_bonus = 1.0
                vel_rew = (curr_dist - prev_dist) / self.dt
                # action_pen = 1e-3 * np.square(a).sum()
                joint_pen = 2e-1 * joint_limit_pen
                deviation_pen = 1e-3 * abs(side_deviation)
                rew[i] = vel_rew + alive_bonus - joint_pen - deviation_pen

                if self._is_done:
                    rew[i] = 0.0

                # set it back
                self._robot.x = robot_x
                self._state = state

        if acs is None:
            return -np.ones((n, ac_dim))*2e-3, np.zeros((n, ac_dim)), rew
        else:
            act_pen = 1e-3 * np.square(acs).sum(axis=1)
            rew -= act_pen
            return rew

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        def get_ang(v):
            d = self._robot.bodynodes[0].to_world(v) - self._robot.bodynodes[0].to_world(np.zeros(3))
            d /= la.norm(d)
            return np.arccos(np.dot(v, d))

        u = np.array([0.0, 1.0, 0.0])
        f = np.array([1.0, 0.0, 0.0])
        ang_uwd, ang_fwd = get_ang(u), get_ang(f)

        _, height, side_deviation = self._robot.bodynodes[0].com()
        done = not (np.isfinite(self._state).all() and (np.abs(self._state[2:]) < 100).all() and
                    (height > 1.05) and (height < 2.0) and
                    (abs(ang_uwd) < 0.84) and (abs(ang_fwd) < 0.84))
        return done

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _obs(self):
        return np.concatenate([self._robot.q[1:], np.clip(self._robot.dq, -10.0, 10.0)])

    def _sample_initial_state(self):
        return self._robot.x + self._np_rand.uniform(low=-0.005, high=0.005, size=len(self._robot.x))


class Walker2d(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None):
        model_path = 'walker2d.skel'
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)

        try:
            self._dart_world.set_collision_detector(3)
        except Exception:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self._dart_world.set_collision_detector(2)

    def _a2tau(self, a):
        tau = np.zeros(self._robot.ndofs)
        tau[3:] = a
        return tau

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        robot_x = self._robot.x
        self._robot.x = prev_state
        prev_dist = self._robot.bodynodes[0].com()[0]
        self._robot.x = robot_x  # self._state  # set it back!
        curr_dist, _, side_deviation = self._robot.bodynodes[0].com()

        joint_limit_pen = 0.0
        for j in [-3, -9]:
            if self._robot.q_lower[j] - self._robot.q[j] > -0.05:
                joint_limit_pen += 1.5
            if self._robot.q_upper[j] - self._robot.q[j] < 0.05:
                joint_limit_pen += 1.5

        alive_bonus = 1.0
        vel_rew = (curr_dist - prev_dist) / self.dt
        action_pen = 1e-3 * np.square(a).sum()
        joint_pen = 2e-1 * joint_limit_pen
        deviation_pen = 1e-3 * abs(side_deviation)
        rew = vel_rew + alive_bonus - action_pen - joint_pen - deviation_pen

        if self._is_done:
            rew = 0.0

        return rew

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = obs.shape[0]
        ac_dim = len(self.action_space.high)

        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            rew = np.zeros(n)
            for i, (prev_state, a) in enumerate(zip(sts, acs)):
                # it cannot be batched....
                robot_x = self._robot.x
                state = self._state

                self._robot.x = prev_state
                self._state = prev_state
                prev_dist = self._robot.bodynodes[0].com()[0]
                self._robot.x = self._advance(a)
                curr_dist, _, side_deviation = self._robot.bodynodes[0].com()

                contacts = self._dart_world.collision_result.contacts
                total_force_mag = 0.0
                for contact in contacts:
                    total_force_mag += np.square(contact.force).sum()

                joint_limit_pen = 0.0
                for j in [-3, -9]:
                    if self._robot.q_lower[j] - self._robot.q[j] > -0.05:
                        joint_limit_pen += 1.5
                    if self._robot.q_upper[j] - self._robot.q[j] < 0.05:
                        joint_limit_pen += 1.5

                alive_bonus = 1.0
                vel_rew = (curr_dist - prev_dist) / self.dt
                # action_pen = 1e-3 * np.square(a).sum()
                joint_pen = 2e-1 * joint_limit_pen
                deviation_pen = 1e-3 * abs(side_deviation)
                rew[i] = vel_rew + alive_bonus - joint_pen - deviation_pen

                if self._is_done:
                    rew[i] = 0.0

                # set it back
                self._robot.x = robot_x
                self._state = state

        if acs is None:
            return -np.ones((n, ac_dim))*2e-3, np.zeros((n, ac_dim)), rew
        else:
            act_pen = 1e-3 * np.square(acs).sum(axis=1)
            rew -= act_pen
            return rew

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        def get_ang(v):
            d = self._robot.bodynodes[0].to_world(v) - self._robot.bodynodes[0].to_world(np.zeros(3))
            d /= la.norm(d)
            return np.arccos(np.dot(v, d))

        u = np.array([0.0, 1.0, 0.0])
        f = np.array([1.0, 0.0, 0.0])
        ang_uwd, ang_fwd = get_ang(u), get_ang(f)

        _, height, side_deviation = self._robot.bodynodes[0].com()
        done = not (np.isfinite(self._state).all() and (np.abs(self._state[2:]) < 100).all() and
                    (height > 1.05) and (height < 2.0) and
                    (abs(ang_uwd) < 0.84) and (abs(ang_fwd) < 0.84))
        return done

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _obs(self):
        return np.concatenate([self._robot.q[1:], np.clip(self._robot.dq, -10.0, 10.0)])

    def _sample_initial_state(self):
        return self._robot.x + self._np_rand.uniform(low=-0.005, high=0.005, size=len(self._robot.x))


class Reacher(DartEnvWithModel):
    def __init__(self, env, predict=None, rw_fun=None, model_inacc=None, seed=None):
        model_path = 'reacher.skel'
        self.target = np.array([0.8, -0.6, 0.6])
        self.fingertip = np.array([0.0, -0.25, 0.0])
        super().__init__(env, model_path, predict, rw_fun, model_inacc, seed)

    def _a2tau(self, a):
        return a

    @DartEnvWithModel._require_robot_x_update_to_date()
    def _default_reward(self, prev_state, a):
        a = np.clip(a, *self._action_clip)                    
        robot_x = self._robot.x
        self._robot.x = prev_state
        vec = self._robot.bodynodes[2].to_world(self.fingertip) - self.target
        rew_dist = 2.0 - la.norm(vec)  # to make it positive
        rew_ctrl = -np.square(a).sum() * 0.1  # due to the action scale 10
        rew = rew_dist + rew_ctrl

        self._robot.x = robot_x
        return rew

    def _batch_reward(self, obs, sts, acs=None, omit_c=False):
        n = obs.shape[0]
        ac_dim = len(self.action_space.high)

        if omit_c and acs is None:
            rew = np.zeros(n)
        else:
            vec = obs[:, -3:]
            rew = 2.0 - la.norm(vec, axis=1)
        if acs is None:
            return -np.ones((n, ac_dim))*0.2, np.zeros((n, ac_dim)), rew
        else:
            acs = np.clip(acs, *self._action_clip)            
            ac_pen = 0.1 * np.square(acs).sum(axis=1)
            rew -= ac_pen
            return rew

    def _batch_is_done(self, obs):
        # Assume the heights is the last in obs.
        finite_ok = np.logical_not(np.isnan(obs).any(axis=1))
        vec = obs[:, -3:]
        dist_ok = la.norm(vec, axis=1) > 0.1
        horizon_ok = np.array([self._i_step < self._horizon] * len(obs), dtype=bool)
        return np.logical_not(np.logical_and.reduce([finite_ok, dist_ok, horizon_ok]))

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _is_done(self):
        vec = self._robot.bodynodes[2].to_world(self.fingertip) - self.target
        dist = la.norm(vec)
        done = not (np.isfinite(self._state).all() and (dist > 0.1))
        return done

    @property
    @DartEnvWithModel._require_robot_x_update_to_date()
    def _obs(self):
        vec = self._robot.bodynodes[2].to_world(self.fingertip) - self.target
        return np.concatenate([np.cos(self._robot.q), np.sin(self._robot.q), self.target,
                               self._robot.dq, vec])

    def _sample_initial_state(self):
        self._dart_world.reset()
        qpos = self._robot.q + self._np_rand.uniform(low=-.01, high=.01, size=self._robot.ndofs)
        qvel = self._robot.dq + self._np_rand.uniform(low=-.01, high=.01, size=self._robot.ndofs)
        self.set_state(np.concatenate([qpos, qvel]))
        # Just for plotting...
        while True:
            self.target = self._np_rand.uniform(low=-1, high=1, size=3)
            if np.linalg.norm(self.target) < 1.5:
                break
        self._dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]
        return self.state
