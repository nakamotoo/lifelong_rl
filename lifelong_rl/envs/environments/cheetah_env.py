import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

"""
 State-Space (name/joint/parameter):
        - rootx     slider      position (m)
        - rootz     slider      position (m)
        - rooty     hinge       angle (rad)
        - bthigh    hinge       angle (rad)
        - bshin     hinge       angle (rad)
        - bfoot     hinge       angle (rad)
        - fthigh    hinge       angle (rad)
        - fshin     hinge       angle (rad)
        - ffoot     hinge       angle (rad)
        - rootx     slider      velocity (m/s)
        - rootz     slider      velocity (m/s)
        - rooty     hinge       angular velocity (rad/s)
        - bthigh    hinge       angular velocity (rad/s)
        - bshin     hinge       angular velocity (rad/s)
        - bfoot     hinge       angular velocity (rad/s)
        - fthigh    hinge       angular velocity (rad/s)
        - fshin     hinge       angular velocity (rad/s)
        - ffoot     hinge       angular velocity (rad/s)
    Actuators (name/actuator/parameter):
        - bthigh    hinge       torque (N m)
        - bshin     hinge       torque (N m)
        - bfoot     hinge       torque (N m)
        - fthigh    hinge       torque (N m)
        - fshin     hinge       torque (N m)
        - ffoot     hinge       torque (N m)
"""

class PartialHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, partial_mode="vel", reward_mode=None):
        self._partial_mode = partial_mode
        self._reward_mode = reward_mode
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        if partial_mode == "vel":
            self.hidden_state_dim = self.sim.data.qvel.shape[0]
        elif partial_mode == "ffoot":
            self.hidden_state_dim = 2

        print("cheetah reward mode:", self._reward_mode)


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        if self._reward_mode == "back":
            reward = reward_ctrl - reward_run
        else:
            reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        if self._partial_mode == "vel":
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                # self.sim.data.qvel.flat,
            ])
        elif self._partial_mode == "ffoot":
            return np.concatenate([
                self.sim.data.qpos.flat[1:-1],
                self.sim.data.qvel.flat[:-1],
            ])

    def _get_hidden_state(self):
        if self._partial_mode == "vel":
            return np.concatenate([
                # self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ])
        elif self._partial_mode == "ffoot":
            return np.array([
                self.sim.data.qpos.flat[-1],
                self.sim.data.qvel.flat[-1],
            ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_mode=None):
        self._reward_mode = reward_mode
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        print("cheetah reward mode:", self._reward_mode)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        if self._reward_mode == "back":
            reward = reward_ctrl - reward_run
        else:
            reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5