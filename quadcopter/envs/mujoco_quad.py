import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R
from scipy.stats import special_ortho_group


class QuadcopterEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_file="quadcopter.xml",
        frame_skip=5,
        max_time_steps=1000,
        observation_noise_std=0,
        env_radius=2,
        position_reward_weight=4e-3,
        orientation_reward_weight=0,
        linear_velocity_reward_weight=5e-4,
        angular_velocity_reward_weight=3e-4,
        reach_goal_reward=0.15,
        alive_reward=0.05,
        control_cost_weight=2e-4,
        render_mode=None,
        reset_info=False,
        random_target=True,
    ):
        """
        # observation_space: [-inf, inf] shape=(18,)
            err_xyz: quad_pos - target_pos                     m
            rotation matrix                     (9)            rad
            linear_velocity                     (3)            m/s
            angular_velocity                    (3)            rad/s

        # action space: 4 motors control

        MujocoEnv params: model_path, frame_skip, observation_space: Space, render_mode: Optional[str] = None, width: int = 480, height: int = 480,
                        camera_id: Optional[int] = None, camera_name: Optional[str] = None

        """
        utils.EzPickle.__init__(
            self,
            model_file,
            frame_skip,
            max_time_steps,
            observation_noise_std,
            env_radius,
            position_reward_weight,
            orientation_reward_weight,
            linear_velocity_reward_weight,
            angular_velocity_reward_weight,
            reach_goal_reward,
            alive_reward,
            control_cost_weight,
            render_mode,
            reset_info,
            random_target,
        )

        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float16)
        self.target_pos = np.array([0, 0, 2.0])

        self.max_time_steps = max_time_steps
        self.observation_noise_std = observation_noise_std
        self.reset_info = reset_info
        self.random_target = random_target

        self.env_radius = env_radius

        self.position_reward_weight = position_reward_weight
        self.orientation_reward_weight = orientation_reward_weight
        self.linear_velocity_reward_weight = linear_velocity_reward_weight
        self.angular_velocity_reward_weight = angular_velocity_reward_weight
        self.control_cost_weight = control_cost_weight
        self.reach_goal_reward = reach_goal_reward
        self.alive_reward = alive_reward

        self._time_steps = 0
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "assets", model_file)
        )

        MujocoEnv.__init__(self, model_path, frame_skip, observation_space, render_mode)
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )  # must be after to override value written by MujocoEnv

    def reset_model(self):
        self._time_steps = 0

        ## random intialization ##

        # target position
        self.target_pos = np.array([0, 0, 2.0])

        # quad position
        x = self.np_random.uniform(low=-1, high=1, size=(1,))
        y = self.np_random.uniform(low=-1, high=1, size=(1,))
        z = self.np_random.uniform(low=-1.9, high=1, size=(1,))
        err = np.concatenate((x, y, z))
        init_pos = err + self.target_pos

        # orientation
        init_rot_mat = special_ortho_group.rvs(3)
        quat = R.from_matrix(init_rot_mat).as_quat()

        # velocity
        init_linear_velocity = self.np_random.uniform(low=-1, high=1, size=(3,))
        init_angular_velocity = self.np_random.uniform(low=-1, high=1, size=(3,))

        # set states
        qpos = np.concatenate((init_pos, quat))
        qvel = np.concatenate((init_linear_velocity, init_angular_velocity))
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        info = {
            "initial_pos": init_pos,
            "targest_pos": self.target_pos,
            "linear_vel": init_linear_velocity,
            "angular_vel": init_angular_velocity,
        }
        if self.reset_info:
            print(info)
        return observation

    def step(self, action):
        self._time_steps += 1

        action = np.array(action)
        action = self._adjust_action(action)

        # xyz_position_before = self.get_body_com("core")[:3].copy()
        self.do_simulation(action, self.frame_skip)
        # xyz_position_after = self.get_body_com("core")[:3].copy()

        # xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        # x_velocity, y_velocity, z_velocity = xyz_velocity
        obs = self._get_obs()
        pos = self.target_pos + obs[:3]
        # obs[12] = x_velocity
        # obs[13] = y_velocity
        # obs[14] = z_velocity

        reward = self._get_reward(obs)
        cost = self._control_cost(action)
        total_reward = reward - cost

        pos_norm = np.linalg.norm(pos)

        terminated = bool(
            (not np.isfinite(obs).all()) or pos_norm > self.env_radius or pos[2] < 0.1
        )
        truncated = self._time_steps >= self.max_time_steps

        info = {
            "reward": reward,
            "action_cost": cost,
            "qpos": self.data.qpos,
            "qvel": self.data.qvel,
        }

        obs += self.np_random.normal(
            0, self.observation_noise_std, size=obs.shape
        )  # noise

        return obs, total_reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        position = qpos[0:3]

        quat = np.array(qpos[3:7])
        rot_mat = R.from_quat(quat).as_matrix().flatten()

        error_pos = position - self.target_pos
        obs = np.concatenate((error_pos, rot_mat, qvel), dtype=np.float16).ravel()
        return obs

    def _get_reward(self, obs):
        """
        minimize: position error, linear/angular velocity, motor thrust
        bonus reward to reach goal
        punish pitch, roll when near target?
        """
        err_norm = np.linalg.norm(obs[:3])
        euler = R.from_quat(self.data.qpos[3:]).as_euler("xyz")
        euler_norm = np.linalg.norm(euler)
        linear_vel_norm = np.linalg.norm(obs[12:15])
        angular_vel_norm = np.linalg.norm(obs[15:18])

        pos_reward = -self.position_reward_weight * err_norm
        orientation_reward = -self.orientation_reward_weight * euler_norm
        linear_vel_reward = -self.linear_velocity_reward_weight * linear_vel_norm
        angular_vel_reward = -self.angular_velocity_reward_weight * angular_vel_norm

        reach_goal_reward = 0
        if err_norm < 0.01:
            reach_goal_reward = self.reach_goal_reward

        alive_reward = self.alive_reward

        total_reward = (
            pos_reward
            + orientation_reward
            + linear_vel_reward
            + angular_vel_reward
            + reach_goal_reward
            + alive_reward
        )
        return total_reward

    def _control_cost(self, action):
        control_cost = self.control_cost_weight * np.linalg.norm(action)
        return control_cost

    def _adjust_action(self, action):
        mass = self.model.body_mass[1]
        hover_force = mass * 9.81 * 0.25
        action += hover_force / 3.0
        action = np.clip(action, 0.0, 1.0)
        return action


# test = QuadcopterEnv()
# test.reset()
# test.step([0.0, 0.0, 0.0, 1.0])
# test.step([0.0, 0.0, 0.0, 1.0])
# test.step([0.0, 0.0, 1.0, 1.0])
# test.step([1.0, 1.0, 1.0, 1.0])
# test.step([1.0, 1.0, 1.0, 1.0])
# test.step([1.0, 1.0, 1.0, 1.0])
# test.step([1.0, 1.0, 1.0, 1.0])
# test.step([1.0, 1.0, 1.0, 1.0])
