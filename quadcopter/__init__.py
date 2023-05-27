from gymnasium.envs.registration import register

register(
    id="Quadcopter-v0",
    entry_point="quadcopter.envs.mujoco_quad:QuadcopterEnv",
)
