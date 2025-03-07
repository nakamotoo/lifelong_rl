import torch

import argparse
import sys

from lifelong_rl.envs.env_processor import make_env
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.util.visualize_mujoco import record_mujoco_video_from_states

from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.


"""
Visualize replay buffer of agent and store as .mp4
"""


def get_env_states(snapshot_name):
    with open(snapshot_name + '.pt', 'rb') as f:
        snapshot = torch.load(f, map_location='cpu')
        env_states = snapshot['replay_buffer/env_states']
        desired_goals = snapshot['replay_buffer/desired_goals'] if 'replay_buffer/desired_goals' in snapshot else None
    return env_states, desired_goals


parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', '-name', type=str,
                    help='Name of snapshot to visualize (ex. 12-07-hopper/run_1/itr_999')
parser.add_argument('--env', type=str,
                    help='Which environment to visualize for')
parser.add_argument('--output', '-o', type=str, default='visualize',
                    help='Name of file to output mp4 video to')
parser.add_argument('--start', '-s', type=int, default=0,
                    help='Timestep to start visualization from')
parser.add_argument('--end', '-e', type=int, default=100,
                    help='Timestep to end visualization (should be > start)')
parser.add_argument('--time_delay', '-dt', type=float, default=0.03,
                    help='Length of time between frames')
args = parser.parse_args(sys.argv[1:])


ptu.set_gpu_mode(False)

env_states, desired_goals = get_env_states(args.snapshot)
env, _ = make_env(args.env)
print(env)
print(args.output)

dg = None if desired_goals is None else desired_goals[args.start:args.end]

record_mujoco_video_from_states(
    env,
    args.output,
    env_states[args.start:args.end],
    dg,
    time_delay=args.time_delay,
)
