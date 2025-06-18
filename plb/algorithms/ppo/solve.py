import argparse
import random
import numpy as np
import torch

from ..logger import Logger

import gym
from .env import PlasticineEnv
from gym import register

import os

PATH = os.path.dirname(os.path.abspath(__file__))

ENVS = []
for env_name in ['Move', 'Torus', 'Rope', 'Writer', 'HardWriter', "Pinch", "Rollingpin", "Chopsticks", "Table",
                 'TripleMove', 'TripleWrite', 'Assembly', 'ToothPaste', 'HardRope', 'FingerWriter',
                 'MultiStage_Write', 'MultiStage_Pinch', 'MultiStage_Rope']:
    for id in range(5):
        register(
            id=f'{env_name}-v{id + 1}',
            entry_point=f"plb.algorithms.ppo.env:PlasticineEnv",
            kwargs={'cfg_path': os.path.join(PATH,f"../../envs/env_configs/{env_name.lower()}.yml"), "version": id + 1},
            max_episode_steps=50
        )

def make(env_name, nn=False, sdf_loss=10, density_loss=10, contact_loss=1, soft_contact_loss=False):
    env: PlasticineEnv = gym.make(env_name, nn=nn)
    env.taichi_env.loss.set_weights(sdf=sdf_loss, density=density_loss,
                                    contact=contact_loss, is_soft_contact=soft_contact_loss)
    return env

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='ppo')
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--path", type=str, default='./tmp')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=50)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args=parser.parse_args()

    return args

def main():
    args = get_args()
    if args.num_steps is None:
        args.num_steps = 500000

    logger = Logger(args.path)
    set_random_seed(args.seed)

    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)

    if args.algo == 'ppo':
        from plb.algorithms.ppo.run_ppo import train_ppo
        train_ppo(env, args.path, logger, args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
