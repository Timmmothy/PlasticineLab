import gymnasium as gym
from .env import PlasticineEnv
from gymnasium import register

ENVS = []
for env_name in ['Move', 'Torus', 'Rope', 'Writer', 'HardWriter', "Pinch", "Rollingpin", "Chopsticks", "Table",
                 'TripleMove', 'TripleWrite', 'Assembly', 'ToothPaste', 'HardRope', 'FingerWriter',
                 'MultiStage_Write', 'MultiStage_Pinch', 'MultiStage_Rope']:
    for id in range(5):
        register(
            id=f'{env_name}-v{id + 1}',
            entry_point=f"plb.envs.env:PlasticineEnv",
            kwargs={'cfg_path': f"./env_configs/{env_name.lower()}.yml", "version": id + 1},
            max_episode_steps=50
        )


def make(env_name, nn=False, sdf_loss=10, density_loss=10, contact_loss=1, soft_contact_loss=False, seed=None):
    env: PlasticineEnv = gym.make(env_name, nn=nn, seed=seed)
    env.taichi_env.loss.set_weights(sdf=sdf_loss, density=density_loss,
                                    contact=contact_loss, is_soft_contact=soft_contact_loss)
    return env
