import os
import gym
import numpy as np
from sac_agent import SAC
from utils import atari_env


#Parametros
start_episode = 0
env_name = "MsPacman-v4"
memory = 10000
eval = False


def main():
    dirs = 'SAC/runs/' + env_name
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    img_size = (4, 84, 84)
    env = atari_env(env_name)
    memory_par = (memory, img_size)
    action_space = np.array([i for i in range(env.action_space.n)], dtype=np.uint8)
    game = (env_name, env)


    agent = SAC(memory_par=memory_par,
                action_space=action_space,
                game=game,
                )

    agent.train(net_path=dirs,
                   start_episodes=start_episode,
                   eval=eval,
                   start_frames=0)


if __name__ == '__main__':
    main()
