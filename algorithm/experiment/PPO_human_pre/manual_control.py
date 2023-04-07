import gym
import numpy as np
import pygame
import sys
import time
import os

from config import config
from atari_util import ImageProcess


def Presss_key_action():
    pygame.event.pump()
    Key_pressed = pygame.key.get_pressed()
    if Key_pressed[pygame.K_KP4]:
        action = 3
        key = 0
    elif Key_pressed[pygame.K_KP6]:
        action = 2
        key = 1
    elif Key_pressed[pygame.K_KP5]:
        action = 1
        key = 2
    else:
        key = 3
        return 0, key
    return action, key


def main():
    env = gym.make(config['env_name'])
    image_process = ImageProcess()
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    states, actions = [], []
    # if os.path.exists('tools/states.npy'):
    #     states = np.load('tools/states.npy').tolist()
    #     actions = np.load('tools/actions.npy').tolist()
    # else:
    #     states, actions = [], []
    print(len(actions))

    for episode in range(1, 5):
        score = 0.0
        s = env.reset()
        done = False
        s_shadow = image_process.StackInit(s)

        while not done:
            for event in pygame.event.get():  # 防窗口超时
                if event.type == pygame.QUIT:
                    sys.exit()
            env.render()
            time.sleep(0.10)
            a, key = Presss_key_action()
            s_, r, done, info = env.step(a)
            states.append(s_shadow)
            actions.append(a)
            s_shadow = image_process.StackNext(s_)
            score += r

        print("episode :{}, score : {}".format(episode, score))
        print(len(states))
        states = np.array(states)
        states = states.astype(np.uint8)
        np.save('states', states)
        np.save('actions', np.array(actions))
        states = states.tolist()


if __name__ == '__main__':
    main()
