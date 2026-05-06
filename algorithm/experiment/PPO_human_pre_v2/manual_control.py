import os
import sys
import time
import numpy as np
from common import ImageProcess, ensure_data_dir, make_env, reset_env, step_env
from config import config


def pressed_key_action(pygame):
    pygame.event.pump()
    key_pressed = pygame.key.get_pressed()
    if key_pressed[pygame.K_KP4] or key_pressed[pygame.K_LEFT]:
        return 3
    if key_pressed[pygame.K_KP6] or key_pressed[pygame.K_RIGHT]:
        return 2
    if key_pressed[pygame.K_KP5] or key_pressed[pygame.K_SPACE]:
        return 1
    return 0


def main():
    import pygame

    ensure_data_dir()
    env = make_env(render_mode='human')
    image_process = ImageProcess()
    pygame.init()
    pygame.display.set_mode((600, 400))

    if os.path.exists(config['states_file']) and os.path.exists(config['actions_file']):
        states = np.load(config['states_file']).tolist()
        actions = np.load(config['actions_file']).tolist()
    else:
        states, actions = [], []
    print("loaded samples:", len(actions))

    for episode in range(1, 5):
        score = 0.0
        s = reset_env(env)
        done = False
        s_shadow = image_process.StackInit(s)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            time.sleep(0.10)
            a = pressed_key_action(pygame)
            s_, r, done, terminal, info = step_env(env, a)
            states.append(s_shadow.copy())
            actions.append(a)
            s_shadow = image_process.StackNext(s_)
            score += r

        np.save(config['states_file'], np.asarray(states, dtype=np.uint8))
        np.save(config['actions_file'], np.asarray(actions, dtype=np.int64))
        states = np.asarray(states, dtype=np.uint8).tolist()
        print("episode :{}, score : {}, samples : {}".format(episode, score, len(actions)))


if __name__ == '__main__':
    main()
