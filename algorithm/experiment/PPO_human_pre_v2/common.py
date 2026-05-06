import os
import cv2
import numpy as np
from config import config


def ensure_data_dir():
    os.makedirs(config['data_dir'], exist_ok=True)


def make_env(render_mode=None):
    import gym

    names = [config['env_name']] + config.get('fallback_env_names', [])
    last_error = None
    for name in names:
        try:
            if render_mode is None:
                return gym.make(name)
            return gym.make(name, render_mode=render_mode)
        except Exception as exc:
            last_error = exc
    raise last_error


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, terminated, info
    state, reward, done, info = result
    return state, reward, done, done, info


class ImageProcess(object):
    def __init__(self):
        self.frame_stack = config['frame_stack']
        self.frame_size = config['frame_size']
        self.s_shadow = np.zeros([self.frame_stack, self.frame_size, self.frame_size], dtype=np.uint8)

    def ColorMat2Gray(self, state):
        top = config.get('crop_top')
        bottom = config.get('crop_bottom')
        if top is not None and bottom is not None:
            state = state[top:bottom]
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        return state.astype(np.uint8)

    def StackInit(self, state):
        state = self.ColorMat2Gray(state)
        self.s_shadow = np.stack([state] * self.frame_stack, axis=0)
        return self.s_shadow

    def StackNext(self, state):
        s_prime = np.reshape(self.ColorMat2Gray(state), (1, self.frame_size, self.frame_size))
        self.s_shadow = np.append(self.s_shadow[1:], s_prime, axis=0)
        return self.s_shadow
