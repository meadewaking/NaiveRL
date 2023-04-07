import cv2
import numpy as np


class ImageProcess_pool(object):
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.vector_s_shadow = np.zeros([num_envs, 4, 84, 84])

    def ColorMat2Gray(self, vector_state):
        vector_temp = []
        for state in vector_state:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            # state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
            state = cv2.resize(state[34:194], interpolation=cv2.INTER_AREA)
            vector_temp.append(state)
        vector_state = np.stack(vector_temp, axis=0)
        return vector_state

    def StackInit(self, vector_state):
        vector_state = self.ColorMat2Gray(vector_state)
        self.vector_s_shadow = np.stack((vector_state, vector_state, vector_state, vector_state), axis=0)
        return self.vector_s_shadow

    def StackNext(self, vector_state):
        vector_s_prime = np.reshape(self.ColorMat2Gray(vector_state), (1, self.num_envs, 84, 84))
        self.vector_s_shadow = np.append(self.vector_s_shadow[:, 1:, :, :], vector_s_prime, axis=1)
        return self.vector_s_shadow


class ImageProcess(object):
    def __init__(self):
        self.s_shadow = np.zeros([4, 84, 84])

    def ColorMat2Gray(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
        state = cv2.resize(state[34:194], (84, 84), interpolation=cv2.INTER_AREA)
        return state

    def StackInit(self, state):
        state = self.ColorMat2Gray(state)
        self.s_shadow = np.stack((state, state, state, state), axis=0)
        return self.s_shadow

    def StackNext(self, state):
        s_prime = np.reshape(self.ColorMat2Gray(state), (1, 84, 84))
        self.s_shadow = np.append(self.s_shadow[1:, :, :], s_prime, axis=0)
        return self.s_shadow
