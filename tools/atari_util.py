import cv2
import numpy as np


class ImageProcess(object):
    def __init__(self):
        self.s_shadow = np.zeros([4, 84, 84])

    def ColorMat2Gray(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
        return state

    def StackInit(self, state):
        state = self.ColorMat2Gray(state)
        self.s_shadow = np.stack((state, state, state, state), axis=0)
        return self.s_shadow

    def StackNext(self, state):
        s_prime = np.reshape(self.ColorMat2Gray(state), (1, 84, 84))
        self.s_shadow = np.append(self.s_shadow[1:, :, :], s_prime, axis=0)
        return self.s_shadow
