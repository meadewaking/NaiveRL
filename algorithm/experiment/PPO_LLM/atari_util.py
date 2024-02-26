import cv2
import numpy as np


class ImageProcess(object):
    def __init__(self):
        self.s_shadow = np.zeros([4, 3, 224, 224])

    def ColorMatResize(self, state):
        state = cv2.resize(state, (224, 224), interpolation=cv2.INTER_AREA)
        return state

    def StackInit(self, state):
        state = self.ColorMatResize(state)
        state = np.reshape(state, (3, 224, 224))
        self.s_shadow = np.stack([state] * 4, axis=0)
        return self.s_shadow

    def StackNext(self, state):
        state = self.ColorMatResize(state)
        s_prime = np.reshape(state, (1, 3, 224, 224))
        self.s_shadow = np.append(self.s_shadow[1:], s_prime, axis=0)
        return self.s_shadow
