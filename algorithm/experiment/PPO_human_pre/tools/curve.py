import matplotlib.pyplot as plt
import numpy as np

test_1 = np.load('scores.npy')
test_2 = np.load('pre.npy')
test_3 = np.load('frozen_feature.npy')
step = 300


def single(curve, name, color):
    plt.plot(curve, color, alpha=0.2)
    plt.plot(np.convolve(curve, np.ones(step) / step, mode='same')[:-step], color, label=name)


single(test_1, 'scratch', 'red')
single(test_2, 'human_pre', 'green')
single(test_3, 'frozen_feature', 'brown')
plt.axhline(35, linestyle='--', color='b', label='human')
plt.axhline(100, linestyle='--', color='black', label='expert')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()
