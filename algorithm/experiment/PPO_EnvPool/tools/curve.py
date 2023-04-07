import matplotlib.pyplot as plt
import numpy as np

test_1 = np.load('scores.npy')
# test_2 = np.load('global_epr.npy')
test_1 = np.mean(test_1, axis=1)
# test_2 = np.mean(test_2, axis=1)
step = 20


def single(curve, name, color):
    plt.plot(range(len(curve)), curve, color, alpha=0.2)
    plt.plot(range(len(curve)), np.convolve(curve, np.ones(step) / step)[:-step + 1], color, label=name)


single(test_1, 'test_1', 'red')
# single(test_2, 'test_2', 'blue')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()
