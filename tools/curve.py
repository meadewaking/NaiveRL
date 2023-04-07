import matplotlib.pyplot as plt
import numpy as np

test_1 = np.load('test_1.npy')
test_2 = np.load('test_2.npy')
step = 30


def single(curve, name, color):
    plt.plot(range(len(curve)), curve, color, alpha=0.2)
    plt.plot(range(len(curve)), np.convolve(curve, np.ones(step) / step)[:-step + 1], color, label=name)


single(test_1, 'test_1', 'red')
single(test_2, 'test_2', 'blue')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()
