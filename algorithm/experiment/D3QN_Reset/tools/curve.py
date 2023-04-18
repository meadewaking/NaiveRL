import matplotlib.pyplot as plt
import numpy as np

test_1 = np.load('scores_10.npy')
test_2 = np.load('scores.npy')
test_3 = np.load('scores_reset.npy')
test_1 = np.mean(test_1, axis=1)
test_2 = np.mean(test_2, axis=1)
test_3 = np.mean(test_3, axis=1)
step = 20


def single(curve, name, color):
    plt.plot(curve, color, alpha=0.2)
    plt.plot(np.convolve(np.pad(curve, step//2, mode='edge'), np.ones(step) / step, mode='valid'), color, label=name)


single(test_1, 'DQN_10', 'red')
single(test_2, 'DQN', 'blue')
single(test_3, 'DQN_Reset', 'green')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()
