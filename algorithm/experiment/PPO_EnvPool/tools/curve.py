import matplotlib.pyplot as plt
import numpy as np

test_1 = np.load('scores.npy')
# test_2 = np.load('global_epr.npy')
test_1 = np.mean(test_1, axis=1)
# test_2 = np.mean(test_2, axis=1)
step = 20


def single(curve, name, color):
    plt.plot(curve, color, alpha=0.2)
    plt.plot(
        np.convolve(np.concatenate((curve[step // 2:0:-1], curve, curve[-1:-step // 2 - 1:-1])), np.ones(step) / step,
                    mode='valid'), color, label=name)


single(test_1, 'test_1', 'red')
# single(test_2, 'test_2', 'blue')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()
