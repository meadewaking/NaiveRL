import matplotlib.pyplot as plt
import numpy as np

test_1 = np.load('scores.npy')
test_2 = np.load('pre.npy')
test_3 = np.load('frozen_feature.npy')
step = 300


def single(curve, name, color):
    plt.plot(curve, color, alpha=0.2)
    # plt.plot(np.convolve(np.pad(curve, step//2, mode='edge'), np.ones(step) / step, mode='valid'), color, label=name)
    plt.plot(
        np.convolve(np.concatenate((curve[step // 2:0:-1], curve, curve[-1:-step // 2 - 1:-1])), np.ones(step) / step,
                    mode='valid'), color, label=name)


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
