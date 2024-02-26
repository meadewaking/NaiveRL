import matplotlib.pyplot as plt
import numpy as np

total_len = 10000
# test_1 = np.load('res50_scratch.npy')
# test_2 = np.load('scores.npy')[:total_len]
test_3 = np.load('llarp.npy')
step = 30


def single(curve, name, color):
    plt.plot(curve, color, alpha=0.2)
    # plt.plot(np.convolve(np.pad(curve, step//2, mode='edge'), np.ones(step) / step, mode='valid'), color, label=name)
    plt.plot(
        np.convolve(np.concatenate((curve[step // 2:0:-1], curve, curve[-1:-step // 2 - 1:-1])), np.ones(step) / step,
                    mode='valid'), color, label=name)


# single(test_1, 'res50_scratch', 'red')
# single(test_2, 'mlp_scratch', 'green')
single(test_3, 'llarp', 'brown')
plt.axhline(35, linestyle='--', color='b', label='human')
plt.axhline(100, linestyle='--', color='black', label='expert')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()
