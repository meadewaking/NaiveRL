import numpy as np

# 初始化数组
actions = np.load('tools/actions.npy')
states = np.load('tools/states.npy')

# 找到所有为0的元素的索引
indices = np.where(actions == 0)[0]
print(actions.shape)
print(states.shape)
print(indices.shape)
# 从这些索引中随机选择100个
selected_indices = np.random.choice(indices, size=10000, replace=False)
mask = np.ones(actions.shape[0], bool)
mask[selected_indices] = False

# 删除选定的元素
actions = actions[mask]
states = states[mask]
print(actions.shape)
print(states.shape)

shuffle_ix = np.random.permutation(np.arange(len(actions)))
actions = actions[shuffle_ix]
states = states[shuffle_ix]
print(actions.shape)
print(states.shape)
# np.save('tools/actions.npy', actions)
# np.save('tools/states.npy', states)
