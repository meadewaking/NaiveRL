import numpy as np

actions = np.load('actions.npy')
print(actions.shape)
o_a = np.load('tools/actions.npy')
print(o_a.shape)
a = np.concatenate((actions, o_a))
print(a.shape)

states = np.load('states.npy')
print(states.shape)
o_s = np.load('tools/states.npy')
print(o_s.shape)
s = np.concatenate((states, o_s))
print(s.shape)

np.save('tools/actions.npy', a)
np.save('tools/states.npy', s)
