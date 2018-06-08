import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id = 'FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n,env.action_space.n])

#learning_rate = .85
dis = .99
num_episodes = 2000

rList = []
#direction = []
#dir_src = ['up','down','right','left']

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    path_arr = []

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
#        print('state =',state)
#        print('action =',action)
#        print('Q[state, :] =',Q[state, :])
#        idx = np.where(Q[state, :] == 1)[0]
#        print('idx =',idx)
#        print('idx[0] =',idx[0])
#        print('dir',dir_src[idx[0])
#        print('env.action_space.n =', env.action_space.n)

        new_state, reward, done,_ = env.step(action)
#        print('state =',new_state,'reward =',reward)

        Q[state,action] = reward + dis*np.max(Q[new_state, :])
#        print('Q =',Q)
        rAll += reward
#        print('rAll =',rAll)
        state = new_state
#        print('-----------/')
    rList.append(rAll)
    print('------')
    for i in range(16):
        path_arr.append(np.argmax(Q[i]))
    b = np.array(path_arr)
    b = b.reshape(4, 4)
    print(b)

#print('rList= \n', rList)
print('final state =',state)
print('===========\n')

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print('Q = \n',Q)
plt.bar(range(len(rList)), rList, color="blue")
#plt.show()
np.set_printoptions(precision=1)
Q_copy = Q.copy()
Q_copy = np.round(Q_copy,1)
print('Q_copy = \n',Q_copy)

print('argmax = ',np.argmax(Q[0]))
for i in range(16):
    path_arr.append(np.argmax(Q[i]))

#b = np.reshape(path_arr,4,4)
print(path_arr)
b = np.array(path_arr)
print('b')
print(b)