import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
#import random as pr

print('dir(gym) =',dir(gym))

'''
def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)
'''


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n,env.action_space.n])

dis = .99
num_episodes = 200

rList = []
direction = []
dir_src = ['up','down','right','left']

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    print('++++++++++')

    while not done:
        print('/---------')
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        print('state =',state)
        print('action =',action)
        print('Q[state, :] =',Q[state, :])
        idx = np.where(Q[state, :] == 1)[0]
        print('idx =',idx)
#        print('idx[0] =',idx[0])
#        print('dir',dir_src[idx[0])
        print('env.action_space.n =', env.action_space.n)

        new_state, reward, done,_ = env.step(action)
        print('state =',new_state,'reward =',reward)

        Q[state,action] = reward + np.max(Q[new_state, :])
        print('Q =',Q)

        rAll += reward
        print('rAll =',rAll)
        state = new_state
        print('-----------/')
    print('final state =',state)
    print('===========\n')
    print(Q)

    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print('Q = \n',Q)
plt.bar(range(len(rList)), rList, color="blue")
#plt.show()

Q_copy = Q.copy()

print('Q_copy = \n',Q_copy)

test = [1]

for x in Q:
    if 1 in x:
        print(1)
        test.insert(-1,1)
    else:
        print(0)
        test.insert(-1,0)

del test[-1]
test = np.array(test).reshape(4,4)
print('Q = \n',Q)
print('test = \n',test)
#b.resize(4,4)
#print('b =',b)
'''
b = np.ones(1)

for x in Q:
    print('->','x =',x)
    if 1 in x:
        rlist.append(1)
        b = np.append(b,1)
    else:
        rlist.append(0)
        b = np.append(b,0)
    print('rlist =',rlist)
    print('b =',b)

b.resize(4,4)
'''