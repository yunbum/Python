import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n,env.action_space.n])

learning_rate = .85
dis = .99
num_episodes = 200

rList = []
#direction = []
#dir_src = ['up','down','right','left']

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

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
        print('state =',new_state,'reward =',reward)

        Q[state,action] = (1-learning_rate) * Q[state, action] \
        + learning_rate*(reward + dis*np.max(Q[new_state, :]))
#        print('Q =',Q)
        rAll += reward
        print('rAll =',rAll)
        state = new_state
        print('-----------/')
    rList.append(rAll)

print('final state =',state)
print('===========\n')
print(Q)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
