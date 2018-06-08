import numpy as np

a = np.zeros(16)
a.resize((4,4))

print(a)


test = [[0,0,1,0],
        [0,1,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,1,0,0],
        [0,0,0,0],
        [0,0,1,0],
        [0,0,1,0]
        ]

print('test =',test)
print('\n')
rlist = [0]
b = np.ones(1)
print('b =',b)

for x in test:
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
#b = np.empty(4,4)
print('b =',b)

c = np.empty(0)

c = np.append(c,1)
print('c =',c)

