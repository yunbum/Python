
import numpy as np

r = np.arange(36)
r.resize((6,6))

print('----\n')
print(r)

print('----\n')

print(r[:2, :-1])
print('----\n')

print('----\n')
print(r>30)

print('----\n')
print(r[r>10])

print('----\n')
r[r>25] = -1

print('----\n')
r2 = r[:3, :3]
print(r2)

print('----\n')
r2[:] = 0
print(r2)

print('----\n')
print(r)

print('----\n')
test = np.random.randint(0,10, (4,3))
print(test)

print('----\n')
for row in test:
    print(row)

print('----\n')
for i in range(len(test)):
    print(test[i])

print('----\n')
for i, row in enumerate(test):
    print('row',i, 'is', row)


test2 = test**2
for i, j in zip(test, test2):
    print(i, '+', j, '=', i+j)




