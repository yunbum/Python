import pandas as pd
import numpy as np
import time as dt

# print(pd.Series?)

animals = ['Tiger', 'Bear', 'Moose']
print(pd.Series(animals))

numbers = [1, 2, 3]
print(pd.Series(numbers))

from pandas import Series, DataFrame

obj = Series([1, 3, 5, 7])
print('\n obj =\n', obj)
print('\n obj.values =', obj.values)
print('\n obj.index =', obj.index)

obj = Series([1, 3, 5, 7], index=['a', 'b', 'c', 'd'])
print('\n obj =')
print(obj)

dic = {'first': 10000, 'second': 20000, 'third': 30000, 'forth': 40000}
obj = Series(dic)

print('\n obj')
print(obj)

dic = {'state': ['Seoul', 'Busan', 'Pohang', 'Gimpo'],
       'year': [2012, 2013, 2014, 2015]}
frame = DataFrame(dic)
print('\n frame')
print(frame)
print('\n frame[state]')
print(frame['state'])

print('\n frame.ix[2]')
print(frame.ix[2])

frame['capital'] = frame['state'] == 'Seoul'
print('\n insert col \n', frame)

del frame['year']
print('\n del frame \n', frame)

a = pd.Series(dic)
print('\n a\n', a)
print('\n a.index[0] \n', a.index[1])

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)

print('s[Golf] =', s['Golf'])
print('s[3] =', s[3])


ss = pd.Series(np.random.randint(0, 1000, 10))
print('\n ss.head() \n',ss.head())
print('\n len(ss)', len(ss))
print('sum(ss)',sum(ss))

summary = 0
t_start = dt.clock()
for item in ss:
    summary += item
t_end = dt.clock()
print('ss-summary =',summary)
print('dt =',t_end-t_start)

t_start = dt.clock()
summary = np.sum(ss)
t_end = dt.clock()
print('\n ss-summary =', summary)
print('dt =',t_end-t_start)

print('')
k = pd.Series(np.random.randint(0,100,10))
print('randint k > k[0], k[1], k[2],',k[0], k[1], k[2])
#for label, value in k.iteritems():
#    s[label] = value+2
#print('s.head() = \n', s.head())
print('\nk.iteritems() = ',k.iteritems())
for i in k.iteritems():
    print('i =',i)

print('\nk =\n',k)

print('\nk.head()\n',k.head())
k += 2
print('\nk +=2 k.head()\n',k.head())
