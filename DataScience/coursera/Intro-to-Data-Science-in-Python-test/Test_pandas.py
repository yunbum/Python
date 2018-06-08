import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1, 3, 4, np.nan, 6, 8])
print('s =\n', s)

dates = pd.date_range('20180101', periods=10)
print(dates)

df = pd.DataFrame(np.random.randn(10, 4), index=dates, columns=list('ABCD'))
print('\n df =')
print('df.head(3)')
print(df.head(3))
print('---')
print('df.tail(3)')
print(df.tail(3))
print('\n df.dtypes')
print(df.dtypes)
print(dir(df))

print('df.index')
for i in range(5):
    print('\t', df.index[i])

print('df.columns')
print(df.columns)

print('df.values')
for i in range(5):
    print(df.values[i])

print('\n df.describe()')
print(df.describe())

print('')
print('sorted by axis 1')
print(df.sort_index(axis=1, ascending=False))

print('')
print('sorted by B')
print(df.sort_values(by='B'))

print('')
print('selection C')
print(df['C'])
print('')
print('selection C[0:3]')
print(df['C'][0:3])
print('')
print('df[20180101:20180105]')
print(df['20180101':'20180105'])

print('')
print('loc')
print(df.loc[dates[0]])

print('')
print('df.loc[:,[A,B]')
print(df.loc[:, ['A', 'B']])

print('')
print('df.loc[date,[column]]')
print(df.loc['20180104':'20180107', ['A', 'B']])

print('')
print('df.loc[],[column]')
print(df.loc[dates[5], ['A', 'B']])

print('')
print('df.at[date], A')
print(df.at[dates[5], 'A'])

print('')
print('df.mean()')
print(df.mean())
print('tmp_loc.df.mean() test')
tmp_loc = df.loc['20180104':'20180107', ['A']]
print(tmp_loc.mean())
