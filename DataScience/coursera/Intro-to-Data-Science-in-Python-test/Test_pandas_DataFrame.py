import pandas as pd

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.5})

purchase_2 = pd.Series({'Name': 'Kevyn',
                         'Item Purchased': 'Kitty Litter',
                         'Cost': 2.5})

purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.0})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3],
                  index = ['Store 1','Store 1','Store 2'])

print('df.head() =\n',df.head())

print('\n\n df.ix[:1]\n',df.ix[:1])
print('\n\n df.iloc[1:] =\n',df.iloc[1:])
print('\n\n df.iloc[1] =\n',df.iloc[1])
print('\n\n df.loc[Store 2]\n',df.loc['Store 2'])

print('\n\n df[Item Purchased]\n', df['Item Purchased'])
#print('\n\n df.loc[Item Purchased]\n',df.loc['Item Purchased']) ->error

print('\n\n df.loc[Store 1, Cost]\n',df.loc['Store 1', 'Cost'])
print('\n\n df.T\n',df.T)
print('\n\n df.T.loc[Item Purchased]\n',df.T.loc['Item Purchased'])

print('\n\n df.loc[Store 1][Cost] \n',df.loc['Store 1']['Cost'])
print('\n\n df.loc[:,[Name,Cost]] \n', df.loc[:,['Name', 'Cost']])

print('\n\n df.drop(Store 1)\n', df.drop('Store 1'))

copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
print('\n\n df.copy()\n',copy_df)

del copy_df['Name']
print('\n\n del copy_df\n', copy_df)

df['Location'] = None
print('\n\n add column\n', df)

print('\n\n df \n',df)
df['Cost'] = 0.8 * df['Cost']
print('\n\n discount df \n',df)

costs = df['Cost']
print('\n\n df[Cost] \n', df['Cost'])

costs += 2
print('\n\n cost += 2\n', df['Cost'])

print('\n\n df\n',df)
