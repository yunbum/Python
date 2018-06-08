import pandas as pd

#df = pd.read_csv('olympics.csv')
df = pd.read_csv('olympics2.csv')

print(' pd.read_csv(file_name) \n df.head() =\n', df.head())



df = pd.read_csv('olympics2.csv', index_col=0, skiprows=1 )
print('\n\n pd.read_csv(file_name, index_col=0, skiprows=1) \n df.head() =\n', df.head())



print('\n\n df.columns \n', df.columns)

print('df.columns[1][:2]', df.columns[1][:2])

for col in df.columns:
    if col[:2] == '01':
        df.rename(columns = {col:'Gold'+col[4:]}, inplace = True)
    if col[:2] == '02':
        df.rename(columns = {col:'Silver'+col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns = {col:'Bronze'+col[4:]}, inplace = True)
    if col[:1] == '№':
        df.rename(columns = {col:'#'+col[4:]}, inplace = True)


print('\n\n df.head() /rename, inplace/ \n', df.head())

print('\n\n index[2] =',df.index[2])

print('\n\n df[Gold] > 0 \n ', df['Gold'] > 0)
print('\n\n df[Gold].head() = \n', df['Gold'].head() )

only_gold = df.where(df['Gold'] > 0)
print('\n\n only_gold.head() =\n', only_gold.head())
print('\n\n only_gold[Gold].head() \n', only_gold['Gold'].head())
print('\n\n only_gold[Gold].count() =', only_gold['Gold'].count())
print('\n\n only_gold\n',only_gold)
print('\n\n only_gold.count() =\n', only_gold.count())
print('\n\n df[Gold].count =', df['Gold'].count())
print('\n\n df.count() =\n',df.count())


only_gold_over900 = df.where(df['Gold'] > 900)
print('\n\n only_gold_over900, ', only_gold_over900['Gold'].count())

only_gold = df[df['Gold']>0]
print('\n\n only_gold / df[df[Gold]]\n', only_gold)


only_gold2 = only_gold.dropna()
print('\n\n only_gold2.count() =\n', only_gold2.count())
print('\n\n only_gold2 =\n', only_gold2)

print('\n\n Gold or Silver\n',    df[(df['Gold']>5) | (df['Silver']>5) ])
print('\n\n Gold & Silver\n',df[(df['Gold']>10) & (df['Silver']>100)])

