import pandas as pd

#df = pd.read_csv('olympics.csv')
df = pd.read_csv('olympics2.csv')

#print(' pd.read_csv(file_name) \n df.head() =\n', df.head())

df = pd.read_csv('olympics2.csv', index_col=0, skiprows=1 )
#print('\n\n pd.read_csv(file_name, index_col=0, skiprows=1) \n df.head() =\n', df.head())

#print('\n\n df.columns \n', df.columns)
#print('df.columns[1][:2]', df.columns[1][:2])

for col in df.columns:
    if col[:2] == '01':
        df.rename(columns = {col:'Gold'+col[4:]}, inplace = True)
    if col[:2] == '02':
        df.rename(columns = {col:'Silver'+col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns = {col:'Bronze'+col[4:]}, inplace = True)
    if col[:1] == '№':
        df.rename(columns = {col:'#'+col[4:]}, inplace = True)

print(df)
print('---------------------------------------------------------')
print(df.index)

print('---------------------------------------------------------')
print('country column 추가')
df['country'] = df.index
df = df.set_index('Gold')
print('\n df')
print(df)

print('---------------------------------------------------------')
df = df.reset_index()
print('index 리셋, 초기화')
print('df')
print(df.head(3))

print('/////////////////////////////////////////////////////////')
df = pd.read_csv('census.csv')
print('census.csv data table')
print(df.head(3))

print('---------------------------------------------------------')
print('unique() for SUMLEV ')
print(df['SUMLEV'].unique())

print('---------------------------------------------------------')
df = df[df['SUMLEV'] == 50 ]
#print(df[ df['SUMLEV'] == 40])
print('SUMLEV ==50')
print(df.head())


print('---------------------------------------------------------')
columns_to_keep = ['STNAME','CTYNAME',
                   'BIRTHS2010', 'BIRTHS2011','BIRTHS2012',
                   'BIRTHS2013','BIRTHS2014','BIRTHS2015',
                   'POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012',
                   'POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']
df = df[columns_to_keep]
print('## df[STNAME, CTYNAME, BIRTHS2010, BIRTHS2011, POPESTIMATE2010, POPESTIMATE2011]')
print(df)

print('---------------------------------------------------------')
df = df.set_index(['STNAME','CTYNAME'])
print('@@ set_index([STNAME,CTYNAME]')
print(df)

print('---------------------------------------------------------')
print('Michigan, Washtenaw County')
print(df.loc[('Michigan', 'Washtenaw County')])


