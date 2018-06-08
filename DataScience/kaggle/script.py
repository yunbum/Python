 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:50:23 2018

@author: brijnanda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Label encoder
from sklearn.preprocessing import LabelEncoder
# Metrics for root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression # Import Linear Regression



#dftest=pd.read_csv('../input/Test.csv')
dftest=pd.read_csv('../data/4study/bigmart-sales-data/Test.csv')

#dftrain=pd.read_csv('../input/Train.csv')
dftrain=pd.read_csv('../data/4study/bigmart-sales-data/Train.csv')

data=pd.concat([dftrain,dftest],ignore_index=True)
col=dftrain.columns
dftrain[col[1]].fillna(value=dftrain[col[1]].mean(),inplace=True) # for Item_Weight
#dftrain[col[1]].fillna(value=mean,inplace=True)
dftrain[col[8]].value_counts()
dftrain['New']=dftrain['Outlet_Size'].map({'Small':1,'Medium':2,'High':3})  #mapping for Categ. var. Outlet_Size is col[8]
dftrain.drop('Outlet_Size',axis=1,inplace=True)
dftrain.rename(columns={'New':'Outlet_Size'},inplace=True)
dftrain['Outlet_Size'].fillna(value=dftrain['Outlet_Size'].mean(),inplace=True) #Outlet_Size has 0 Null/NaN values 
categorical_columns = [x for x in dftrain.dtypes.index if dftrain.dtypes[x]=='object']
print(len(categorical_columns))
categorical_columns=[x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
for x in categorical_columns:
    print("\n frequency of %s"%x)
    print(dftrain[x].value_counts())
dftrain['Item_type_combined']=dftrain['Item_Identifier'].apply(lambda x:x[0:2])
dftrain['Item_type_combined']=dftrain['Item_type_combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
dftrain['Item_Fat_Content']=dftrain['Item_Fat_Content'].map({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat','Regular':'Regular','Low Fat':'Low Fat'})
dftrain['Item_Fat_Content'].value_counts().sum()

# drop columns not required

dftrain.drop(['Item_Type','Outlet_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)
le = LabelEncoder()
dftrain['Item_Fat_Content'] = le.fit_transform(dftrain['Item_Fat_Content'])
dftrain['Item_type_combined'] = le.fit_transform(dftrain['Item_type_combined'])
dftrain['Item_Identifier'] = le.fit_transform(dftrain['Item_Identifier'])
corr=dftrain.corr()
#sns.pairplot(dftrain)  #Features  shows up  now try using a model like regression
#let us try PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=7)  # I have selected 7 components to test as main features

pca.fit(dftrain)
print(pca.components_)
print(pca.explained_variance_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#from plot we see that 6 components account for 100% variance of Data


X_train = dftrain
X_test = dftest
y = dftrain['Item_Outlet_Sales']
# Initialize models
#xgb = XGBRegressor(max_depth=5);
lr = LinearRegression();
# Initialize Ensemble
#model = StackingRegressor(regressors=[svr, mlp, elastic, lasso, ridge, bridge],
#                          meta_regressor=lr);

# Fit the model on our data
lr.fit(X_train, y)

# Predict training set
y_pred = lr.predict(X_train)
print(sqrt(mean_squared_error(np.log(y), np.log(y_pred))))
#  Y_pred = lr.predict(X_test)
col=dftest.columns
dftest[col[1]].fillna(value=dftest[col[1]].mean(),inplace=True) # for Item_Weight
dftest['Outlet_Size']=dftest['Outlet_Size'].map({'Small':1,'Medium':2,'High':3})
dftest['Outlet_Size']=dftest['Outlet_Size'].fillna(value=2.0,inplace=True)
dftest['Item_type_combined']=dftest['Item_Identifier'].apply(lambda x:x[0:2])
dftest['Item_type_combined']=dftest['Item_type_combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
dftest['Item_Fat_Content']=dftest['Item_Fat_Content'].map({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat','Regular':'Regular','Low Fat':'Low Fat'})
dftest.drop(['Item_Type','Outlet_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)
le = LabelEncoder()
dftest['Item_Fat_Content'] = le.fit_transform(dftest['Item_Fat_Content'])
dftest['Item_type_combined'] = le.fit_transform(dftest['Item_type_combined'])
dftest['Item_Identifier'] = le.fit_transform(dftest['Item_Identifier'])
dftest['Item_Outlet_Sales']=0.0
Y_pred = lr.predict(dftest)
dftest['Item_Outlet_Sales']=np.expm1(Y_pred)  # Sales are successfully predicted


        



