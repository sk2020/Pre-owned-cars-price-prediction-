 # -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:36:29 2020

@author: shruti
"""
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize':(11.10,8)})   #setting dimensions  for plot
cars_data=pd.read_csv('E:\data sets\cars_sampled.csv')
cars=cars_data.copy(deep=True)
cars.info()
print(cars.describe())
pd.set_option('display.float_format', lambda x:'%.3f'% x)
pd.set_option('display.max_columns',500)
print(cars.describe())
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)
cars.drop_duplicates(keep='first',inplace=True)
print(cars.isnull().sum())                  #data cleaning
yw_count = cars['yearOfRegistration'].value_counts().sort_index()
print(yw_count)
sum(cars['yearOfRegistration'])>2018
sum(cars['yearOfRegistration'])<1950
sns.regplot(x='yearOfRegistration', y='price',scatter=True,fit_reg=False,data=cars)
#histogram
price_count=cars['price'].value_counts().sort_index()
print(price_count)
sns.distplot(cars['price'])
print(cars['price'].describe())
sns.boxplot(y=cars['price'])
sum(cars['price']>100000)
sum(cars['price']>100)

pw_count=cars['powerPS'].value_counts().sort_index()
print(pw_count)
sns.distplot(cars['powerPS'])
print(cars['powerPS'].describe())

sns.boxplot(y=cars['powerPS'])
sns.regplot(x ='powerPS', y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['price']>10)
cars= cars[
    (cars.yearOfRegistration<=2018)
    &(cars.yearOfRegistration>=1950)
    &(cars.price>=100)
    &(cars.price<=150000)
    &(cars.powerPS>=10)
    &(cars.powerPS<=500)]
cars['monthOfRegistration']/=12
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'], 2)
print(cars['Age'].describe())
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='Age', y='price',scatter=True,fit_reg=False,data=cars)
sns.regplot(x='powerPS', y='price',scatter=True,fit_reg=False,data=cars)
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
sns.boxplot(x=cars['abtest'],y=cars['price'])
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x=cars['vehicleType'],y=cars['price'])
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x=cars['gearbox'],y=cars['price'])
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x=cars['model'],y=cars['price'])
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x=cars['kilometer'],y=cars['price'])
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='powerPS', y='price',scatter=True,fit_reg=False,data=cars)
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x=cars['fuelType'],y=cars['price'])
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x=cars['brand'],y=cars['price'])
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x=cars['notRepairedDamage'],y=cars['price'])
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()
cars_selection=cars.select_dtypes(exclude=[object])
correlation=cars_selection.corr()
round(correlation,3)
cars_selection.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
#build a model(linear regression,random forest)
#omitting value
cars_omit=cars.dropna(axis=0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
x1 = cars_omit.drop(["price"], axis = "columns", inplace = False )
y1 = cars_omit["price"]

prices = pd.DataFrame({"1. Before": y1, "2. After":np.log(y1)})
prices.hist()

y1=np.log(y1)
print(y1)
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#baselinemodel
base_pr=np.mean(y_test)
print(base_pr)
base_pr=np.repeat(base_pr,len(y_test))
brmse=np.sqrt(mean_squared_error(y_test, base_pr))
print(brmse)
#intializing the model
lgr=LinearRegression(fit_intercept=True)
model1=lgr.fit(x_train, y_train)
cars_pred=lgr.predict(x_test)
lin_mse=mean_squared_error(y_test,cars_pred)
print(lin_mse)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
#r squared value
r_lintest=model1.score(x_test, y_test) 
r_lintrain=model1.score(x_train, y_train)
print(r_lintest , r_lintrain)
residual = y_test-cars_pred
sns.regplot(x=cars_pred, y=residual,scatter=True,fit_reg=True)
residual.describe()
#model2
rf=RandomForestRegressor(n_estimators=100,max_features="auto",max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)
model_rf=rf.fit(x_train, y_train)
cars_predrf = rf.predict(x_test)
rf_mse = mean_squared_error(y_test,cars_predrf)
print(rf_mse)
rf_rmse = np.sqrt(rf_mse)
print(rf_rmse)

r_rftest  = model_rf.score(x_test,y_test)
r_rftrain = model_rf.score(x_train,y_train)
print(r_rftest,r_rftrain)
cars_inputed = cars.apply(lambda x:x.fillna(x.median())\
                     if x.dtype=='float'  else \
                     x.fillna(x.value_counts().iloc[0]))
cars_inputed.isnull().sum()
cars_inputed = pd.get_dummies(cars_inputed,drop_first=True)
#separate input and output feature
x2=cars_inputed.drop(['price'],axis='columns',inplace=False)
y2=cars_inputed['price']
#plottting
prices = pd.DataFrame({"1. Before": y2, "2. After":np.log(y2)})
prices.hist()
x_train1 , x_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size = 0.3, random_state=1)
print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)
base_pre =np.mean(y_test1)
print(base_pre)

lgr2=LinearRegression(fit_intercept=True)
model_lin=lgr2.fit(x_train1, y_train1)
cp_lin2=lgr2.predict(x_test1)
lmse2=mean_squared_error(y_test1,cp_lin2)
print(lmse2)
lrmse2=np.sqrt(lmse2)
print(lrmse2)
rltrain=model_lin.score(x_train1,y_train1)
rltest=model_lin.score(x_test1,y_test1)
print(rltrain,rltest)
#random forest
rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)
model_rf1=rf2.fit(x_train1, y_train1)
cp_rf2=rf2.predict(x_test1)
rf_mse1=mean_squared_error(y_test1 ,cp_rf2)
rf_rmse2=np.sqrt(rf_mse1)
print(rf_rmse2)
rfltrain=model_rf1.score(x_train1,y_train1)
rfltest=model_rf1.score(x_test1,y_test1)
print(rfltrain,rfltest)
