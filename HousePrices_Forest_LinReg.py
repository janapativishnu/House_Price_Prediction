# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:28:23 2018

@author: Vishnuvardhan Janapati
"""

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import svm, preprocessing, cross_validation, linear_model
import seaborn as sns
from scipy.stats import norm
from scipy import stats

# 1 : defining the problem: 

# Given a training dataset with 79 features of a home and its sale price, learn a model that can predict sale price of a home


# 2: load data set

# training data
train_df=pd.read_csv('train.csv')  
# test data
test_df=pd.read_csv('test.csv')
# submission data to kaggle
samp_submission_df=pd.read_csv('sample_submission.csv');

# 3: Understand the problem (relationship between features and sale price)
# check features and sale price
#print(train_df.count())

# check stats of each features
#print(train_df.describe())

# check stats of each sale price
#print(train_df.SalePrice.describe())

# plot of distribution of sale price
#sns.distplot(train_df['SalePrice'])

#print("Skewness of sale price is :",train_df['SalePrice'].skew())
#print("Kurtosis of sale price is :",train_df['SalePrice'].kurt())


## missing data %ge
total=train_df.isnull().sum().sort_values(ascending=False)
percent=(train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)

# delete features corresponding to missing data more than 1% (will be different for other problems)
drop_cols=(missing_data[missing_data['Percent']>0.03]).index
#len((missing_data[missing_data['Percent']>0.01]))
#train_df=train_df.drop((missing_data[missing_data['Percent']>0.01]).index,1)
train_df=train_df.drop(drop_cols,1)
int_float_cols=train_df.select_dtypes(['int64','float64']).columns

# convert categorical 
cat_cols=train_df.select_dtypes(['object']).columns
train_df[cat_cols]=train_df[cat_cols].astype('category')
train_df[cat_cols]=train_df[cat_cols].apply(lambda x:x.cat.codes)

train_df.fillna(value=train_df.median(),inplace=True)
# checking missing data
print("Number of missing data : ",train_df.isnull().sum().max())

# -------   4. Select training data and features
y=train_df['SalePrice']
X=train_df.drop(['Id','SalePrice'],1)

# Normalize data
#scaler=preprocessing.StandardScaler()
scaler=preprocessing.RobustScaler()
X=scaler.fit_transform(X)

# process (test set data)
test_df=test_df.drop(drop_cols,1)
test_df[cat_cols]=test_df[cat_cols].astype('category')
test_df[cat_cols]=test_df[cat_cols].apply(lambda x:x.cat.codes)

test_df.fillna(value=test_df.median(),inplace=True)
# checking missing data
print("Number of missing data : ",test_df.isnull().sum().max())
X_test=test_df.drop(['Id'],1)
X_test=scaler.transform(X_test)


# split data for testing parameters
#X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

# -------   5. Model preparation  (Random Forest)
import sklearn.ensemble as ske

reg_forest=ske.RandomForestRegressor(n_estimators=100).fit(X,y)
predict_forest=reg_forest.predict(X_test)
submission_forest=pd.DataFrame({'Id':samp_submission_df['Id'],'SalePrice':predict_forest})
submission_forest.to_csv('Submission.csv',index=False)
print("The accuracy of prediction with train set using Random Forest is " + str(round(reg_forest.score(X,y)*100,2)) +" Percent")


# -------   5. Model preparation  (Linear Regression Model)
Lreg=linear_model.LassoCV(eps=0.000008,max_iter=600,tol=1e-7)
#
Lreg.fit(X,y)
predict_Lreg=Lreg.predict(X_test)
submission_Lreg=pd.DataFrame({'Id':samp_submission_df['Id'],'SalePrice':predict_Lreg})
submission_Lreg.to_csv('Submission_Lreg.csv',index=False)
print("The accuracy of prediction with train set using Linear Regression is " + str(round(Lreg.score(X,y)*100,2)) +" Percent")


# -------   5. Model preparation  (SVM for regression)
reg_svm=svm.SVR(kernel='linear',C=100.0)
#
reg_svm.fit(X,y)
predict_reg_svm=Lreg.predict(X_test)
submission_reg_svm=pd.DataFrame({'Id':samp_submission_df['Id'],'SalePrice':predict_reg_svm})
submission_reg_svm.to_csv('Submission_reg_svm.csv',index=False)
print("The accuracy of prediction with train set using SV Regression is " + str(round(reg_svm.score(X,y)*100,2)) +" Percent")



# Random forest was so far best for this problem































