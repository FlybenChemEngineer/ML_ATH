#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Author: Gao Ben
# Schoolof Chemistry and Material Sciences,
# Hangzhou Institute of Advanced Study,
# University of Chinese Academy of Sciences.
#--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os    
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split

data = pd.read_excel(r'C:\Users\Ben\Desktop\Dataset.xlsx',header = None) #读取数据

size = data.shape
raws = size[0]
cols = size[1]

features_label = data.iloc[0,1:cols-1]
target_label = data.iloc[cols-1:]

dataset = data.iloc[1:raws,1:cols].values

# X is the feature dataset, Y is the target dataset
#-------------------------------------------------------
X = dataset[:,0:cols-2]
Y = dataset[:,cols-2]
X1 = np.zeros([raws-1,cols-1])
Y1 = np.zeros(raws-1,)

print("The size of the data read is：",size)

#-------------------------------------------------------
n = 100 #运行交叉验证的次数

scoresresult = np.zeros((n,5))
mseresult = np.zeros((n,5))

m = 0
sumscore = 0
summse = 0


print('Ready to start cross-validation:')

for i in range(0,n):
    
    seed = random.randint(0,100000)
    
    cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    rfr = RandomForestRegressor( n_estimators = 750
                                ,criterion='mse'
                                ,random_state = seed
                                ,n_jobs=-1
                                )
    
    score = cross_val_score(rfr, X, Y, cv=cv) 
    mse = cross_val_score(rfr, X, Y, cv=cv,scoring="neg_mean_squared_error") 
    
    scoresresult[i-1][0]=score[0]
    scoresresult[i-1][1]=score[1]
    scoresresult[i-1][2]=score[2]
    scoresresult[i-1][3]=score[3]
    scoresresult[i-1][4]=score[4]
    
    mseresult[i-1][0]=mse[0]
    mseresult[i-1][1]=mse[1]
    mseresult[i-1][2]=mse[2]
    mseresult[i-1][3]=mse[3]
    mseresult[i-1][4]=mse[4]
    
    summse = summse + (np.sqrt(-mse[0])+np.sqrt(-mse[1])+np.sqrt(-mse[2])+np.sqrt(-mse[3])+np.sqrt(-mse[4]))/5
    sumscore = sumscore + np.mean(score)    
    m = m+1
    
avescore = sumscore/n
avemse = summse/n
print("The calculation is completed, a total of %d times have been calculated"%m)
print("Average R^2：",avescore)
print("Average RMSE：",avemse)
print(scoresresult)
print("Finished")
print("------------------------------------------------------")


# In[ ]:




