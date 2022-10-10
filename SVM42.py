#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Gao Ben
# Schoolof Chemistry and Material Sciences,
# Hangzhou Institute of Advanced Study,
# University of Chinese Academy of Sciences.
#--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import os    

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split
from sklearn.svm import SVR

data = pd.read_excel(r'C:\Users\Ben\Desktop\Dataset.xlsx',header = None) 

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

#-------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split( X
                                                    ,Y
                                                    ,test_size = 0.2
                                                    ,random_state = 42)
rfr =  SVR(kernel='rbf')

rfr.fit(X_train,Y_train)

train_score = rfr.score(X_train,Y_train)
test_score = rfr.score(X_test,Y_test)


Y_train_pred = rfr.predict(X_train)
Y_test_pred = rfr.predict(X_test)


train_mse = mean_squared_error(Y_train,Y_train_pred)
train_rmse = np.sqrt(train_mse)
test_mse = mean_squared_error(Y_test,Y_test_pred)
test_rmse = np.sqrt(test_mse)


print('train data RMSE:',train_rmse)
print('train data R2:',train_score)
print('--------------------------------------------')
print('test data RMSE:',test_rmse)
print('test data R2:',test_score)


# In[2]:


plt.rcParams['xtick.direction'] = 'in'  
plt.rcParams['ytick.direction'] = 'in'  

plt.figure(dpi=1000)
plt.figure(figsize=(8,8)) 
plt.xlim(-5,110)
plt.ylim(-5,110)

plt.scatter(Y_train_pred,Y_train,color="#ca3e47", label='train')
plt.scatter(Y_test_pred,Y_test,color="#6b48ff",label='test')

plt.legend(loc='upper left')
plt.title('SVM: RMSE is 26.7.2, R2 is -0.35 (test dataset)')
plt.xlabel('Predicted ee(%)')
plt.ylabel('Observed ee(%)')
plt.grid()
#plt.show()
plt.savefig("./SVM.eps", format='eps', dpi=1000)




# In[ ]:




