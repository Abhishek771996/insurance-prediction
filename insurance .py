#!/usr/bin/env python
# coding: utf-8

# In[1]:


# regression

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


data=pd.read_csv('insurance.csv') 
print(data)


# In[3]:


data.isnull().sum() 


# In[4]:


x=data.iloc[:,:-1] # independent
x


# In[5]:


y=data['charges']  # dependent
y


# In[6]:


# encoding
from sklearn.preprocessing import LabelEncoder  
le=LabelEncoder()                                  
x['gender']=le.fit_transform(x['gender']) 
x


# In[7]:


lo=LabelEncoder()   
x['smoker']=lo.fit_transform(x['smoker'])
x


# In[8]:


# one hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encode",OneHotEncoder(drop="first",sparse=False),[5])],remainder="passthrough")
x=ct.fit_transform(x)            


# In[9]:


print(x) 
  
 


# In[10]:


# feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)                       


# In[11]:


x


# In[12]:


# splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) # random state use hogi taki value bar bar change na ho
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[13]:


# linear Regression algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()                
               
regressor.fit(x_train,y_train)   


# In[15]:


y_pred=regressor.predict(x_test) 
y_pred                                                


# In[16]:


# optional                                           
ins_pre=pd.DataFrame(y_pred)
ins_pre                     


# In[20]:


# find the mae
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_pred))    


# In[21]:


# r2score of model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))    


# In[22]:


# support vector regression
from sklearn.svm import SVR
regressor=SVR(C=5000)               
regressor.fit(x_train,y_train)  


# 

# In[23]:


y_pred=regressor.predict(x_test) 
y_pred                                                


# In[24]:


# find the mae
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_pred))    


# In[25]:


# r2score of model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))    


# 
