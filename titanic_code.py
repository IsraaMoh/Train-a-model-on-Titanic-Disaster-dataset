#!/usr/bin/env python
# coding: utf-8

# #  Train a model on Titanic Disaster dataset
# 

# In[57]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


# Import the dataset

# In[37]:


data = pd.read_csv("train.csv") 
data.head()


# Plot a histogram for Age featrue

# In[38]:


data['Age'].hist()
plt.xlabel("Passenger Age")
plt.ylabel("Passenger Count")


# Plot a scatterplot colored according to survive, plot all survive points with y = 1 as orange and all unsurvive points with y=0  as blue

# In[39]:


sn.FacetGrid(data,hue = 'Survived',palette='muted',height=4).map(plt.scatter,'Pclass', 'Age').add_legend()
plt.show()


# To know if there are missing values in data, I use here count method.

# In[41]:


data.count()


# Through the above information, Age class and Embarked have less information than other features.
# Age is numerical class, we filling the missing values by calculating mean.
# Embarked is categorical class we filling the missing values by finding the mode.

# In[42]:


data.groupby('Embarked').size()


# The (clean data) function perform the next:
# - a. Removing irrelevant features.
# - b. Imputing missing values by computing the mean of numerical features or the most common categorical value.
# - c. Encode categorical values to numerical

# In[43]:


def clean_data(data):
    cols_to_drop = ['PassengerId','Name','Ticket', 'Fare', 'Cabin']
    data.drop(cols_to_drop,1,inplace=True)
    mean_pass_age= data['Age'].mean()
    data['Age'].fillna(mean_pass_age,inplace=True)
    data['Embarked'].fillna('S',inplace=True)
    encode=LabelEncoder()
    encode_cols=['Embarked','Sex']
    data[encode_cols]=data[encode_cols].apply(lambda y: encode.fit_transform(y))
    
    


# In[44]:


clean_data(data)


# In[45]:


data.head()


# The traing_model) function perform the next:
# - Train and test the model using Naive Bayes classifier. Use Gaussian, Multinomial, and Bernoulli classifier.
# - Test and evaluate the performance model by computing the accuracy for each classifier.

# In[85]:


def traing_model(data):
    label = data.iloc[:,:1]
    features=data.iloc[:,1:]
    x_train,x_test, y_train,y_test = train_test_split(features,label, test_size=0.25)
    
    gaussian=GaussianNB()
    berno=BernoulliNB()
    multinom=MultinomialNB()
  
    g_model=gaussian.fit(x_train,y_train.values.ravel())
    g_predict = g_model.predict(x_test)
    g_accuracy=accuracy_score(y_test,g_predict)
    
    b_model=berno.fit(x_train,y_train.values.ravel())
    b_predict = b_model.predict(x_test)
    b_accuracy=accuracy_score(y_test,b_predict)
    
    mul_model=multinom.fit(x_train,y_train.values.ravel())
    mul_predict = mul_model.predict(x_test)
    mul_accuracy=accuracy_score(y_test,mul_predict)
    
    print("Accuracy for Gaussian,Bernoulli and Multinomial Respectively :",format(g_accuracy*100,".2f"),
    format(b_accuracy*100,".2f"), format(mul_accuracy*100,".2f"))
    acc=[g_accuracy*100,b_accuracy*100,mul_accuracy*100]
    a=pd.DataFrame(acc)
    a.hist(bins=70, figsize=(8,8))
    plt.show()
    


# In[88]:


traing_model(data)


# In[ ]:




