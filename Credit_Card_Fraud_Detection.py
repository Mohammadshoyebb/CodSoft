#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING IMPORTANT LIBRARIES

# In[35]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# #### Loading the given dataset to pandas dataframe and taking data set name as cfd(credit_card_fraud_Detection)

# In[36]:


cfd = pd.read_csv(r'C:\Users\shoai\Downloads\creditcard.csv')


# In[37]:


cfd


# In[38]:


#fetching first five rows of given dataset

cfd.head()


# In[39]:


# CHECKING LAST FIVE ENTRIES OF GIVEN DATASET

cfd.tail()


# In[40]:


# FETCHING IMPORTANT DETAILS ABOUT DATASET
cfd.info()


# In[41]:


# CHECKING FOR MISSING VALUES IN THE GIVEN DATASET

cfd.isnull().sum()

#  We don't have any missing values in our given data set

# checking the distribution of legit and fraud transactions where 1 shows fraud transaction and 0 shows legit transaction

# In[42]:


cfd['Class'].value_counts()


# # GIVEN DATASET IS HIGHLY UNBALANCED 
# 

# In[43]:


# separating dataset for analysis

fraud = cfd[cfd.Class == 1]
legit = cfd[cfd.Class == 0]


# In[44]:


print(fraud)


# In[45]:


print(legit)


# In[46]:


print(fraud.shape)
print(legit.shape)


# # STATICAL DESCRIPTION OF DATA

# In[47]:


legit.Amount.describe()


# In[48]:


fraud.Amount.describe()


# In[49]:


#Comparing values of both transaction

cfd.groupby('Class').mean()


# # Under - Sampling

# In[50]:


Legit_sample = legit.sample(n=2000)


# In[51]:


#merging dataset row wise 

new_dataset = pd.concat([Legit_sample, fraud], axis = 0)


# In[52]:


new_dataset.head()


# In[53]:


new_dataset.tail()


# In[54]:


new_dataset['Class'].value_counts()


# In[55]:


new_dataset.groupby('Class').mean()


# ### SPLITTING DATA INTO FETURES AND TARGET
# 

# In[56]:


X= new_dataset.drop(columns = 'Class', axis = 1)
Y = new_dataset['Class']


# In[57]:


print(X)


# In[58]:


print(Y)


# ### SPLIT DATA INTO TRAINING AND TESTING

# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(X , Y , test_size = 0.2 , stratify = Y , random_state = 2)


# In[60]:


print(X_train, X_test, Y_train, Y_test)


# In[61]:


print(X.shape , X_train.shape , X_test.shape)


# ### MODEL  TRAINING  USING  LOGISTIC  REGRESSION

# In[62]:


model = LogisticRegression(max_iter = 10000)


# #### TRAINING LOGISTIC REGRESSION MODEL WITH TRAINING DATA

# In[63]:


model.fit(X_train,Y_train)


# In[64]:


X_train_prediction = model.predict(X_train)


# In[65]:


training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[66]:


print("ACCURACY ON TRAINING DATA : ", training_data_accuracy)


# ### ACCURACY ON TEST DATA

# In[68]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[69]:


print('ACCURACY SCORE ON TEST DATA ; ' , test_data_accuracy)


# In[ ]:




