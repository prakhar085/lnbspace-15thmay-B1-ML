#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv('Churn_Modelling.csv')
df.head()


# In[3]:


df.drop('RowNumber',axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


df['Geography'].value_counts()


# In[6]:


df['Gender'].value_counts()


# In[7]:


df['Male'] = pd.get_dummies(df['Gender'],drop_first=True)


# In[8]:


df = pd.concat([df,df['Geography'].str.get_dummies()],axis=1)
df.head()


# In[9]:


df.drop(['CustomerId','Surname','Geography','Gender'],axis=1,inplace=True)


# In[10]:


df.head()


# In[11]:


x = df.drop('Exited',axis=1)
y = df['Exited']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25, random_state=101)


# In[14]:


x.shape


# In[15]:


from keras.metrics import Precision, Recall


# In[16]:


model = Sequential()
model.add(Dense(activation='relu',input_shape=(12,), units=48))
model.add(Dense(activation='relu', units=48))
model.add(Dense(activation='relu', units=48))
model.add(Dense(activation='relu', units=48))
model.add(Dense(activation='relu', units=48))
model.add(Dense(activation = 'sigmoid', units = 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Precision(),Recall()])


# In[17]:


model.summary()


# In[18]:


h = model.fit(xtrain,ytrain,epochs=200,batch_size=10)


# In[19]:


plt.plot(h.history['accuracy'])
plt.legend(['Accuracy'])
plt.grid()
plt.xticks(range(1,51))
plt.xlabel('Epochs-->')
plt.show()


# In[20]:


plt.plot(h.history['loss'])
plt.legend(['Loss'])
plt.grid()
plt.xticks(range(1,51))
plt.xlabel('Epochs-->')
plt.show()


# In[ ]:




