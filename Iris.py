#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv('C:/Users/UDHAYA KUMAR . R/Desktop/Oasis/archive (1).zip')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['Species'].unique()


# In[7]:


df['Species'].value_counts()


# In[8]:


df['Species'].replace({'Iris-setosa':'setosa','Iris-versicolor':'versicolor','Iris-virginica':'virginica'},inplace=True)


# In[9]:


df.columns


# In[10]:


df['Species']


# In[11]:


df.head()


# In[12]:


df.drop('Id', axis=1,inplace=True)


# In[13]:


df.head()


# # Visualizing the dataset

# In[52]:


#sns.boxplot(df['SepalLengthCm'])

plt.boxplot(df['SepalLengthCm'])
plt.text(x=1.1,y=df['SepalLengthCm'].min(),s='min')
plt.text(x=1.1,y=df['SepalLengthCm'].quantile(0.25),s='Q1')
#plt.text(x=1.1,y=df['SepalLengthCm'].quantile(0.50),s='Q2')
plt.text(x=1.1,y=df['SepalLengthCm'].median(),s='Median')
plt.text(x=1.1,y=df['SepalLengthCm'].quantile(0.75),s='Q3')
plt.text(x=1.1,y=df['SepalLengthCm'].max(),s='Max')


plt.title('Boxplot of Sepal Length in cm')


plt.show()


# In[65]:


plt.boxplot(df['SepalWidthCm'])
plt.text(x=1.1,y=df['SepalWidthCm'].min(),s='min')
plt.text(x=1.1,y=df['SepalWidthCm'].quantile(0.25),s='Q1')
plt.text(x=1.1,y=df['SepalWidthCm'].median(),s='Median')
plt.text(x=1.1,y=df['SepalWidthCm'].quantile(0.75),s='Q3')
plt.text(x=1.1,y=df['SepalWidthCm'].max(),s='Max')


plt.title('Boxplot of Sepal Length in cm')


plt.show()


# In[68]:


plt.boxplot(df['PetalLengthCm'])
plt.text(x=1.1,y=df['PetalLengthCm'].min(),s='min')
plt.text(x=1.1,y=df['PetalLengthCm'].quantile(0.25),s='Q1')
plt.text(x=1.1,y=df['PetalLengthCm'].median(),s='Median')
plt.text(x=1.1,y=df['PetalLengthCm'].quantile(0.75),s='Q3')
plt.text(x=1.1,y=df['PetalLengthCm'].max(),s='Max')


plt.title('Boxplot of Sepal Length in cm')


plt.show()


# In[70]:


plt.boxplot(df['PetalWidthCm'])
plt.text(x=1.1,y=df['PetalWidthCm'].min(),s='min')
plt.text(x=1.1,y=df['PetalWidthCm'].quantile(0.25),s='Q1')
plt.text(x=1.1,y=df['PetalWidthCm'].median(),s='Median')
plt.text(x=1.1,y=df['PetalWidthCm'].quantile(0.75),s='Q3')
plt.text(x=1.1,y=df['PetalWidthCm'].max(),s='Max')


plt.title('Boxplot of Sepal Length in cm')


plt.show()


# In[73]:


df.corr()


# In[83]:


fig,ax=plt.subplots(figsize=(6,4))
sns.heatmap(df.corr(),annot=True,ax=ax)

plt.show()


# In[88]:


sns.pairplot(df,hue='Species')

plt.show()


# # Data Pre-Processing

# In[90]:


data={'setosa':1,'versicolor':2,'virginica':3}


# In[91]:


data


# In[92]:


df.Species = [data[i] for i in df.Species]
df


# In[103]:


X=df.iloc[:,0:4]
X


# In[109]:


y=df.iloc[:,4]
y


# In[112]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33,random_state=42)


# # Traning Model

# In[113]:


model=LinearRegression()


# In[114]:


model


# In[115]:


model.fit(X,y)


# In[116]:


model.score(X,y)


# In[117]:


model.coef_


# In[118]:


model.intercept_


# # Predictions

# In[131]:


predictions=model.predict(X_test)


# # Model Evoultion

# In[132]:


mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# In[137]:


#print('Mean Squared Error: %.2f' % np.mean((predictions - y_test)**2))


# In[ ]:




