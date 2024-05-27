#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt


# In[31]:


df=pd.read_csv("C:/Users/User/OneDrive/Desktop/Social_Network_Ads.csv")
df


# In[32]:


df['Gender']=df['Gender'].map({'Male': 1,'Female': 0})
df


# In[33]:


df['Purchased'].value_counts()


# In[34]:


x=df.drop('Purchased',axis=1)
y=df['Purchased']


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[36]:


classifier=SVC(kernel='linear')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[37]:


y_pred


# In[38]:


print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Confusion Matrix : ",confusion_matrix(y_test,y_pred))


# In[39]:


plt.scatter(x_test['EstimatedSalary'],x_test['Age'],c=y_pred)
plt.show()


# In[ ]:




