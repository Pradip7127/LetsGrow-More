#!/usr/bin/env python
# coding: utf-8

# # Lets Grow More -Data Science Intern

# # Task 1 - Iris Flower Classoification MLProject

# Author:Pradip Jagdale

# This particular ML project is usualy referred to as the "Hello World" of Machine Learning. The iris flowers data set 
# contains numeric attributes and it is perfect for beginners to learn about supervised ML algorithms mainly how to load
# and handle data. Also since this is a small dataset, it can easily fit in memory without requiring special transformation or
# scalling capabilities.

# # Import Libraries

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt


# # Loading Data

# In[35]:


df=sns.load_dataset("iris")
df


# In[36]:


df.head()


# In[37]:


df.tail()


# In[38]:


df.describe()


# # Finding Missing Value

# In[58]:


df.isnull().sum()


# # Visualize the data set

# In[40]:


sns.boxplot(data=df,x='species',y='sepal_length')


# In[41]:


sns.boxplot(data=df,x='species',y='sepal_width')


# In[42]:


sns.boxplot(data=df,x='species',y='petal_length')


# In[43]:


sns.boxplot(data=df,x='species',y='petal_width')


# In[44]:


sns.pairplot(df,hue='species')


# In[45]:


sns.heatmap(df.corr(),annot=True)


# In[46]:


sns.histplot(data=df,x="sepal_length",kde=True)


# In[47]:


sns.histplot(data=df,x="sepal_width",kde=True)


# In[48]:


sns.histplot(data=df,x="petal_length",kde=True)


# In[49]:


sns.histplot(data=df,x="petal_width",kde=True)


# # Data Separation

# In[50]:


x=df.iloc[:,0:4]
x


# In[51]:


y=df.iloc[:,4]
y


# # Train Test Split

# In[52]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)
x_train


# # KNN Classifier

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[54]:


knn=KNeighborsClassifier(n_neighbors=7)
knn
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
y_pred


# # Accuracy

# In[55]:


a=accuracy_score(y_test,y_pred)
print(a)


# # Confusion Matrix

# In[59]:


from sklearn.metrics import confusion_matrix
c_m=confusion_matrix(y_test,y_pred)
print(c_m)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn,x_test,y_test)
plt.show


# # Classification Report

# In[60]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Random Forest Classifier

# In[26]:


from sklearn.ensemble import RandomForestClassifier
#creating random forest classifier
rfc=RandomForestClassifier(n_estimators=10) #By deafault Tress=100
rfc


# In[27]:


#training the classifier
rfc.fit(x_train,y_train)
#applying the trained classifier to the test
y_pred=rfc.predict(x_test)
y_pred


# # Accuracy

# In[28]:


from sklearn.metrics import accuracy_score
a=accuracy_score(y_test,y_pred)
afrom sklearn.metrics import accuracy_score
a=accuracy_score(y_test,y_pred)
a


# In[29]:


from sklearn.metrics import  confusion_matrix
c_m=confusion_matrix(y_test,y_pred)
print(c_m)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(rfc,x_test,y_test)
plt.show()


# # Classification Report

# In[30]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Conclusion

# I have Performed Iris Flower Classification using two classifiers namely Random Forest Classifier And Knn Classifier.
# Accuracy of Random Forest Classifier is 90%.Accuracy of Knn Classifieris 96%. Hence we conclude that KNN Classifier is
# better for classification of iris Flower
