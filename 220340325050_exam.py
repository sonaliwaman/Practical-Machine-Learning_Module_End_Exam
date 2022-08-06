#!/usr/bin/env python
# coding: utf-8

# # Problem Statement 1:

# # Import liabraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Import dataset

# In[4]:


dataset=pd.read_csv("data.csv")


# In[5]:


dataset.head()


# In[6]:


#check shape
dataset.shape


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[137]:


dataset.describe() #min value is .01 and max value 1


# # Spliting data of dependent and independent variable

# In[138]:


X=dataset.iloc[:,:-1].values  #dependent variable
y=dataset.iloc[:,-1].values  #indepdent variable


# In[139]:


X


# In[140]:


y


# In[141]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test =train_test_split(X,y,test_size=1/3, random_state=0)
X_train


# # Built Model

# In[142]:


from sklearn.linear_model import LinearRegression
la=LinearRegression()
la.fit(X_train,y_train)


# # Prediction

# In[143]:



y_pred=la.predict(X_test)
y_pred


# In[144]:


X_test[:3]


# In[145]:


y_test[:3]


# In[146]:


la.predict([[0.68,0.13]])


# In[ ]:





# # Conclusion:

# In[ ]:


Hence test value of F=0.68 and N=0.13 is 361.97 and predicted by 421.


# # Problem Statement 2:
# 

# In[ ]:


#A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lipstick is shown in the table below. Use this dataset to build a decision tree, with Buys as the target variable, to help in buying lip-sticks in the future. Find the root node of the decision tree. According to the decision tree, you have made from the previous Training data set, what is the decision for the test data: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]? Write the concluding statement for the implemented application.


# # Import liabraries

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Import dataset

# In[97]:


data=pd.read_csv("customers .csv")


# In[98]:


data.head()


# In[99]:


data.info()                     #to show info about data


# In[100]:


data=data.drop('ID',axis=1)                      #id column is not required so i will drop it
data.head()


# # Check shape

# In[102]:


data.shape


# In[103]:


data.isnull().sum() 


# #### given dataset have not null value

# In[104]:


data.describe()


# ### using describe function I found min value is 1 and max value is 14.

# In[ ]:





# # Count Value from each column

# In[105]:


data['Income'].value_counts()   


# In[106]:


data['Age'].value_counts()     


# In[107]:


data['Gender'].value_counts()     


# In[111]:


data['Buys'].value_counts()   


# # Countplot for each categorical variable
# 

# In[42]:


plt.figure(figsize=(5,5))
sns.countplot(dataset['Income'])


# In[43]:


plt.figure(figsize=(5,5))
sns.countplot(dataset['Age'])


# In[44]:


plt.figure(figsize=(5,5))
sns.countplot(dataset['Gender'])


# In[112]:


plt.figure(figsize=(5,5)) #check balance
sns.countplot(dataset['Buys'])


# # Preprocessing

# In[113]:


# convert categorical data into numerical data


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data=data.apply(label.fit_transform)


# In[114]:


print(data)


# # Independent and dependent variables

# In[115]:


x=data.drop('Buys',axis=1)
y=data['Buys']


# # split data into train and test data
# 

# In[116]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=12)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:





# # Building Decision tree model

# In[117]:


#build decision tree model


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=12)
classifier.fit(x,y)


# In[118]:


y_pred=classifier.predict(x_test)


# In[119]:


y_pred


# # Prediction for given values of test data

# In[121]:


x_test=np.array([1,1,0,0])


# In[122]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
con_m=confusion_matrix(y_test,y_pred)
con_m


# In[124]:


accuracy_score(y_test,y_pred)


# In[125]:


from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,12))
a = plot_tree(classifier, feature_names=x.columns, fontsize=12, filled=True, 
              class_names=['No', 'yes'])


# In[ ]:





# In[ ]:





# # Conclusion

# In[ ]:


hence Age is <21 , Income is 'Low ',Gender 'Female' ,MaritalStatus is  'Married' Buys column should be 'Yes'
and
hence Age is 21-35 , Income is 'Low ',Gender 'Female' ,MaritalStatus is  'Married' Buys column should be 'Yes'


conclusion matrix and accuracy score we can say that there all obseravtions are correctly predicted.
But there might be possibility of overfitting due to some imbalace data

