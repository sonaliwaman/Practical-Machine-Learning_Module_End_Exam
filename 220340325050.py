#!/usr/bin/env python
# coding: utf-8

# # Problem Statement 1:

# # Import liabraries

# In[133]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Import dataset

# In[134]:


dataset=pd.read_csv("data.csv")


# In[135]:


dataset.head()


# In[136]:


#check shape
dataset.shape


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

# In[155]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Import dataset

# In[156]:


dataset=pd.read_csv("customers .csv")


# In[157]:


dataset.head()


# # Check shape

# In[158]:


dataset.shape


# In[159]:


dataset.isnull().sum() 


# #### given dataset have not null value

# In[160]:


dataset.describe()


# ### using describe function I found min value is 1 and max value is 14.

# In[161]:


#spliting data between dependent and independent variable
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[162]:


X


# In[163]:


y


# In[164]:


#Encoding

#preprocessing label encoding(because we want label only like 0,1,2)
from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
label=la.fit_transform(dataset['Buys'])   #Target variable
label


# In[165]:


#label encoding attribute income
Income=la.fit_transform(dataset['Income'])
Income


# In[166]:


Gender=la.fit_transform(dataset['Gender'])
Gender


# In[167]:


MaritalStatus=la.fit_transform(dataset['Marital Status'])
MaritalStatus


# In[168]:


feature=list(zip(Income,Gender,MaritalStatus))


# In[169]:


feature


# In[170]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[171]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(feature)


# # Building Decision tree model

# In[172]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(feature,label)


# In[173]:


y_pred=classifier.predict(feature)


# In[174]:


y_pred


# In[175]:


print((np.concatenate((y_pred.reshape(len(y_pred),1), label.reshape(len(label),1)),1)))


# In[176]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(label,y_pred)
cm


# In[177]:


accuracy_score(label,y_pred)


# In[185]:


d = dataset.iloc[6]
d


# In[187]:


# Test on the New Data
data = {'Age':'<21','Income':'Low','Gender':'Female','MaritalStatus':'Married'}


# In[188]:


data


# # Conclusion

# In[ ]:


hence Age is <21 , Income is 'Low ',Gender 'Female' ,MaritalStatus is  'Married' Buys column should be 'Yes'
and
hence Age is 21-35 , Income is 'Low ',Gender 'Female' ,MaritalStatus is  'Married' Buys column should be 'Yes'

