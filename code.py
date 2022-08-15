#!/usr/bin/env python
# coding: utf-8

# ## 2021F CBD 2204 1 [B108] Big Data Strategies Assignment 1- Classification
# ## Prasanth Moothdath Padmakumar - C0796752
# 
# 
# ## Stroke Prediction Dataset 
# https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
# ### Predicting if an individual will have a stroke or not based on some parameters
# 
# 

# #### Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# #### Importing Datasets

# In[2]:


df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.head()


# In[3]:


df.info()


# #### Checking for noise and cleaning data

# In[4]:


# Removing ID column
if ('id' in df):
    df.drop(['id'], axis = 1, inplace=True)

# checking for missing values and clearing them

# print ("\nMissing values :  ", df.isNan().sum().values.sum())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
df.isnull().sum().sort_values(ascending=False)[:]


# In[5]:


# Remove entries with bmi values as null
df.dropna(how='any', inplace=True)


# #### Data preprocessing

# In[6]:


# Assigning urban resident as 0 and rural resident as 1
residence_mapping = {'Urban': 0, 'Rural': 1}
df['Residence_type'] = df['Residence_type'].map(residence_mapping)

# Assigning unmarried as 0 and mariied as 0
marriage_mapping = {'No': 0, 'Yes': 1}
df['ever_married'] = df['ever_married'].map(marriage_mapping)

df.sample()
df


# In[7]:


#### With columns having non numerical values, we create new column for each of the values and mark 1 or 0


# In[8]:


# Altering gender column
if ('gender' in df):
    print("Gender Values : ", df['gender'].unique() )
    df['gender'] = pd.Categorical(df['gender'])
    df_dummies_gender = pd.get_dummies(df['gender'], prefix = 'gender_')
    df.drop("gender", axis=1, inplace=True)
    df = pd.concat([df, df_dummies_gender], axis=1)
df.sample()
df


# In[9]:


# Altering smoking_status column
if ('smoking_status' in df):
    print("Smoking status Values : ", df['smoking_status'].unique() )
    df['smoking_status'] = pd.Categorical(df['smoking_status'])
    df_dummies_smoking_status = pd.get_dummies(df['smoking_status'], prefix = 'smoking_status_')
    df.drop("smoking_status", axis=1, inplace=True)
    df = pd.concat([df, df_dummies_smoking_status], axis=1)
df.sample()


# In[10]:


# Altering work_type column
if ('work_type' in df):
    print("Work type Values : ", df['work_type'].unique() )
    df['work_type'] = pd.Categorical(df['work_type'])
    df_dummies_work_type = pd.get_dummies(df['work_type'], prefix = 'work_type_')
    df.drop("work_type", axis=1, inplace=True)
    df = pd.concat([df, df_dummies_work_type], axis=1)
df.sample()
df


# In[11]:


# Scaling using standard scaler
featuresToScale = ['avg_glucose_level','bmi','age']
standardScalar = StandardScaler()
df[featuresToScale] = standardScalar.fit_transform(df[featuresToScale])
df


# #### Splitting train and test data

# In[12]:


# Initializing dependant and independent variables  
y = df["stroke"]
X = df.drop(['stroke'],axis=1)


# In[13]:


# Splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 101)
print('Total entries in whole dataset:', len(X))
print('Total entries in train dataset:', len(X_train))
print('Total entries in test dataset:', len(X_test))


# #### K-NN Classification

# In[14]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
k_5_accuracy = accuracy_score(y_test, knn_predicted)
print("Accuracy of K-NeighborsClassifier with k as 5:", k_5_accuracy,'\n')


# #### Finding best value for k

# In[15]:


from sklearn import metrics
accuracy = []
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, yhat))
    
plt.plot(range(1,40),accuracy,'g')
plt.ylabel('Accuracy')
plt.xlabel('K Value')
plt.tight_layout()
plt.show()

print("Best value for k: ", max(accuracy),"at K =", accuracy.index(max(accuracy)))


# ### In the graph we can see accuracy reaches the saturation point of around 0.94 at k value of 7
