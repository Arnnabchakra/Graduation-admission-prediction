#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# read the csv file 
admission_df = pd.read_csv('Admission_Predict_Ver1.1.csv')


# In[6]:


admission_df.head()


# In[7]:


admission_df.drop('Serial No.', axis=1, inplace=True)
admission_df.head()


# In[8]:


# checking the null values
admission_df.isnull().sum()


# In[9]:


# Check the dataframe information
admission_df.info()


# In[10]:


# Statistical summary of the dataframe
admission_df.describe()


# In[11]:


# Grouping by University ranking 
df_university = admission_df.groupby(by = 'University Rating').mean()
df_university


# In[12]:


admission_df.hist(bins = 30, figsize = (20, 20), color='r')


# In[13]:


sns.pairplot(admission_df)


# In[14]:


corr_matrix = admission_df.corr()
plt.figure(figsize=(12,12,))
sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[15]:


admission_df.columns


# In[16]:


X = admission_df.drop(columns=['Chance of Admit '])


# In[17]:


y = admission_df['Chance of Admit ']


# In[18]:


X.shape


# In[19]:


y.shape


# In[20]:


X = np.array(X)
y = np.array(y)


# In[21]:


y = y.reshape(-1,1)
y.shape


# In[22]:


y.shape


# In[23]:


# scaling the data before training the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)


# In[24]:


scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


# In[25]:


# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# In[27]:


linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)


# In[28]:


accuracy_LinearRegression = linear_regression_model.score(X_test, y_test)
accuracy_LinearRegression


# In[29]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


# In[30]:


ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim = 7))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(50))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))
ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()


# In[31]:


ANN_model.compile(optimizer='Adam', loss='mean_squared_error')


# In[32]:


epochs_hist = ANN_model.fit(X_train, y_train, epochs = 100, batch_size = 20, validation_split = 0.2)


# In[33]:


result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1 - result
print("Accuracy : {}".format(accuracy_ANN))


# In[34]:


epochs_hist.history.keys()


# In[35]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])


# In[36]:


# Decision tree builds regression or classification models in the form of a tree structure. 
# Decision tree breaks down a dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. 
# The final result is a tree with decision nodes and leaf nodes.
# Great resource: https://www.saedsayad.com/decision_tree_reg.htm

from sklearn.tree import DecisionTreeRegressor
decisionTree_model = DecisionTreeRegressor()
decisionTree_model.fit(X_train, y_train)


# In[37]:


accuracy_decisionTree = decisionTree_model.score(X_test, y_test)
accuracy_decisionTree


# In[38]:


from sklearn.ensemble import RandomForestRegressor
randomForest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
randomForest_model.fit(X_train, y_train)


# In[39]:


accuracy_randomforest = randomForest_model.score(X_test, y_test)
accuracy_randomforest


# In[40]:


y_pred = linear_regression_model.predict(X_test)
plt.plot(y_test, y_pred, '^', color='r')


# In[41]:


y_predict_orig = scaler_y.inverse_transform(y_pred)
y_test_orig = scaler_y.inverse_transform(y_test)


# In[42]:


plt.plot(y_test_orig, y_predict_orig, '^', color='r')


# In[43]:


k = X_test.shape[1]
n = len(X_test)
n


# In[44]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# In[45]:


import pickle


# In[46]:


s = np.array([320, 110, 1, 5, 5, 9, 1])
print(s.shape)
s = s.reshape(1,-1)
print(s.shape)


# In[47]:


pickle.dump(linear_regression_model, open('linear_regression_model_sc.pkl', 'wb'))


# In[48]:


model = pickle.load(open('linear_regression_model_sc.pkl', 'rb'))
print(model.predict(s))


# In[ ]:





# In[ ]:





# In[ ]:




