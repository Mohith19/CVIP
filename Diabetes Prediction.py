#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv("pima-data.csv")


# In[6]:


data.shape


# In[7]:


data.head(5)


# In[8]:


data.head(10)


# In[9]:


data.isnull().values.any()


# In[10]:


import seaborn as sns


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


corrmat = data.corr()


# In[13]:


top_corr_features = corrmat.index


# In[14]:


plt.figure(figsize=(20,20))


# In[15]:


g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[16]:


data.corr()


# In[17]:


diabetes_map = {True: 1, False: 0}


# In[18]:


data['diabetes'] = data['diabetes'].map(diabetes_map)


# In[19]:


data.head()


# In[20]:


diabetes_true_count = len(data.loc[data['diabetes'] == True])


# In[21]:


diabetes_false_count = len(data.loc[data['diabetes'] == False])


# In[22]:


(diabetes_true_count,diabetes_false_count)


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']


# In[25]:


predicted_class = ['diabetes']


# In[26]:


X = data[feature_columns].values
y = data[predicted_class].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# In[27]:


print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


# In[28]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[29]:


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())


# In[30]:


predict_train_data = random_forest_model.predict(X_test)
from sklearn import metrics
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# In[31]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}


# In[32]:


pip install  xgboost


# In[33]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost


# In[34]:


classifier=xgboost.XGBClassifier()


# In[35]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[36]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[37]:


from datetime import datetime
start_time = timer(None) 
random_search.fit(X,y.ravel())
timer(start_time)


# In[38]:


random_search.best_estimator_


# In[39]:


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.0, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[40]:


classifier.fit(X_train,y_train)


# In[ ]:




