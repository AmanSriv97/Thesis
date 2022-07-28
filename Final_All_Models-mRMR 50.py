#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#!pip install mrmr_selection
#import mrmr


# In[48]:


import sys
sys.path
sys.path.append(r'c:\users\mailt\appdata\local\packages\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\localcache\local-packages\python39\site-packages')


# In[49]:


get_ipython().system('pip install mrmr_selection')
import mrmr


# In[5]:


#pip install statsmodels


# In[ ]:


X_notcommon = pd.read_csv(r"C:\Users\mailt\Desktop\alzheimer data\finalnotcommon1.csv")
X_notcommon


# In[ ]:


X_notcommon = X_notcommon.iloc[:,1:]
X_notcommon


# In[5]:


# merge both the dataframes

len1 = 68059
len2 = 68861

# creating a label of 1 for alzheimer data and 0 for control data
lstAD = []
for i in range(len1):
    lstAD.append(1)

lstNC =[]
for i in range(len2):
    lstNC.append(0)

Y_Label = lstAD+lstNC


# In[6]:


X_notcommon['Label'] = Y_Label


# In[7]:


X_notcommon


# In[8]:


# shuffling the dataframe
X_notcommon = X_notcommon.sample(frac = 1)
X_notcommon


# In[9]:


Y = np.array(X_notcommon['Label'])


# In[10]:


Y


# In[11]:


# dropping label coloumn from dataframe
X_notcommon= X_notcommon.drop(['Label'], axis=1)


# In[12]:


X_notcommon


# # mrmr selection

# In[17]:


from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X_notcommon, y=Y, K=50)


# In[18]:


selected_features


# In[18]:





# In[19]:


final = X_notcommon[selected_features]
final


# In[ ]:





# In[ ]:





# In[20]:


# split a dataset into train and test sets

from sklearn.model_selection import train_test_split
# create dataset
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(final, Y, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[21]:


from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(50,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))


# In[22]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)


# In[23]:


y_pred = model.predict(X_test)


# In[24]:


score = model.evaluate(X_test, y_test,verbose=1)

print(score)


# In[25]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred.round(), normalize = None)


# In[26]:


precision_score(y_test, y_pred.round())


# In[ ]:





# In[27]:


import numpy as np
from sklearn import metrics


# In[28]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
metrics.auc(fpr, tpr)


# In[29]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[30]:


#create ROC curve
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[48]:


recall_score(y_test, y_pred.round())


# In[45]:


f1_score(y_test, y_pred.round())


# In[46]:


cohen_kappa_score(y_test, y_pred.round())


# In[50]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred.round())
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:





# In[ ]:





# # SVC - L1

# In[13]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.0001, penalty="l1", dual=False).fit(X_notcommon, Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_notcommon)
X_new.shape


# In[14]:


model.get_support()


# In[15]:


constant_columns = [column for column in X_notcommon.columns
                   if column not in X_notcommon.columns[model.get_support()]]

print(len(constant_columns))


# In[16]:


for feature in constant_columns:
    print(feature)


# In[17]:


X_notcommon = X_notcommon.drop(constant_columns, axis =1)


# In[18]:


X_notcommon


# In[43]:


cols= X_notcommon.columns


# In[46]:


for i in cols:
    print (i)


# from mrmr import mrmr_classif
# selected_features = mrmr_classif(X=X_notcommon, y=Y_Label, K=100)

# selected_features

# X_selected = X_total[selected_features]
# X_selected

# In[18]:


#Y = np.array(Y_Label)


# In[19]:


# split a dataset into train and test sets

from sklearn.model_selection import train_test_split
# create dataset
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X_notcommon, Y, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[24]:


from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(153,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))


# In[25]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)


# In[26]:


test_loss


# In[27]:


test_acc


# In[28]:


y_pred = model.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred.round())


# In[ ]:





# In[31]:


score = model.evaluate(X_test, y_test,verbose=1)

print(score)


# In[32]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred.round(), normalize = None)


# In[33]:


precision_score(y_test, y_pred.round())


# In[34]:


recall_score(y_test, y_pred.round())


# In[35]:


f1_score(y_test, y_pred.round())


# In[36]:


cohen_kappa_score(y_test, y_pred.round())


# In[38]:


import numpy as np
from sklearn import metrics


# In[39]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
metrics.auc(fpr, tpr)


# In[40]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[41]:


#create ROC curve
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[42]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred.round())
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# # Extra Validation 

# In[49]:


OutdfAD = pd.read_csv(r"C:\Users\mailt\Desktop\GSE157827\GSE157827_RAW\AD21\AD21df.csv")
OutdfNC = pd.read_csv(r"C:\Users\mailt\Desktop\GSE157827\GSE157827_RAW\NC18\NC18df.csv")


# In[50]:


# merge both the dataframes

lenAD = len(OutdfAD)
lenNC = len(OutdfNC)

# creating a label of 1 for alzheimer data and 0 for control data
lstAD = []
for i in range(lenAD):
    lstAD.append(1)

lstNC =[]
for i in range(lenNC):
    lstNC.append(0)

Yex_Label = lstAD+lstNC


# In[51]:


OutValdf = pd.concat((OutdfAD, OutdfNC), axis = 0)


# In[52]:


OutValdf


# In[53]:


Col = X_notcommon.columns


# In[51]:


Ex_Test = OutValdf[Col]


# In[77]:


Ex_Test['Label'] = Yex_Label


# In[60]:


Ex_Test


# In[61]:


Ex_Test1 = Ex_Test.sample(frac = 1)


# In[62]:


Ex_Test1


# In[63]:


Yex_test = Ex_Test1["Label"]


# In[64]:


Ex_Test1 = Ex_Test1.drop(['Label'], axis=1)


# In[65]:


y_predex = model.predict(Ex_Test1)


# In[66]:


y_predex


# In[67]:


y_predex.round()


# In[68]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(Yex_test, y_predex.round(), normalize = None)


# In[69]:


score = model.evaluate(Ex_Test1, Yex_test,verbose=1)

print(score)


# In[ ]:




