#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#!pip install mrmr_selection
#import mrmr


# In[6]:


import sys
sys.path
sys.path.append(r'c:\users\mailt\appdata\local\packages\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\localcache\local-packages\python39\site-packages')


# In[7]:


get_ipython().system('pip install mrmr_selection')
import mrmr


# In[5]:


#pip install statsmodels


# In[8]:


X_notcommon = pd.read_csv(r"C:\Users\mailt\Desktop\alzheimer data\finalnotcommon1.csv")
X_notcommon


# In[9]:


X_notcommon = X_notcommon.iloc[:,1:]
X_notcommon


# In[10]:


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


# In[11]:


X_notcommon['Label'] = Y_Label


# In[12]:


X_notcommon


# In[13]:


# shuffling the dataframe
X_notcommon = X_notcommon.sample(frac = 1)
X_notcommon


# In[14]:


Y = np.array(X_notcommon['Label'])


# In[15]:


Y


# In[16]:


# dropping label coloumn from dataframe
X_notcommon= X_notcommon.drop(['Label'], axis=1)


# In[17]:


X_notcommon


# # mrmr selection

# In[31]:


from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X_notcommon, y=Y, K=100)


# In[32]:


selected_features


# In[18]:


selected_features= ['EDF1',
 'AC004158.1',
 'MT-ATP6',
 'MFAP3L',
 'ANKRD24',
 'LINC00844',
 'MXD4',
 'AC012593.1',
 'PNPLA7',
 'SDC3',
 'SYT12',
 'RPS9',
 'BCOR',
 'LGI4',
 'NXPH1',
 'DNHD1',
 'MAL',
 'METTL26',
 'CTSA',
 'IRF2BP2',
 'IL1RAP',
 'CALR',
 'PIEZO2',
 'CNDP1',
 'HIC2',
 'RPS3',
 'TMOD1',
 'FGF17',
 'GRAMD2B',
 'COL11A1',
 'MAZ',
 'FA2H',
 'HCFC1R1',
 'CRELD1',
 'PAXX',
 'PPP1R14A',
 'SERPINE2',
 'MYO1E',
 'AC139887.2',
 'RPL8',
 'HAGHL',
 'GPR37',
 'ZBTB1',
 'RPL35',
 'KNOP1',
 'OMG',
 'PGK1',
 'GOLGA7',
 'SNAP23',
 'ITGB3BP',
 'MAP1LC3B',
 'LINC00609',
 'LHPP',
 'HPCA',
 'SEMA3B',
 'SPSB3',
 'MIR646HG',
 'SEMA6B',
 'BCL6',
 'C7orf50',
 'ZMIZ1',
 'RNASE1',
 'NTNG1',
 'GNAL',
 'PPDPF',
 'MGAT3',
 'HSPA1B',
 'CHCHD10',
 'MARCKSL1',
 'GINM1',
 'SNX32',
 'PMP22',
 'ZNF516',
 'SCG3',
 'UBA52',
 'ABCC8',
 'JAKMIP3',
 'CPNE5',
 'NENF',
 'MAPK4',
 'SELENOP',
 'PPFIA4',
 'RPL29',
 'TPD52L2',
 'INTU',
 'RPS8',
 'FOXO1',
 'EIF3H',
 'KLF13',
 'PNCK',
 'LARP6',
 'S100B',
 'RFNG',
 'TRAP1',
 'ZDHHC9',
 'OLFM2',
 'C5orf24',
 'CXADR',
 'FANCC',
 'PAIP2']


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
model.add(Dense(12, activation='relu', input_shape=(100,)))

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


# In[52]:


y_pred = model.predict(X_test)


# In[53]:


score = model.evaluate(X_test, y_test,verbose=1)

print(score)


# In[54]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred.round(), normalize = None)


# In[55]:


precision_score(y_test, y_pred.round())


# In[ ]:





# In[56]:


import numpy as np
from sklearn import metrics


# In[57]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
metrics.auc(fpr, tpr)


# In[58]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[59]:


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

# In[7]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(X_notcommon, Y_Label)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_notcommon)
X_new.shape


# In[9]:


model.get_support()


# In[12]:


constant_columns = [column for column in X_notcommon.columns
                   if column not in X_notcommon.columns[model.get_support()]]

print(len(constant_columns))


# In[13]:


for feature in constant_columns:
    print(feature)


# In[14]:


X_notcommon = X_notcommon.drop(constant_columns, axis =1)


# In[46]:


X_notcommon


# from mrmr import mrmr_classif
# selected_features = mrmr_classif(X=X_notcommon, y=Y_Label, K=100)

# selected_features

# X_selected = X_total[selected_features]
# X_selected

# In[18]:


Y = np.array(Y_Label)


# In[19]:


# split a dataset into train and test sets

from sklearn.model_selection import train_test_split
# create dataset
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X_notcommon, Y, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[20]:


from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(701,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))


# In[21]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)


# In[22]:


test_loss


# In[23]:


test_acc


# In[24]:


y_pred = model.predict(X_test)


# In[25]:


score = model.evaluate(X_test, y_test,verbose=1)

print(score)


# In[26]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred.round(), normalize = None)


# In[27]:


precision_score(y_test, y_pred.round())


# In[28]:


recall_score(y_test, y_pred.round())


# In[29]:


f1_score(y_test, y_pred.round())


# In[30]:


cohen_kappa_score(y_test, y_pred.round())


# # Extra Validation 

# In[42]:


OutdfAD = pd.read_csv(r"C:\Users\mailt\Desktop\GSE157827\GSE157827_RAW\AD21\AD21df.csv")
OutdfNC = pd.read_csv(r"C:\Users\mailt\Desktop\GSE157827\GSE157827_RAW\NC18\NC18df.csv")


# In[43]:


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


# In[44]:


OutValdf = pd.concat((OutdfAD, OutdfNC), axis = 0)


# In[45]:


OutValdf


# In[50]:


Col = final.columns


# In[51]:


Ex_Test = OutValdf[Col]


# In[59]:


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




