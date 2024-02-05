#!/usr/bin/env python
# coding: utf-8

# # Credit Card Transactions Fraud Detection

# ### The purposes of this practise are:
# 
# ### 1. Explore the relationships between different variables 
# 
# ### 2. Construct machine learning models to predict fraud transactions

# The bank is concerned that some fraudulent transactions are going on.

# #### A. Import Packages

# In[2]:


import numpy as np
import pandas as pd
import time

import plotly.express as px
import matplotlib.pyplot as plt


# #### B. Exploratory Data Analysis

# In[3]:


data = pd.read_csv('./creditcardfraud.csv')
data.shape


# In[4]:


data.head()


# The dataset has been processed by PCA and there are 28 main components.

# ##### - Check Null Values

# In[5]:


data.isnull().values.sum()


# ##### - Transaction Classes Distribution

# In[6]:


tran_class = data.Class.value_counts().reset_index()
tran_class.rename(columns = {'index' : 'Label', 'Class': 'Frequency'}, inplace = True)
tran_class['Label'] = tran_class['Label'].map({0:'Normal',1:'Fraud'})
tran_class['Proportion'] = tran_class['Frequency']/tran_class['Frequency'].sum()
print(tran_class)


#  Plot distribution 

# In[7]:


tran_class_fig = px.bar(tran_class, y='Label', x='Frequency', height = 300, width = 800, 
                        text = ['Frequency:{}<br>Proportion:{:.2f}%]'.format(a, 100*b) for a,b in 
                                zip(tran_class['Frequency'], tran_class['Proportion'])])
tran_class_fig.update_layout(title = {'text' : " Label VS Frequency", 'y':0.97, 'x':0.5})


# From the bar chart, it can be observed that there are 492 fraud transactions and 284315 normal transactions

# In[8]:


all_normal = data[data['Class'] == 0]
all_fraud = data[data['Class'] == 1]


# ##### - Explore the time distribution of transactions

# In[9]:


def draw_hist(dataframe, x_name, title, color, nbins_count = 100, height = 400, width = 900, log_y = False):
    fig = px.histogram(dataframe, x_name, nbins = nbins_count, opacity = 0.9, color_discrete_sequence=[color], # color of histogram bars 
                       height = height, width = width, log_y = log_y)
    fig.update_layout(title={'text':title, 'y':0.97, 'x':0.5, 'xanchor':'center','yanchor':'top'})
    fig.show()



# In[10]:


draw_hist(all_normal, 'Time', 'Normal Transaction Distribution', 'rgb(156,219,165)', 100)
draw_hist(all_fraud, 'Time', 'Fraud Transaction Distribution', 'rgb(214,96,77)', 100)


# Applying `log` function to y in order to make the trend more straight forward

# In[11]:


draw_hist(all_normal, 'Time', 'Normal Transaction Distribution', 'rgb(156,219,165)', 100, log_y =True)
draw_hist(all_fraud, 'Time', 'Fraud Transaction Distribution', 'rgb(214,96,77)', 100, log_y =True)


# It is clear that normal transactions distribution is periodic, fraud transactions have an even distribution, thus, we can detect fraud transactions when the normal transactions are at a lower frequency.

# ##### - Explore the transaction amount distribution

# In[12]:


normal_amount_box = px.box(all_normal, x='Amount', width = 700, height = 300)

normal_amount_box.update_layout(title={'text': 'Normal Transaction Amount Boxplot', 'y': 0.97, 'x' : 0.5})
normal_amount_box.show()

fraud_amount_box = px.box(all_fraud, x='Amount', width = 700, height = 300)

fraud_amount_box.update_layout(title={'text': 'Fraud Transaction Amount Boxplot', 'y': 0.97, 'x' : 0.5})
fraud_amount_box.show()


# In[13]:


draw_hist(all_normal, 'Amount', 'Normal Transaction Amount Distribution','rgb(156,219,165)',
          nbins_count=30, height = 300, width = 700)

draw_hist(all_fraud, 'Amount', 'Fraud Transaction Amount Distribution','rgb(270,100,77)',
          nbins_count=30, height = 300, width = 700)


# The amounts of fraud transactions are small. The distributions show there is a huge gap between normal transaction amount and fraud transaction amount. The largest amount of fraud transaction is only $2125.87.

# ##### - Explore the relationship between transaction time and transaction amount

# In[14]:


normal_time_amount_fig = px.scatter(all_normal, x='Time', y = 'Amount', height = 380, width = 700)
normal_time_amount_fig.update_layout(title={'text':'Normal Transaction : Time vs Amount', 'y':0.97, 'x':0.5})
# normal_time_amount_fig.show()


# In[15]:


fraud_time_amount_fig = px.scatter(all_fraud, x='Time', y = 'Amount', height = 380, width = 700)
fraud_time_amount_fig.update_layout(title={'text':'Fraud Transaction : Time vs Amount', 'y':0.97, 'x':0.5})
fraud_time_amount_fig.show()


# The plot of fraud transaction has a more random scatter compare to the plot of normal transaction

# #### C. Data Preprocessing

# The proportions of two transaction types are 0.17% and 99.83%, the dataset is highly unbalanced, therefore, data preprocessing is required to avoid overfitting.

# ##### - Feature Scaling
# 
# Feature scaling is a data preprocessing technique used to transform the values of features or variables in a dataset to a similar scale, it is crutial for data preprocessing to handle highly varying magnitudes.
# 
# `Time` and `Amount` are not on the same scale as other features.
# 
# I choose `RobustScaler` to scale `Time` and `Amount` since there are outliers on the scatter plots above, the outliers could affect the performance of models.

# In[16]:


from sklearn.preprocessing import RobustScaler

#Instantiation

rob_scaler = RobustScaler()

data['scaledAmount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaledTime'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

print('Amount before Scaling：',data['Amount'].values.reshape(-1,1))
print('Amount after Scaling：',data['scaledAmount'].values.reshape(-1,1))
print('Time before Scaling: ',data['Time'].values.reshape(-1,1))
print('Time after Scaling：',data['scaledTime'].values.reshape(-1,1))


# In[17]:


new_data = data.drop(['Time', 'Amount'], axis = 1)
new_data.head


# #### D. Handle Imbalance dataset

# There are different approaches to deal with imbalance dataset, each one has its pros and cons.
# - Oversampling: Oversample minority class using replacement, it could lead to overfitting.
# - Undersampling: Randomly deleting observations from majority class, it could lead to underfitting, the model only learn a part of data.
# 
# I decided to use 2 types of oversampling method to deal with this problem.
# 
# 1. Random Oversampling
#        
#        Con: Overfitting
# 2. SMOTE
#        
#        Con: The model focus too mucn partial information, it leads to overfitting.

# ##### - Spliting training set and test set

# Since the dataset is highly imbalanced, I decide to use `StratifiedShuffleSplit` which can ensure the distribution of labels in training set and test set maintain the same distribution of the whole dataset.

# In[18]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[19]:


X = new_data.iloc[:, new_data.columns != 'Class']
y = new_data.iloc[:, new_data.columns == 'Class']


# In[20]:


# Split into 10 folds.
sss = StratifiedShuffleSplit(n_splits=10,test_size=0.3,train_size=None, random_state=12345)


for train_index, test_index in sss.split(X,y):
    print( 'Train: ', train_index, 'Test: ', test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# In[21]:


# Turn the training sets and test sets into arrays
original_X_train = np.array(original_Xtrain)
original_X_test = np.array(original_Xtest)
original_y_train = np.array(original_ytrain)
original_y_test = np.array(original_ytest)

original_X_train


# ##### - Resampling

# In[22]:


pip install imblearn


# In[23]:


from imblearn.over_sampling import RandomOverSampler, SMOTE


# In[ ]:


#RandomOverSampler
ros = RandomOverSampler(random_state = 0)
X_ros, y_ros = ros.fit_resample(original_X_train, original_y_train)


# In[28]:


#SMOTE
sos = SMOTE(random_state = 0)
X_sos, y_sos = sos.fit_resample(original_X_train, original_y_train)


# The number of samples after resampling

# In[32]:


print('ros:{}, sos:{}'.format(len(y_ros),len(y_sos)))


# In[44]:


print('The number of minority in original dataset is {}\nAfter resampling\nRandam Oversampler : {} and SMOTE : {}\n'.
     format(original_y_train.sum(), y_ros.sum(), y_sos.sum()))

print('The proportion of minority in original dataset is {:.2f}%, it increases to {:.0f}% after resampling\n'.format(100*original_y_train.sum()/len(original_y_train), 100*y_ros.sum()/len(y_ros)))


# #### D. Modelling

# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, classification_report


# ##### - Logistic Regression

# In[51]:


lr = LogisticRegression(max_iter=500)
paramaters = {'C':[0.01,0.1,1,5,10,100]} 
lr_cv1 = GridSearchCV(lr, param_grid = paramaters, cv=10, n_jobs=-1, verbose=5, scoring='f1') 


# This is imbalance dataset, so I use `f1` in scoring

# - Original dataset

# In[63]:


lr_cv1.fit(original_X_train, original_y_train.ravel())
predict_test = lr_cv1.predict(original_X_test)
print('AUC:{} Recall:{} Precision:{}'.format(
    metrics.roc_auc_score(original_y_test, predict_test),
    metrics.recall_score(original_y_test, predict_test),
    metrics.precision_score(original_y_test, predict_test)
))



# - Random Oversampler

# In[64]:


lr_cv1.fit(X_ros, y_ros)
predict_test = lr_cv1.predict(original_X_test)
print('AUC:{} Recall:{} Precision:{}'.format(
    metrics.roc_auc_score(original_y_test, predict_test),
    metrics.recall_score(original_y_test, predict_test),
    metrics.precision_score(original_y_test, predict_test)
))


# - SMOTE

# In[65]:


lr_cv1.fit(X_ros, y_ros)
predict_test = lr_cv1.predict(original_X_test)
print('AUC:{} Recall:{} Precision:{}'.format(
    metrics.roc_auc_score(original_y_test, predict_test),
    metrics.recall_score(original_y_test, predict_test),
    metrics.precision_score(original_y_test, predict_test)
))


# In[66]:


print('Best Parameter：', lr_cv1.best_params_)


# Original sample has the highest precision but recall is low, only 66.9% fraudulent transactions are detected.
# The samples after resampling have higher recall but the precisions drop down.
# 
# Therefore, we can to make some improvement on the model by **assigning weight** to original sample.

# In[72]:


lr = LogisticRegression(max_iter = 500)
param_grid= {'C':[0.01,0.1,1,5,10,100], 
            'class_weight':[{0:1,1:3}, {0:1,1:5},{0:1,1:10}, {0:1,1:15}]
            } 

lr_cv2 = GridSearchCV(lr, param_grid = param_grid, cv=10, n_jobs=-1, verbose=5, scoring='f1')  # n_jobs=-1


lr_cv2.fit(original_X_train, original_y_train)
predict2 = lr_cv2.predict(original_X_test)

print('AUC:{:.3f} Recall:{:.3f} Precision:{:.3f}'.format(
        metrics.roc_auc_score(original_y_test, predict2),
        metrics.recall_score(original_y_test, predict2),
        metrics.precision_score(original_y_test, predict2)
    ))


# In[73]:


print('Best parameters：',lr_cv2.best_params_)


# **Confusion Matrix**

# In[75]:


confusion_matrix = metrics.confusion_matrix(original_y_test, predict2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# #### - Decision Tree

# In[82]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


# Original Sample

# In[83]:


tree_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": list(range(2, 4, 1)), 
    "min_samples_leaf": list(range(5, 7, 1)) 
}
dt_cv = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring='f1')
dt_cv.fit(original_X_train, original_y_train.ravel())


# In[84]:


print('Best Parameters：',dt_cv.best_params_)


# In[85]:


predict3 = dt_cv.predict(original_X_test)
print('AUC:{:.3f} Recall:{:.3f} Precision:{:.3f}'.format(
        metrics.roc_auc_score(original_y_test, predict3),
        metrics.recall_score(original_y_test, predict3),
        metrics.precision_score(original_y_test, predict3)
    ))


# Random Oversampler

# In[87]:


dt_cv.fit(X_ros, y_ros.ravel())

predict4 = dt_cv.predict(original_X_test)
print('AUC:{:.3f} Recall:{:.3f} Precision:{:.3f}'.format(
        metrics.roc_auc_score(original_y_test, predict3),
        metrics.recall_score(original_y_test, predict3),
        metrics.precision_score(original_y_test, predict3)
    ))


# SMOTE

# In[89]:


dt_cv.fit(X_sos, y_sos.ravel())

predict5 = dt_cv.predict(original_X_test)
print('AUC:{:.3f} Recall:{:.3f} Precision:{:.3f}'.format(
        metrics.roc_auc_score(original_y_test, predict3),
        metrics.recall_score(original_y_test, predict3),
        metrics.precision_score(original_y_test, predict3)
    ))


# #### E. ROC Curve

# In[90]:


# Logistic Regression
fpr, tpr, thresholds = roc_curve(original_y_test, predict2)
roc_auc = auc(fpr, tpr)
print('Logistic Regression AUC：{:.2f}%'.format(100*roc_auc))

# Decision Regression
dt_fpr, dt_tpr, dt_thresholds = roc_curve(original_y_test, predict3)
dt_roc_auc = auc(dt_fpr, dt_tpr)
print('Decision Tree AUC：{:.2f}%'.format(100*dt_roc_auc))

# Plot ROC

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='Logistic Regression AUC = %0.3f'% roc_auc)
plt.plot(dt_fpr, dt_tpr, 'y', label='Decision Tress AUC = %0.3f'% dt_roc_auc)
plt.legend(loc='lower right') # 设置legend的位置
plt.plot([0,1],[0,1],'r--') # red, --
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Decision tree has a higher AUC than logistic regression, therefore, decision tree model can identify fraudulent transaction more precisely. 

# In[ ]:



