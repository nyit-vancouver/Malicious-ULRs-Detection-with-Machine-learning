#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import randint
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import xgboost as xgb

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

import csv

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import joblib

from sklearn.preprocessing import StandardScaler


# In[16]:


# Show all the columns
pd.set_option('display.max_columns', None)
# Show all the rows
pd.set_option('display.max_rows',None)


# In[64]:


data = pd.read_csv('result.csv')


# In[65]:


sns.countplot(x = 'result', data = data, order = data['result'].value_counts().index)


# In[63]:


data.head()


# In[19]:


data.info()


# In[20]:


data.shape


# In[21]:


data.columns


# In[22]:


data.head()


# In[23]:


data.describe()


# ## Feature Selection

# In[24]:


#Heatmap
corrmat = data[[
       'result', 'use_of_ip', 'have_@', 'url_length', 'dir_depth', 'is_redirection',
       'is_https', 'have_-', 'domain_Age', 'domain_End', 'email',
       'google_index', 'web_traffic', 'subdomain_len', 'tld_len', 'fld_len',
       'url_path_len', 'hostname_len', 'url_alphas', 'url_digits', 'url_puncs',
       'count.', 'count@', 'count-', 'count%', 'count?', 'count=',
       'count_dirs', 'first_dir_len', 'pc_alphas', 'pc_digits', 'pc_puncs']].corr()
f, ax = plt.subplots(figsize=(25,19))
sns.heatmap(corrmat, square=True, annot = True, annot_kws={'size':10})


# In[33]:


#Predictor Variables
x = data[['use_of_ip', 'have_@', 'url_length', 'dir_depth', 'is_redirection',
       'is_https', 'have_-', 'domain_Age', 'domain_End', 'email',
       'google_index', 'web_traffic', 'subdomain_len', 'tld_len', 'fld_len',
       'url_path_len', 'hostname_len', 'url_alphas', 'url_digits', 'url_puncs',
       'count.', 'count@', 'count-', 'count%', 'count?', 'count=',
       'count_dirs', 'first_dir_len', 'pc_alphas', 'pc_digits', 'pc_puncs']]

#Target Variable
y = data['result']

#Splitting the data into Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 42, shuffle = True)


# In[34]:


from lightgbm import LGBMClassifier

lgb = LGBMClassifier(objective='binary',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
LGB_C = lgb.fit(x_train, y_train)


y_pred = LGB_C.predict(x_test)
print(classification_report(y_test, y_pred))

score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[35]:


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators= 100)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))


score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[36]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=100,max_features='sqrt')
gbdt.fit(x_train,y_train)
y_pred = gbdt.predict(x_test)
print(classification_report(y_test,y_pred))

score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[37]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data["multi_label"] = lb_make.fit_transform(data["result"])
data["multi_label"].value_counts()


# In[38]:


y = data['multi_label']


# In[39]:


def create_multi_class_conf_matrix_plot(y_test, y_pred):
    
    tick_labels = ['benign', 'malicious']
    fig, axes = plt.subplots(1, figsize=(10, 10), sharey=True)
    sns.heatmap(confusion_matrix(y_test, y_pred), ax=axes, annot=True, fmt='g', cmap='Blues')
    axes.set_xticklabels(tick_labels)
    axes.set_yticklabels(tick_labels)


# In[40]:


lgb = LGBMClassifier(objective='binary',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
LGB_C = lgb.fit(x_train, y_train)


y_pred = LGB_C.predict(x_test)
print(classification_report(y_test, y_pred))

score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[41]:


lgb_features = lgb.feature_importances_.tolist()


# In[42]:


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators= 100)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))


score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[43]:


xgb_features = model.feature_importances_.tolist()


# In[44]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=100,max_features='sqrt')
gbdt.fit(x_train,y_train)
y_pred = gbdt.predict(x_test)
print(classification_report(y_test,y_pred))

score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[45]:


gbdt_features = gbdt.feature_importances_.tolist()


# In[46]:


cols = x.columns
feature_importances = pd.DataFrame({'features': cols,
    
    'gbt': gbdt_features,
    'xgb': xgb_features,
    'lgbm': lgb_features                             
    })


# In[47]:


feature_importances['mean_importance'] = feature_importances.mean(axis=1)


# In[48]:


feature_importances = feature_importances.sort_values(by='mean_importance', ascending=False)


# In[49]:


feature_importances


# In[50]:


fig, axes = plt.subplots(1, figsize=(20, 10))
sns.barplot(data=feature_importances, x="mean_importance", y='features', ax=axes)


# In[52]:


fig, axes = plt.subplots(1,3, figsize=(25, 10))
feature_importances = feature_importances.sort_values(by='gbt', ascending=False)
sns.barplot(data=feature_importances, x="gbt", y='features', ax=axes[0])
feature_importances = feature_importances.sort_values(by='xgb', ascending=False)
sns.barplot(data=feature_importances, x="xgb", y='features', ax=axes[1])
feature_importances = feature_importances.sort_values(by='lgbm', ascending=False)
sns.barplot(data=feature_importances, x="lgbm", y='features', ax=axes[2])


# ## Model Training

# In[66]:


# Scaler
standard_scale = StandardScaler()  
inlist = ['dir_depth', 'subdomain_len', 'tld_len', 'fld_len',
       'url_path_len', 'hostname_len', 'url_alphas', 'url_digits', 'url_puncs',
       'count.', 'count@', 'count-', 'count%', 'count?', 'count=',
       'count_dirs', 'first_dir_len']

for item in inlist:
    x_train[item] = standard_scale.fit_transform(x_train[[item]])
for item in inlist:
    x_test[item] = standard_scale.fit_transform(x_test[[item]])


# ###  1. Random Forest

# In[128]:


# Random Forest
rnd_clf = RandomForestClassifier()
rnd_clf.fit(x_train, y_train)
rnd_clf_predictions = rnd_clf.predict(x_test)


# In[129]:


# Print results
print ('The accuracy is:', accuracy_score(y_test, rnd_clf_predictions))
print ('The precision is:', precision_score(y_test, rnd_clf_predictions))
print ('The recall is:', recall_score(y_test, rnd_clf_predictions))
print ('The f1 is:', f1_score(y_test, rnd_clf_predictions))
print (metrics.confusion_matrix(y_test, rnd_clf_predictions))


# ### 2. Logistic Regression

# In[49]:


lg_clf = LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)
lg_clf.fit(x_train, y_train)
lg_clf_predictions = lg_clf.predict(x_test)
accuracy_score(y_test, lg_clf_predictions)

# Print results
print ('The accuracy is:', accuracy_score(y_test, lg_clf_predictions))
print ('The precision is:', precision_score(y_test, lg_clf_predictions))
print ('The recall is:', recall_score(y_test, lg_clf_predictions))
print ('The f1 is:', f1_score(y_test, lg_clf_predictions))
print (metrics.confusion_matrix(y_test, lg_clf_predictions))


# ### 3. KNN

# In[118]:


knn_clf = KNeighborsClassifier(n_neighbors = 3, weights='distance')
knn_clf.fit(x_train, y_train)
knn_clf_predictions = knn_clf.predict(x_test)
accuracy_score(y_test, knn_clf_predictions)

# Print results
print ('The accuracy is:', accuracy_score(y_test, knn_clf_predictions))
print ('The precision is:', precision_score(y_test, knn_clf_predictions))
print ('The recall is:', recall_score(y_test, knn_clf_predictions))
print ('The f1 is:', f1_score(y_test, knn_clf_predictions))
print (metrics.confusion_matrix(y_test, knn_clf_predictions))


# ### 4. SVM

# In[52]:


svm_clf =  SVC(kernel="poly", degree=3, coef0=1, C=0.1,gamma=0.1)
svm_clf.fit(x_train2, y_train2)
svm_clf_predictions = svm_clf.predict(x_test2)

# Print results
print ('The accuracy is:', accuracy_score(y_test, svm_clf_predictions))
print ('The precision is:', precision_score(y_test, svm_clf_predictions))
print ('The recall is:', recall_score(y_test, svm_clf_predictions))
print ('The f1 is:', f1_score(y_test, svm_clf_predictions))
print (metrics.confusion_matrix(y_test, svm_clf_predictions))


# ### 5. Decision Trees

# In[53]:


tree_clf = DecisionTreeClassifier(criterion= 'gini', max_depth=7, random_state=42, max_leaf_nodes=23, min_samples_split = 3)
tree_clf.fit(x_train, y_train)

tree_clf_predictions = tree_clf.predict(x_test)

# Print results
print ('The accuracy is:', accuracy_score(y_test, tree_clf_predictions))
print ('The precision is:', precision_score(y_test, tree_clf_predictions))
print ('The recall is:', recall_score(y_test, tree_clf_predictions))
print ('The f1 is:', f1_score(y_test, tree_clf_predictions))
print (metrics.confusion_matrix(y_test, tree_clf_predictions))


# ### 6. Bagging

# In[54]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=400, bootstrap=True, random_state=42)
bag_clf.fit(x_train, y_train)
bag_clf_predictions = bag_clf.predict(x_test)

# Print results
print ('The accuracy is:', accuracy_score(y_test, bag_clf_predictions))
print ('The precision is:', precision_score(y_test, bag_clf_predictions))
print ('The recall is:', recall_score(y_test, bag_clf_predictions))
print ('The f1 is:', f1_score(y_test, bag_clf_predictions))
print (metrics.confusion_matrix(y_test, bag_clf_predictions))


# ### 7. AdaBoosting

# In[59]:


ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)

ada.fit(x_train, y_train)
ada_predictions = ada.predict(x_test)

# Print results
print ('The accuracy is:', accuracy_score(y_test, ada_predictions))
print ('The precision is:', precision_score(y_test, ada_predictions))
print ('The recall is:', recall_score(y_test, ada_predictions))
print ('The f1 is:', f1_score(y_test, ada_predictions))
print (metrics.confusion_matrix(y_test, ada_predictions))


# ### Model Evaluation

# In[60]:


models_=[]
models_.append(('Logistic',lg_clf))
models_.append(('KNN',knn_clf))
models_.append(('SVM',svm_clf))
models_.append(('Decision Trees',tree_clf))
models_.append(('Random Forest',rnd_clf))
models_.append(('Bagging',bag_clf))
models_.append(('AdaBoosting',ada))


# In[64]:


results_=[]
acc_score_=[]
auc_score_=[]
bias_=[]
f1_score_=[]
precision_score_=[]
recall_score_=[]
names_=[]
for name,model in models_:
    kfold=model_selection.KFold(shuffle=True,n_splits=10,random_state=0)
    cv_results=model_selection.cross_val_score(model,x_train2,y_train2,cv=kfold,scoring='roc_auc')
    results_.append(cv_results)
    bias_.append(np.var(cv_results,ddof=1))
    auc_score_.append(np.mean(cv_results))
    f1=model_selection.cross_val_score(model,x_train2,y_train2,cv=kfold,scoring='f1_weighted')
    f1_score_.append(np.mean(f1))
    
    acc=model_selection.cross_val_score(model,x_train2,y_train2,cv=kfold,scoring='accuracy')
    acc_score_.append(np.mean(acc))
    
    p=model_selection.cross_val_score(model,x_train2,y_train2,cv=kfold,scoring='precision_weighted')
    precision_score_.append(np.mean(p))
    
    r=model_selection.cross_val_score(model,x_train2,y_train2,cv=kfold,scoring='recall_weighted')
    recall_score_.append(np.mean(r))

    names_.append(name)

result_df=pd.DataFrame({'Model':names_,
                           'Accuracy Score':acc_score_,
                            'ROC-AUC Score':auc_score_,
                            'Variance Error':bias_,
                            'F1 Score':f1_score_,
                            'Precision Score':precision_score_,
                            'Recall Score':recall_score_})


# In[65]:


result_df


# In[ ]:


# Save the model
joblib.dump(rnd_clf, 'random_forest.pkl', compress=9)

