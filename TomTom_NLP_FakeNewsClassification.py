#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing requird python libraries and modules

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# In[5]:


# Reading training and testing CSV file
# File paths are as of my local machine, we can change it to the desired location as needed

train = pd.read_csv("/TomTom/train.csv")
test  = pd.read_csv ("/TomTom/test.csv")


# In[6]:


train.info()


# In[7]:


train.head()


# In[8]:


test.info()


# In[ ]:


# EDA & Pre-processing
train.isnull().sum()


# In[9]:


test=test.fillna(' ')
train=train.fillna(' ')


# In[10]:


# Create a combined column with title, author and text
test['total']=test['title']+' '+test['author']+' '+test['text']
train['total']=train['title']+' '+train['author']+' '+train['text']


# In[11]:


# Dividing the training set into 80/20 ratio using train and test split
X_train, X_test, y_train, y_test = train_test_split(train['total'], train.label, test_size=0.20, random_state=0)


# In[12]:


# Checking on the number of records with splitted training & testing set
X_train.shape


# In[13]:


X_test.shape


# In[17]:


# Initialize the `count_vectorizer` 
count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english') 
# Fit and transform the training data.
count_train = count_vectorizer.fit_transform(X_train)
# Transform the test set 
count_test = count_vectorizer.transform(X_test)
submit_test_count = count_vectorizer.transform(test['total'])


# In[13]:


count_train.shape


# In[15]:


# Initialize the `tfidf_vectorizer` anf transforming the data 
# Stopwords removal is also included in the function itself
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
#Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#Transform the test set 
tfidf_test = tfidf_vectorizer.transform(X_test)
submit_test_tfidf = tfidf_vectorizer.transform(test['total'])


# In[18]:


def print_evaluation_scores(y_val, predicted):
    
    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='weighted'))
    print(average_precision_score(y_val, predicted))


# In[19]:


#Trying fitting decision tree with Count Vectorizer values

dec_tree = DecisionTreeClassifier(max_depth=5)
dec_tree.fit(tfidf_train, y_train)
dec_tree_count = dec_tree.predict(count_test)
print(print_evaluation_scores(y_test,dec_tree_count))


# In[20]:


#Trying fitting decision tree with TFIDF Vectorizer values

dec_tree = DecisionTreeClassifier(max_depth=5)
dec_tree.fit(tfidf_train, y_train)
dec_tree_tfidf = dec_tree.predict(tfidf_test)
print(print_evaluation_scores(y_test,dec_tree_tfidf))


# In[21]:


#Trying random forest with Count Vectorizer values 

random_forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
random_forest.fit(tfidf_train, y_train)
random_forest_count = random_forest.predict(count_test)
print(print_evaluation_scores(y_test,random_forest_count))


# In[22]:


#Trying random forest with TFIDF Vectorizer values 

random_forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
random_forest.fit(tfidf_train, y_train)
random_forest_tfidf = random_forest.predict(tfidf_test)
print(print_evaluation_scores(y_test,random_forest_tfidf))


# In[23]:


#Trying Logistic regression with Count Vectorizer values

logreg = LogisticRegression(C=1e5)
logreg.fit(tfidf_train, y_train)
pred_logreg_count = logreg.predict(count_test)
print(print_evaluation_scores(y_test,pred_logreg_count))


# In[24]:


#Trying Logistic regression with TFIDF values

logreg = LogisticRegression(C=1e5)
logreg.fit(tfidf_train, y_train)
pred_logreg_tfidf = logreg.predict(tfidf_test)
print(print_evaluation_scores(y_test,pred_logreg_tfidf))


#  As we can see from the evaluation scores of average precision & f1, simple logistic regression performs best with TFIDF Vectorization. Hence we are finally selectig logistic regression with TFIDF vectorization as our model to run on our test file

# In[25]:


#Runinng finalized logistic regression model on TEST.csv file
submit_pred_logreg_tfidf = logreg.predict(submit_test_tfidf)


# In[26]:


# Writing it back to the Submit.csv file 

final_sub = pd.DataFrame()
final_sub['id']=test['id']
final_sub['label'] = submit_pred_logreg_tfidf
final_sub.to_csv("/TomTom/Submit.csv",index=False)

