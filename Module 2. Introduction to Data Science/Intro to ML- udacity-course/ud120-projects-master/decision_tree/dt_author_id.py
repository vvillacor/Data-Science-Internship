#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import joblib
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### NEW SCRIPT FOT DATA LOADING 
features_train = joblib.load("../word_data_features_train.pkl")
features_test = joblib.load("../word_data_features_test.pkl")
labels_train = joblib.load("../emails_train_labels.pkl")
labels_test = joblib.load("../emails_test_labels.pkl")

#########################################################
### YOUR CODE HERE ###
### THIS EXCERCISES WERE PROVIDED FROM 
### https://docs.google.com/document/d/1h6UwiyNjdoyiQz6reh2sfch1O5A0dfYP1I8pH9G05eM/edit?tab=t.0


###############           PART 1             ################

clf = DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}s")
t1 = time()
pred = clf.predict(features_test)
print(f"Pred time: {round(time()-t1, 3)}s")
accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {accuracy}")

# Accuracy: 0.9743013170575008

###############           PART 2             ################

#  FEATURE SELECTION (MINI PRACTICE)
#code change in fix_enron_data.py in order to 

# FEATURES WITH selector = SelectPercentile(f_classif, percentile=10)
#print(len(features_train[0]))
#5377 features in kaggle ds / 3785 in course ds

# FEATURES WITH selector = SelectPercentile(f_classif, percentile=1)
## PERCENTILE = 1
##538 features in kaggle ds /  379 in course ds

###############           PART 3             ################

# What's the accuracy of your decision tree when you use only 1% of 
# your available features (i.e. percentile=1)?

# Accuracy = 0.9617732091230324


#########################################################
