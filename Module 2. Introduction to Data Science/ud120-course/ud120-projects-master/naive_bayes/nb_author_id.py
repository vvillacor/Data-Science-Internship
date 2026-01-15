#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import joblib
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# NEW SCRIPT FOR DATA LOADING AND SPLIT
features_train = joblib.load("../word_data_features_train.pkl")
features_test = joblib.load("../word_data_features_test.pkl")
labels_train = joblib.load("../emails_train_labels.pkl")
labels_test = joblib.load("../emails_test_labels.pkl")



##############################################################
# Enter Your Code Here

t0 = time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}s")

t1 = time()
pred = clf.predict(features_test)
print(f"Pred Time: {round(time()-t1, 3)}s")

accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {accuracy}")
#(Accuracy): 0.9550273048506264
##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################