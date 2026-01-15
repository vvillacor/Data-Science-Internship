#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import joblib
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### 
### NEW SCRIPT FOT DATA LOADING 
features_train = joblib.load("../word_data_features_train.pkl")
features_test = joblib.load("../word_data_features_test.pkl")
labels_train = joblib.load("../emails_train_labels.pkl")
labels_test = joblib.load("../emails_test_labels.pkl")


#########################################################
### your code goes here ###

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]


#clf = SVC(kernel = "linear", C=10000.0)
# acc = Precisión (Accuracy): 0.8760038548024414
clf = SVC(kernel = "rbf", C=10000.0)
# acc =  0.9081272084805654
#training time
t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}s")

#pred time 
t1 = time()
pred = clf.predict(features_test)
#pred is an array
print(f"Prediction time: {round(time()-t1, 3)}s")

# acc
accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {accuracy}")

