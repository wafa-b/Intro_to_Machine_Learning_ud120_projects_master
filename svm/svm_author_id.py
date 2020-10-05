#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#reduce size-at the cost of accuracy
# features_train = features_train[:int(len(features_train)/100)] 
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
### your code goes here ###
# clf = SVC(kernel="linear")
clf = SVC(kernel='rbf', C=10000)

t0 = time()
clf = clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time()
prd = clf.predict(features_test)
print ("prediction time:", round(time()-t1, 3), "s")

print ("Accuracy:", accuracy_score(labels_test, prd))

print ("Predictions:")
print ("10:", prd[10])
print ("26:", prd[26])
print ("50:", prd[50])

c = Counter(prd)
print ("Number of predictions for Chris(1):", c[1])

#########################################################


