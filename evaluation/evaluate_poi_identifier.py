#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score

clf = tree.DecisionTreeClassifier()

clf.fit(features, labels)

prd = clf.predict(features)

print ("Accuracy:", accuracy_score(labels, prd))


from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.3,random_state=42)

clf_split = tree.DecisionTreeClassifier()

clf_split.fit(features_train,labels_train)

prd_split = clf_split.predict(features_test)

print ("Accuracy After Split:", accuracy_score(labels_test, prd_split))

print("Number of POI's: ",len([i for i in labels_test if i == 1]))

print ("Number of people in test sest: ",len(labels_test))

print ("Accuracy If your identifier predicted 0. (not POI) for everyone in the test set:", 1. - (4./29.))


true_positives = 0
for actual, predicted in zip(labels_test, prd_split):
    if actual==1 and predicted==1:       
        true_positives += 1

print ("true positives:", true_positives)
print ("precision score:", precision_score(labels_test, prd_split))
print ("recall score:", recall_score(labels_test, prd_split))


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

from sklearn.metrics import confusion_matrix

print(precision_score(true_labels, predictions))
print(recall_score(true_labels, predictions))
print(confusion_matrix(true_labels, predictions))
