#!/usr/bin/python

import sys
import pickle
import _pickle as cPickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] 
features_list = ['poi', 'deferred_income', 'total_stock_value', 'expenses', 'poi_mail_ratio']# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

#Iterate over dataset and use ad hoc filters to remove other outliers, if necessary
salaryList = []
clean_dict = {}
for name, pdata in data_dict.items():
    #print name
    # Remove email freaks
    if pdata in [range(len("from_messages"))] > 10000:
        continue
    clean_dict[name] = pdata

### Task 3: Create new feature(s)
#Here we'll create a new feature poi_mail_ratio which represents
# the share of emails coming from POI

#Lists used for visualization
from_all=[]
from_poi=[]
m_ratio_list=[]

for name, pdata in clean_dict.items():

    pdata["poi_mail_ratio"] =0
    if pdata['from_messages'] != "NaN":
        from_all.append(float(pdata['from_messages']))
        from_poi.append(float(pdata['from_poi_to_this_person']))
        m_ratio = float(pdata['from_poi_to_this_person']) / float(pdata['from_messages'])
        pdata["poi_mail_ratio"] = m_ratio
        m_ratio_list.append(m_ratio)

print ("m ratios", m_ratio_list)

from matplotlib import pyplot as plt

plt.scatter(from_all, from_poi)
plt.xlabel("all emails")
plt.ylabel("from POI")
plt.show()


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=5) 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.metrics import accuracy_score, precision_score, recall_score

clf.fit(features_train, labels_train)

predicted = clf.predict(features_test)

#Evaluation metrics

acc = accuracy_score(labels_test, predicted)
print ("accuracy:", acc)

precis = precision_score(labels_test, predicted)
print ("precision:", precis)

recall = recall_score(labels_test, predicted)
print ("recall:", recall)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
