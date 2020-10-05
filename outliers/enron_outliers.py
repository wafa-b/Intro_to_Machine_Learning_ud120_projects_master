#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
# To remove key-value of biggest Enron outlier
data_dict.pop('TOTAL',0)

data = featureFormat(data_dict, features)


### your code below
print (data.max())

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


max_value=sorted(data,reverse=True,key=lambda sal:sal[0])[0]
print (max_value)

# To find  dictionary key for biggest Enron outlier
for item in data_dict:
    if data_dict[item]['salary']==max_value[0]:
	    print ("dictionary key for biggest Enron outlier: ", item)


# To find Two people made bonuses of at least 5 million dollars, and a salary of over 1 million dollars 
for item in data_dict:
    if data_dict[item]['bonus']!='NaN' and data_dict[item]['salary']!='NaN':
	    if data_dict[item]['bonus']>5e6 and data_dict[item]['salary']>1e6:
		    print (item)
