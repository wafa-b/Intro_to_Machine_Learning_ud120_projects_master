#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

clf_KNeighbors = KNeighborsClassifier(n_neighbors=4)
clf_KNeighbors.fit(features_train, labels_train)
prd_KNeighbors = clf_KNeighbors.predict(features_test)
print ("Accuracy KNeighbors:", accuracy_score(labels_test, prd_KNeighbors))

clf_RandomForest = RandomForestClassifier(n_estimators=15, min_samples_split=6)
clf_RandomForest.fit(features_train, labels_train)
prd_RandomForest = clf_RandomForest.predict(features_test)
print ("Accuracy RandomForest:", accuracy_score(labels_test, prd_RandomForest))

clf_AdaBoost=AdaBoostClassifier(n_estimators=100, random_state=0)
clf_AdaBoost.fit(features_train, labels_train)
prd_AdaBoost = clf_AdaBoost.predict(features_test)
print ("Accuracy AdaBoost:", accuracy_score(labels_test, prd_AdaBoost))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
