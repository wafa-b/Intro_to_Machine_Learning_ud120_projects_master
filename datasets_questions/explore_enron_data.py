#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd
import re
import numpy as np


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

df = pd.DataFrame.from_dict(enron_data, orient='index')
print(df.head())

print("Data Points ",len(enron_data))

print("Features Available ", len(enron_data[list(enron_data.keys())[0]]))


poi_dataset = 0
for name,features in enron_data.items():
    if features['poi']:
        poi_dataset += 1        
print("POI in Dataset:", poi_dataset)

poi_total = 0
with open("../final_project/poi_names.txt") as f:
    content = f.readlines()
for line in content:  
    if re.match( r'\((y|n)\)', line):
        poi_total += 1
        
print("POI Total:", poi_total)

print("Total of stock to James Prentice " ,enron_data["PRENTICE JAMES"]["total_stock_value"])

print("Wesley Colwell POI Email messages " ,enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print("Jeffrey K Skilling Stock Options " ,enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

enron_keyPOIPayment = dict((k,enron_data[k]['total_payments']) for k in ("LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"))
max_earner = max(enron_keyPOIPayment, key=enron_keyPOIPayment.get)
print("Largest total payment earner and payment:", max_earner, enron_keyPOIPayment[max_earner])

print("Unfiled featured denoted", enron_data['FASTOW ANDREW S']['deferral_payments'])


salaries_available = 0
emails_available = 0
total_payments_unavailable = 0
total_payments_unavailable_poi = 0
for name in enron_data:
    if not np.isnan(float(enron_data[name]['salary'])):
        salaries_available += 1
    if enron_data[name]['email_address'] != "NaN":
        emails_available += 1
    if np.isnan(float(enron_data[name]['total_payments'])):
        total_payments_unavailable += 1
        if enron_data[name]['poi']:
            total_payments_unavailable_poi += 1


print("Salaries available:", salaries_available)
print("Emails available:", emails_available)

print("NaN for total payment:", total_payments_unavailable, "and percentage:", float(total_payments_unavailable)/len(enron_data)*100)
print("NaN for total payment of POI:", total_payments_unavailable_poi,"and percentage:", float(total_payments_unavailable_poi)/poi_dataset*100)


print("Number of folks when add 10 to datasets",len(enron_data.keys())+10)
print("Nan for total payment when add 10 to datasets",total_payments_unavailable+10)

print("New Number of POI in Dataset", poi_dataset+10)
print("New Number of POI with NaN for total payment of POI", total_payments_unavailable_poi+10)
