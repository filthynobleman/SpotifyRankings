import pandas as pd
import numpy as np
import sys, os
import utils, preprocessing

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder

# features = ['Artist', 'Track Name', 'Date']
# target = 'Region'

# data = pd.read_csv("data.csv")
# # for feat in features:
# #     le = LabelEncoder()
# #     data[feat] = le.fit_transform(data[feat])
# train, test = utils.split_dataset_sample(data)
# clf = DecisionTreeClassifier()
# clf.fit(train[features], train[target])
# pred = clf.predict(test[features])
# print (pred == test[target]).sum() / float(len(pred))

ds = pd.read_csv("data.csv")
ds = ds[ds['Date'] == '2018-01-01']
print "Readed"
ds = preprocessing.binary_encode(ds, 'Region', True)
print list(ds.columns)
print ds.head()