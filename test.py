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

# ds = pd.read_csv("data.csv")
# ds = ds[ds['Region'] == 'gb']
# print "Readed"
# ds = preprocessing.encode_dates(ds, True)
# ds = preprocessing.encode_artists(ds, True)
# print list(ds.columns)
# print ds[['Date', 'Art0', 'Art1', 'Art2', 'Art3', 'Art4', 'Art5', 'Art6', 'Art7', 'Art8', 'Art9', 'Position']]
        