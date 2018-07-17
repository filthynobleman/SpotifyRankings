'''
This module provides a set of functions that can be used to predict if a certain song of a
certain artist in a given date will be in the top X in a certain country.
Obviously, the more X is large, the more the precision is accurate.
Notice that, even if the module has been written to be used to perform the same prediction on
any dataset with a similar structure, it also has been written with in mind the "Spotify's top
200 songs" dataset from Kaggle.
The source of the dataset is:
https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking
'''
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import utils
import datasetinfo
import os


def initialize_dataset(region, labels = True, verbosity_level = 0):
    '''
    This function reads the dataset of the songs and preprocess it. The preprocessing procedure
    consists in imputing the NaN values for the categorical features and encoding those lasts.
    Notice that the preprocessing procedure also drops the URL column and every row that does
    not belongs to the given region, or to one of the given regions.    
    The input parameter labels indicates whether or not the columns for the artist and the
    track name have to be encoded using label encoding or (a more efficient version of) One-Hot
    encoding.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    if verbosity_level > 0:
        print "The initialization procedure has been started..."
    # Read the dataset and drop the URL and Streams columns
    data = utils.read_dataset(verbosity_level)
    #data = data.drop(columns = [datasetinfo.URL_COLUMN, datasetinfo.STREAMS_COLUMN])
    if verbosity_level > 0:
        print "Successfully dropped URL and Streams columns."
    # Drop every line that does not belongs to the given regions
    regs = None
    if type(region) is str:
        regs = [region]
    else:
        regs = list(region)
    data = data[data[datasetinfo.REGION_COLUMN].isin(regs)]
    if len(data) == 0:
        errstr = "There's nothing in the dataset for the following regions:{0}"
        errstr = errstr.format(os.linesep)
        for reg in regs:
            errstr += "    {0}".format(reg)
        raise IndexError(errstr)
    if verbosity_level > 1:
        print "Successfully dropped rows not belonging to the following regions:"
        for reg in regs:
            print "    {0}".format(reg)
    elif verbosity_level > 0:
        print "Successfully dropped rows not belonging to interesting regions."
    # The region is no more needed. Drop it
    data = data.drop(columns = [datasetinfo.REGION_COLUMN])
    if verbosity_level > 0:
        print "Successfully dropped Region column."
    # Impute missing values
    data = utils.impute_categorical_nans(data, inplace = True,
                                               verbosity_level = verbosity_level)
    # Label encodes all the categorical features
    data = utils.label_encode_columns(data, inplace = True, verbosity_level = verbosity_level)
    # If some other kind of encoding is required for artist and track name, use it
    if not labels:
        if verbosity_level > 1:
            print "Label encoding does not fit well the following columns:"
            print "    {0}".format(datasetinfo.ARTIST_COLUMN)
            print "    {0}".format(datasetinfo.TRACKNAME_COLUMN)
        elif verbosity_level > 0:
            print "Some other encoding method is required for some features."
    # Return the dataset
    if verbosity_level > 0:
        print "Initialization procedure has been completed succesffully."
    return data

def topsongs(region, labels = True, top_length = 10, verbosity_level = 0):
    '''
    This function performs the "Top Songs" prediction on the dataset.
    The prediction consists in finding if a certain song, of a certain artist and in a certain
    date will be in the first n songs in a country identified by one of the given region codes.
    Here, n is the input parameter top_length, whose default value is 10.
    The function also performs the same prediction using less informations. As an example, the
    prediction is performed also using only the track name and the date or the track name and
    the artist.
    Finally, the function prints on standard output all the scores obtained with the
    predictions and the informations used to compute those predictions.
    The input parameter labels indicates whether or not the columns for the artist and the
    track name have to be encoded using label encoding or (a more efficient version of) One-Hot
    encoding.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    print """Executing the "Top Songs" prediction."""
    # Read and initialize the dataset
    data = initialize_dataset(region, labels = labels, verbosity_level = verbosity_level)
    # Split the dataset into training and test sets
    train_ratio = 0.75
    train, test = utils.split_dataset_sample(data, frac = train_ratio)
    # Computes the objective columns
    train_y = train[datasetinfo.POSITION_COLUMN] <= top_length
    test_y = test[datasetinfo.POSITION_COLUMN] <= top_length
    # Compute the list of all the feature combinations to examine
    features = [datasetinfo.ARTIST_COLUMN,
                datasetinfo.TRACKNAME_COLUMN, datasetinfo.DATE_COLUMN]
    print """Executing "Top Songs" prediction with the following features:"""
    for col in features:
        print "    {}".format(col)
    clf = DecisionTreeClassifier()
    clf.fit(train[features], train_y)
    pred = clf.predict(test[features])
    print "Scores of the obtained prediction:"
    print "Accuracy:   {}".format(accuracy_score(test_y, pred))
    print "Precision:  {}".format(precision_score(test_y, pred))
    print "Recall:     {}".format(recall_score(test_y, pred))
    print ""

    lastdates = data['Date'].unique()
    lastdates.sort()
    lastdates = lastdates[-40:]
    train = data[data['Date'].isin(lastdates[:31])]
    test = data[data['Date'].isin(lastdates[31:-2])]
    # Computes the objective columns
    train_y = train[datasetinfo.POSITION_COLUMN] <= top_length
    test_y = test[datasetinfo.POSITION_COLUMN] <= top_length
    # Compute the list of all the feature combinations to examine
    features = [datasetinfo.ARTIST_COLUMN,
                datasetinfo.TRACKNAME_COLUMN, datasetinfo.DATE_COLUMN]
    print """Executing "Top Songs" prediction with the following features:"""
    for col in features:
        print "    {}".format(col)
    clf = RandomForestClassifier()
    clf.fit(train[features], train_y)
    pred = clf.predict(test[features])
    print "Scores of the obtained prediction:"
    print "Accuracy:   {}".format(accuracy_score(test_y, pred))
    print "Precision:  {}".format(precision_score(test_y, pred))
    print "Recall:     {}".format(recall_score(test_y, pred))
    print ""