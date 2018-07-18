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

TOPSONGS_FEATURES = [datasetinfo.ARTIST_COLUMN,
                     datasetinfo.DATE_COLUMN,
                     datasetinfo.TRACKNAME_COLUMN]
'''
This list represents the features list that is used in the "Top Songs" prediction.
It cannot be supplied by the user.
'''


def initialize_dataset(dataset = None, region = 'global', verbosity_level = 0):
    '''
    This function preprocess the given songs' dataset. If no dataset is given, then the
    function reads a default dataset and applies the preprocessing procedure to this one.
    The preprocessing procedure consists in imputing the NaN values for the categorical
    features and encoding those lasts.
    Notice that the preprocessing procedure also drops the URL and Streams columns and every
    row that does not belongs to the given region, or to one of the given regions.
    In no region is given, then the 'global' region is assumed.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    if verbosity_level > 0:
        print "The initialization procedure has been started..."
    # Read the dataset and drop the URL and Streams columns
    if dataset is None:
        data = utils.read_dataset(verbosity_level)
    else:
        data = dataset
    data = data.drop(columns = [datasetinfo.URL_COLUMN, datasetinfo.STREAMS_COLUMN])
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
    # if not labels:
    #     if verbosity_level > 1:
    #         print "Label encoding does not fit well the following columns:"
    #         print "    {0}".format(datasetinfo.ARTIST_COLUMN)
    #         print "    {0}".format(datasetinfo.TRACKNAME_COLUMN)
    #     elif verbosity_level > 0:
    #         print "Some other encoding method is required for some features."
    # Return the dataset
    if verbosity_level > 0:
        print "Initialization procedure has been completed succesffully."
    return data

def fit_classifier(training_set, top_length = 10, verbosity_level = 0):
    '''
    This function returns a Sci-Kit classifier trained with the given training set.
    The type and the parameters of the classifier are hard-coded in the function, according
    to which settings resulted to be the optimal ones. The features' list is given by the
    global variable TOPSONGS_FEATURES.
    The top_length parameter defines the length of the list of the top songs on which the
    classifier is going to be trained. Default value is 10.
    The function operates splitting the training set into features and target sets and then
    fitting the classifier. Once the fitting operation is completed, the classifier is
    returned.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    # Creates the features set and the target sets
    global TOPSONGS_FEATURES
    train_x = training_set[TOPSONGS_FEATURES]
    train_y = training_set[datasetinfo.POSITION_COLUMN] <= top_length
    # Create the classifier
    if verbosity_level > 0:
        print "Training the classifier for \"Top Songs\" prediction..."
    clf = DecisionTreeClassifier(criterion = 'gini')
    clf.fit(train_x, train_y)
    if verbosity_level > 0:
        print "The classifier has been trained successfully."
    if verbosity_level > 1:
        print "The decision path of the trained decision tree over the training set is:"
        print clf.decision_path(train_x)
    return clf

def predict_data(classifier, data, verbosity_level = 0):
    '''
    This function applies the prediction of the given classifier to the given data and
    returns the computed predicted classification.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    if verbosity_level > 0:
        print "Applying the prediction..."
    pred = classifier.predict(data)
    if verbosity_level > 0:
        print "Prediction completed successfully."
    return pred

def predict_test_data(classifier, test, top_length = 10, verbosity_level = 0):
    '''
    This function applies the prediction of the given classifier to the given test set and
    returns a 3-tuple containing the following informations:
        - Accuracy score
        - Precision score
        - Recall score
    The test set is splitted by the function into features set and target set.
    The features' list is given by the global variable TOPSONGS_FEATURES.
    The top_length parameter defines the length of the list of the top songs on which the
    classifier is going to be trained. Default value is 10.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    # Split the test set into features and target
    global TOPSONGS_FEATURES
    test_x = test[TOPSONGS_FEATURES]
    test_y = test[datasetinfo.POSITION_COLUMN] <= top_length
    # Apply the prediction
    pred = predict_data(classifier, test_x, verbosity_level)
    # Compute the scores and return them
    accuracy = accuracy_score(test_y, pred)
    precision = precision_score(test_y, pred)
    recall = recall_score(test_y, pred)
    return (accuracy, precision, recall)

def topsongs(training_set, data, top_length = 10, verbosity_level = 0):
    '''
    This function performs the "Top Songs" prediction.
    The prediction consists in finding if a cartain song of a certain artist will be, in a
    given date, in the list of the top songs. The length of the list of the top songs is
    defined by the top_length parameter, whose default value is 10.
    The function trains a classifier using the given training_set and then computes the 
    prediction on the given dataset.
    The results of the prediction are then returned by the function.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    # Train a classifier over the training set
    clf = fit_classifier(training_set, top_length, verbosity_level)
    # Compute the prediction and return the result
    return predict_data(clf, data, verbosity_level)

def topsongs_main(training_file,    # The file used to train the classifier
                  data_file,        # The file containing the data to predict
                  output_file,      # The file in which the prediction is printed
                  regions,          # The region codes for the interesting regions
                  top_length,       # The length of the list of the top songs
                  verbosity_level): # The verbosity level of the debug
    '''
    This is the entry point of the program for the topsongs functionality.
    The function takes as input the filename of the file containing the data used to train the
    classifier and the filename of the file containing the data that must be classified.
    Also, the filename of the file in which the prediction has to be printed is given to this
    function.
    The input parameter region defines the region codes that limits the dataset to the regions
    which we are interested in.
    The input parameter top_length defines the length of the list of the top songs that we
    want to determine.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    print "Executing the \"Top Songs\" procedure..."
    # Read the datasets
    if verbosity_level > 1:
        print "Reading the datasets..."
    train = pd.read_csv(training_file)
    data = pd.read_csv(data_file)
    # Initialize the datasets
    train = initialize_dataset(train, region, verbosity_level)
    data = initialize_dataset(data, region, verbosity_level)
    if verbosity_level > 0:
        print "Datasets successfully readed and initialized."
    # Execute the topsongs procedure and get the prediction
    if verbosity_level > 1:
        print "Executing the training and prediction procedure..."
    pred = topsongs(train, data, top_length, verbosity_level)
    # Output the prediction to the given file
    pd.Series(pred).to_frame().to_csv(output_file, index = False)
    print "Prediction completed successfully."
    print "Output can be found in file {0}".format(output_file)
    

def topsongs_test(region, labels = True, top_length = 10, verbosity_level = 0):
    '''
    This is a testing function for the "Top Songs" prediction's main.
    '''
    pass