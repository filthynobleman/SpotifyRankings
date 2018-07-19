'''
This module contains a set of functions that can be used to predict the number of streams
of a certain song in a certain date.
Notice that, even if the module has been written to be used to perform the same prediction on
any dataset with a similar structure, it also has been written with in mind the "Spotify's top
200 songs" dataset from Kaggle.
The source of the dataset is:
https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking
'''

import pandas as pd
import numpy as np
import utils
import datasetinfo
import os, sys

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

STREAMSPRED_PREV_STREAMS = "Previous Streams"
'''
This string represents the name of the column containing the streams registered in the
last day when the song has obtained a position in the Spotify's top 200 for the same
country.
It cannot be supplied by the user.
'''

STREAMSPRED_FEATURES = [datasetinfo.TRACKNAME_COLUMN,
                        datasetinfo.ARTIST_COLUMN,
                        datasetinfo.DATE_COLUMN,
                        STREAMSPRED_PREV_STREAMS]
'''
This list represents the list of features used in the "Streams" prediction.
It cannot be supplied by the user.
'''

def add_previous_streams(data, verbosity_level = 0):
    '''
    This function adds to the dataset a new column containing the previous recorded number
    of streams. Specifically, in number of streams in the new column is referred to the
    number of streams registered in the last day when the song has achieved a position in
    the Spotify's top 200 in the same country.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    if verbosity_level > 0:
        print "Adding {} column...".format(STREAMSPRED_PREV_STREAMS)
    # Add the empty column
    data[STREAMSPRED_PREV_STREAMS] = 0
    # Get the list of countries
    countries = data[datasetinfo.REGION_COLUMN].unique()
    for country in countries:
        # Get the tracks for that country
        tracks = data[data[datasetinfo.REGION_COLUMN] == country][datasetinfo.TRACKNAME_COLUMN]
        tracks = tracks.unique()
        for track in tracks:
            subdata = data[data[datasetinfo.REGION_COLUMN] == country]
            subdata = subdata[subdata[datasetinfo.TRACKNAME_COLUMN] == track]
            # Shift the streams back of one day
            streams = subdata[datasetinfo.STREAMS_COLUMN]
            data.loc[subdata.index, STREAMSPRED_PREV_STREAMS] = streams.shift(1)
        if verbosity_level > 1:
            print "\rPrevious streams of tracks for country {} computed.".format(country)
    # Drop rows with NaN values. We cannot know the number of streams in the next day
    dropindex = data[data[STREAMSPRED_PREV_STREAMS].isna()].index
    data = data.drop(index = dropindex)
    return data


def initialize_dataset(dataset = None, regions = 'global', verbosity_level = 0):
    '''
    This function initializes the given songs' dataset. If no dataset is supplied, then the
    function initializes the default one.
    The initialization procedure consists in imputing the NaN values and encoding the
    categorical features.
    Notice that the initialization procedure also drops the URL and the Position columns and
    every row that does not belongs to one of the given regions. Furthermore, the Region
    column is dropped.
    If no region is supplied, then the global region is assumed.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    if verbosity_level > 0:
        print "Starting the initialization procedure..."
    # Read the dataset
    data = dataset
    if data is None:
        data = utils.read_dataset(verbosity_level)
    # Drop the URL and Position columns
    data = data.drop(columns = [datasetinfo.URL_COLUMN, datasetinfo.POSITION_COLUMN])
    # Compute the regions' list
    regs = None
    if type(regions) is str:
        regs = [regions]
    elif type(regions) in [list, set, tuple]:
        regs = list(regions)
    else:
        raise ValueError("Regions must be a string or an iterable object.")
    # Exclude rows not belonging to the given regions
    data = data[data[datasetinfo.REGION_COLUMN].isin(regs)]
    # If no data remains, then the region code are invalid or no informations about
    # the regions are available
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
    data = utils.impute_categorical_nans(data, inplace = True,
                                               verbosity_level = verbosity_level)
    # Add shifted streams
    data = add_previous_streams(data, verbosity_level)
    # Drop the region column
    data = data.drop(columns = [datasetinfo.REGION_COLUMN])
    if verbosity_level > 0:
        print "Successfully dropped Region column."
    # Label encodes all the categorical features
    data = utils.label_encode_columns(data, inplace = True, verbosity_level = verbosity_level)
    # Return the dataset
    if verbosity_level > 0:
        print "Initialization procedure has been completed succesffully."
    return data

def fit_classifier(training_set, verbosity_level = 0):
    '''
    This functions returns a Sci-Kit regressor trained with the given training set.
    The type and the parameters of the regressor are hard-coded in the function, according to
    which settings resulted to be the optimal ones. The features' list is given by the global
    variable STREAMSPRED_FEATURES.
    The function operates splitting the training set into features and target sets and then
    fitting the regressor. Once the fitting operation is completed, the regressor is
    returned.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    # Create the features and target sets
    global STREAMSPRED_FEATURES
    train_x = training_set[STREAMSPRED_FEATURES]
    train_y = training_set[datasetinfo.STREAMS_COLUMN]
    # Train the regressor
    if verbosity_level > 0:
        print "Training the regressor for \"Streams\" prediction..."
    clf = RandomForestRegressor()
    clf.fit(train_x, train_y)
    if verbosity_level > 0:
        print "The classifier has been trained successfully."
    return clf

def predict_data(regressor, data, verbosity_level = 0):
    '''
    This function applies the prediction of the given regressor on the given data and
    returns the computed predicted regression.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    if verbosity_level > 0:
        print "Applying the prediction..."
    pred = regressor.predict(data[STREAMSPRED_FEATURES])
    if verbosity_level > 0:
        print "Prediction completed successfully."
    return pred

def predict_test_data(regressor, test, verbosity_level = 0):
    '''
    This function applies the prediction of the given regressor to the given test set and
    returns the RMSE between the predicted values and the real ones.
    The test set is splitted by the function into features set and target set.
    The features' list is given by the global variable STREAMSPRED_FEATURES.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    # Split into features and target sets
    global STREAMSPRED_FEATURES
    test_x = test[STREAMSPRED_FEATURES]
    test_y = test[datasetinfo.STREAMS_COLUMN]
    # Apply the prediction
    pred = predict_data(regressor, test_y, verbosity_level)
    # Compute and return the RMSE
    return utils.compute_rmse(test_y, pred)

def streamsprediction(trainig_set, data, verbosity_level = 0):
    '''
    This function performs the "Streams" prediction.
    The prediction consists in estimating the number of streams of a certain song in a given
    date.
    The function trains a regressor using the given training_set and then computes the
    prediction on the given dataset.
    The results of the prediction are then returned by the function.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    # Train a classifier on the training set
    clf = fit_classifier(trainig_set, verbosity_level)
    # Apply and return the prediction
    return predict_data(clf, data, verbosity_level)

def streamsprediction_test():
    '''
    This is a testing function for the "Streams" prediction procedure.
    '''
    # Use the same dataset for all the tests. Simply write many times different
    # subsets of him
    main_data = pd.read_csv(os.path.join(utils.DATA_DIRECTORY, utils.DATASET_NAME))

    # Get the relevand part of the dataset
    regions = ['it']
    data = main_data[main_data['Region'].isin(regions)].copy()
    data = initialize_dataset(data, regions, verbosity_level = 2)
    # Split into train and test
    train, test = utils.split_dataset_sample(data)
    print "Starting computation..."
    pred = streamsprediction(train, test, 0) 
    real_y = test[datasetinfo.STREAMS_COLUMN]
    print "RMSE:      {}".format(utils.compute_rmse(real_y, pred))
    print "R squared: {}".format(utils.compute_r_squared(real_y, pred))