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
import datasetinfo as dsinfo
import preprocessing as pp
import os, sys

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

class TodayStreams(object):
    '''
    This class is used to perform the "Today's Streams" prediction.
    The class uses a SciKit regressor to compute the prediction. Which regressor have to
    be used is tunable, according to which results are obtained during the tests.
    The class also provides a set of useful features to filter the dataset according to
    the needs.
    Because the class is used also for testing, it is possible to keep informations about
    a training set and a test set, and also to obtain the results of testing.
    It is possible to filter the dataset (and also the training and test set) in the
    following ways:
        - Streams range
        - Date range
        - Artists' list
        - Tracks names' list
        - Regions' list
    Every parameter of the class can be changed in any moment after the creation.
    '''

    def set_regressor(self, regressor_type):
        '''
        This method sets the regressor used for the prediction to the one corresponding 
        to the given regressor type.
        The allowed regressors are:
            - 'dtree' for DecisionTreeRegressor
            - 'randforest' for RandomForestRegressor
            - 'svr' for SVR (Support Vector Regressor)
            - 'adaboost' for AdaBoostRegressor
            - 'neural' for MLPRegressor
        A regressor not in this list is invalid and it will raise an InvalidTypeException.
        Notice that the regressor name is case insensitive.
        Remember to fit again the regressor, once the method is executed, before it
        reinitializes the regressor and every information about the previous fits are
        lost.
        '''
        if regressor_type.lower() == 'dtree':
            self.clf = DecisionTreeRegressor()
        elif regressor_type.lower() == 'randforest':
            self.clf = RandomForestRegressor()
        elif regressor_type.lower() == 'svr':
            self.clf = SVR()
        elif regressor_type.lower() == 'adaboost':
            self.clf = AdaBoostRegressor()
        elif regressor_type.lower() == 'neural':
            self.clf = MLPRegressor()
        else:
            errstring = "The type {} is not allowed.".format(regressor_type.lower())
            raise InvalidTypeException(errstring)

    def __init__(self, ds_name = utils.DATASET_FILE, regressor_type = 'dtree'):
        '''
        The initialization method creates the TodayStreams object reading the dataset and
        dropping the useless features (URL).
        The allowed regressor types are defined in the description of the method
        'set_regressor()'.
        Default values for regressor_type is 'dtree', that is DecisionTreeRegressor.
        Notice that the training and the test sets are initialized to None. The only way to
        set them is to call the methods 'initialize_train_test()'.
        The initialization method also initialize to None the target dataset to predict.
        Remember to set the prediction dataset before calling the 'fit_regressor()'
        method, because the label encoding for fitting is performed on the union of the
        training, test and prediction dataset
        '''
        super(TodayStreams, self).__init__()
        # Setting the regressor
        self.clf = None
        self.set_regressor(regressor_type)
        # Reading the dataset
        self.dataset = pd.read_csv(ds_name)
        # Transform streams to log1p
        #self.dataset[dsinfo.STREAMS_COLUMN] = np.log1p(self.dataset[dsinfo.STREAMS_COLUMN])
        # Drop URL
        drop_columns = [dsinfo.URL_COLUMN]
        self.dataset.drop(columns = drop_columns, inplace = True)
        # Drop all rows containing NaNs
        for col in set(self.dataset.columns):
            self.dataset = self.dataset[~self.dataset[col].isna()]
        # Initialize train, test and prediction
        self.train = None
        self.test = None
        self.prediction = None
        # Initialize the encoded version of the training, test and predition dataset
        self.enc_train = None
        self.enc_test = None
        self.enc_prediction = None
        # Initialize the RMSE and the R squared
        self.rmse = float('inf')
        self.r_squared = 0

    def filter_date(self, date_lte, date_gte, target = 'dataset'):
        '''
        This method applies a date range filter to the target dataset.
        The filter is applied dropping every row that it's not in the date range defined
        by the input parameters date_lte and date_gte.
        If the target dataset is 'train' or 'test' and the corresponding dataset is not
        yet initialized, then the method raises a NoneReferenceException.
        If the target is not one of 'dataset', 'target' or 'train', then the method raises
        a InvalidTypeException.
        '''
        # Get the reference to the right dataset
        ds = None
        if target.lower() == 'dataset':
            ds = self.dataset
        elif target.lower() == 'train':
            ds = self.train
        elif target.lower() == 'test':
            ds = self.test
        else:
            errstring = "The type {} is not allowed.".format(classifier_type.lower())
            raise InvalidTypeException(errstring)
        # Check if the dataset has been initialized
        if ds is None:
            ds_typename = "training" if target.lower() == 'train' else "test"
            errstring = "The {} set has not yet been initialized.".format(ds_typename)
            raise NoneReferenceException(errstring)
        # Apply the filter
        ds = ds[ds[dsinfo.DATE_COLUMN] <= date_gte]
        ds = ds[ds[dsinfo.DATE_COLUMN] >= date_lte]
        if target.lower() == 'dataset':
            self.dataset = ds
        elif target.lower() == 'train':
            self.train = ds
        elif target.lower() == 'test':
            self.test = ds

    def filter_streams(self, streams_lte, streams_gte, target = 'dataset'):
        '''
        This method applies a streams range filter to the target dataset.
        The filter is applied dropping every row that it's not in the streams range defined
        by the input parameters streams_lte and streams_gte.
        If the target dataset is 'train' or 'test' and the corresponding dataset is not
        yet initialized, then the method raises a NoneReferenceException.
        If the target is not one of 'dataset', 'target' or 'train', then the method raises
        a InvalidTypeException.
        '''
        # Get the reference to the right dataset
        ds = None
        if target.lower() == 'dataset':
            ds = self.dataset
        elif target.lower() == 'train':
            ds = self.train
        elif target.lower() == 'test':
            ds = self.test
        else:
            errstring = "The type {} is not allowed.".format(classifier_type.lower())
            raise InvalidTypeException(errstring)
        # Check if the dataset has been initialized
        if ds is None:
            ds_typename = "training" if target.lower() == 'train' else "test"
            errstring = "The {} set has not yet been initialized.".format(ds_typename)
            raise NoneReferenceException(errstring)
        # Apply the filter
        ds = ds[ds[dsinfo.STREAMS_COLUMN] <= streams_gte]
        ds = ds[ds[dsinfo.STREAMS_COLUMN] >= streams_lte]
        if target.lower() == 'dataset':
            self.dataset = ds
        elif target.lower() == 'train':
            self.train = ds
        elif target.lower() == 'test':
            self.test = ds
    
    def filter_artist(self, artists, target = 'dataset'):
        '''
        This applies an artist filter to the target dataset.
        The filter is applied dropping every row whose the artist is not in the given
        list of artists.
        If the target dataset is 'train' or 'test' and the corresponding dataset is not
        yet initialized, then the method raises a NoneReferenceException.
        If the target is not one of 'dataset', 'target' or 'train', then the method raises
        a InvalidTypeException.
        '''
        # Get the reference to the right dataset
        ds = None
        if target.lower() == 'dataset':
            ds = self.dataset
        elif target.lower() == 'train':
            ds = self.train
        elif target.lower() == 'test':
            ds = self.test
        else:
            errstring = "The type {} is not allowed.".format(classifier_type.lower())
            raise InvalidTypeException(errstring)
        # Check if the dataset has been initialized
        if ds is None:
            ds_typename = "training" if target.lower() == 'train' else "test"
            errstring = "The {} set has not yet been initialized.".format(ds_typename)
            raise NoneReferenceException(errstring)
        # Check if the given artist is a string or a collection
        lartists = None
        if isinstance(artists, str):
            lartists = [artists]
        else:
            for collecttype in [list, set, tuple]:
                if isinstance(artists, collecttype):
                    lartists = list(artists)
        # Apply the filter
        ds = ds[ds[dsinfo.ARTIST_COLUMN].isin(lartists)]
        if target.lower() == 'dataset':
            self.dataset = ds
        elif target.lower() == 'train':
            self.train = ds
        elif target.lower() == 'test':
            self.test = ds
    
    def filter_track(self, tracks, target = 'dataset'):
        '''
        This applies a tracks filter to the target dataset.
        The filter is applied dropping every row whose the track name is not in the given
        list of track names.
        If the target dataset is 'train' or 'test' and the corresponding dataset is not
        yet initialized, then the method raises a NoneReferenceException.
        If the target is not one of 'dataset', 'target' or 'train', then the method raises
        a InvalidTypeException.
        '''
        # Get the reference to the right dataset
        ds = None
        if target.lower() == 'dataset':
            ds = self.dataset
        elif target.lower() == 'train':
            ds = self.train
        elif target.lower() == 'test':
            ds = self.test
        else:
            errstring = "The type {} is not allowed.".format(classifier_type.lower())
            raise InvalidTypeException(errstring)
        # Check if the dataset has been initialized
        if ds is None:
            ds_typename = "training" if target.lower() == 'train' else "test"
            errstring = "The {} set has not yet been initialized.".format(ds_typename)
            raise NoneReferenceException(errstring)
        # Check if the given track name is a string or a collection
        ltracks = None
        if isinstance(tracks, str):
            ltracks = [tracks]
        else:
            for collecttype in [list, set, tuple]:
                if isinstance(tracks, collecttype):
                    ltracks = list(tracks)
        # Apply the filter
        ds = ds[ds[dsinfo.TRACKNAME_COLUMN].isin(ltracks)]
        if target.lower() == 'dataset':
            self.dataset = ds
        elif target.lower() == 'train':
            self.train = ds
        elif target.lower() == 'test':
            self.test = ds
    
    def filter_region(self, regions, target = 'dataset'):
        '''
        This applies a tracks filter to the target dataset.
        The filter is applied dropping every row whose the region is not in the given
        list of region codes.
        If the target dataset is 'train' or 'test' and the corresponding dataset is not
        yet initialized, then the method raises a NoneReferenceException.
        If the target is not one of 'dataset', 'target' or 'train', then the method raises
        a InvalidTypeException.
        '''
        # Get the reference to the right dataset
        ds = None
        if target.lower() == 'dataset':
            ds = self.dataset
        elif target.lower() == 'train':
            ds = self.train
        elif target.lower() == 'test':
            ds = self.test
        else:
            errstring = "The type {} is not allowed.".format(classifier_type.lower())
            raise InvalidTypeException(errstring)
        # Check if the dataset has been initialized
        if ds is None:
            ds_typename = "training" if target.lower() == 'train' else "test"
            errstring = "The {} set has not yet been initialized.".format(ds_typename)
            raise NoneReferenceException(errstring)
        # Check if the given region is a string or a collection
        lregions = None
        if isinstance(regions, str):
            lregions = [regions]
        else:
            for collecttype in [list, set, tuple]:
                if isinstance(regions, collecttype):
                    lregions = list(regions)
        # Apply the filter
        ds = ds[ds[dsinfo.REGION_COLUMN].isin(lregions)]
        if target.lower() == 'dataset':
            self.dataset = ds
        elif target.lower() == 'train':
            self.train = ds
        elif target.lower() == 'test':
            self.test = ds
    
    def initialize_train_test(self, train_ratio = 0.75, sample = True, full = False):
        '''
        This function initializes the training and the test sets.
        The training set is initialized to a fraction of the original dataset given by
        the input parameter train_ratio, while the test set will be initialized to the
        remaining part of the original dataset. Notice that the training and the test
        sets are copies of the original dataset, so every changes made on those datasets
        will not affect the original one. The default value for the parameter train_ratio
        is 0.75.
        If the input parameter sample is True, then the training set is built sampling
        rows from the original dataset. Otherwise, it is built taking the first part of
        the dataset. Default value for sample is True.
        If the input parameter full is True, then the values of the other parameters are
        ignored and both the training and the test sets are initialized to be exact
        copies of the full original one.
        '''
        if full:
            self.train  = self.dataset.copy()
            self.test   = self.dataset.copy()
        else:
            if sample:
                self.train, self.test = utils.split_dataset_sample(self.dataset, train_ratio)
            else:
                self.train, self.test = utils.split_dataset(self.dataset, train_ratio)

    def set_prediction_set(self, prediction_set):
        '''
        This method sets the prediction set to the given one.
        The type of the given prediction set must be convertible to a pandas DataFrame,
        or this method will raise an InvalidTypeException.
        '''
        try:
            self.prediction = pd.DataFrame(prediction_set)
        except:
            errstring = "Prediction set must be convertible to a pandas DataFrame."
            raise InvalidTypeException(errstring)
        # Change the indexes in order to keep them out of the range of the training
        # and test sets
        maxindex = np.max([np.max(self.train.index), np.max(self.test.index)])
        self.prediction.index = self.prediction.index + maxindex + 1

    def fit_regressor(self):
        '''
        This method fits the regressor used for prediction.
        The fitting procedure consists in the encoding of the columns of a dataset that is
        formed by the union of the training, the test and the prediction sets. Then, the
        regressor associated to this object is fitted using the training set.
        Notice that, in order to maintain consistency between the fitting and the
        prediction procedures, the elements of the columns are ordered, before the encoding
        procedure, according to their natural ordering.
        If the regressor associated with this object is None, then this method raises an
        UndefinedRegressorException.
        If the training set is None, then this method raises a NoneReferenceException.
        '''
        # Check if regressor is None
        if self.clf is None:
            raise UndefinedRegressorException("Cannot fit a None regressor.")
        # Check if training set is defined
        if self.train is None:
            raise NoneReferenceException("The training set is None.")
        # Get the union of the datasets
        union = pd.DataFrame(self.train)
        if self.test is not None:
            union = union.append(self.test)
        if self.prediction is not None:
            union = union.append(self.prediction)
        # Get the streams of the previous useful day
        union[dsinfo.PREVSTREAMS_COLUMN] = 0
        for track in union[dsinfo.TRACKNAME_COLUMN].unique():
            local = union[union[dsinfo.TRACKNAME_COLUMN] == track]
            prevs = local[dsinfo.STREAMS_COLUMN].shift(1)
            union.loc[local.index, dsinfo.PREVSTREAMS_COLUMN] = prevs
        # Drop NaNs
        to_drop = union[union[dsinfo.PREVSTREAMS_COLUMN].isna()].index
        union = union.drop(index = to_drop)
        pp.encode_dates(union, True)
        pp.encode_artists(union, True)
        pp.encode_tracks(union, True)
        # The training set is the subset of union whose the indices are those contained
        # in self.train
        # The same holds for test and prediction, if they are not None
        self.enc_train = union[union.index.isin(self.train.index)]
        if self.test is not None:
            self.enc_test = union[union.index.isin(self.test.index)]
        if self.prediction is not None:
            self.enc_prediction = union[union.index.isin(self.prediction.index)]
        # Now, fit the regressor
        features = list(set(self.enc_train.columns) - set([dsinfo.POSITION_COLUMN]))
        features = list(set(features) - set([dsinfo.REGION_COLUMN]))
        features = list(set(features) - set([dsinfo.STREAMS_COLUMN]))
        enc_train_x = self.enc_train[features]
        enc_train_y = self.enc_train[dsinfo.STREAMS_COLUMN]
        self.clf.fit(enc_train_x, enc_train_y)

    def compute_prediction(self, target = 'prediction'):
        '''
        This method computes the prediction for the target dataset.
        The allowed target types are:
            - 'prediction' for the target prediction dataset
            - 'test' for the test set
        Default is 'prediction'.
        The computed prediction is then returned.
        '''
        # Get the target dataset
        ds = None
        if target.lower() == 'prediction':
            ds = self.enc_prediction
        elif target.lower() == 'test':
            ds = self.enc_test
        else:
            raise InvalidTypeException("{} is not a valid type.".format(target.lower()))
        # If target is None, raise exception
        if ds is None:
            raise NoneReferenceException("Cannot predit a None dataset.")
        # Get the features and the target subsets
        features = list(set(ds.columns) - set([dsinfo.POSITION_COLUMN]))
        features = list(set(features) - set([dsinfo.REGION_COLUMN]))
        features = list(set(features) - set([dsinfo.STREAMS_COLUMN]))
        X = ds[features]
        # Compute the prediction and return it
        # return np.expm1(self.clf.predict(X))
        return self.clf.predict(X)
    
    def test_regressor(self):
        '''
        This method tests the prediction computed by the regressor on the training set.
        The test is computed applying the prediction to the test set and then computing
        the RMSE and the R squared.
        '''
        # Get the test Y
        # test_y = np.expm1(self.enc_test[dsinfo.STREAMS_COLUMN])
        test_y = self.enc_test[dsinfo.STREAMS_COLUMN]
        # Compute the test prediction
        pred = self.compute_prediction('test')
        # Compute the scores
        self.rmse = utils.compute_rmse(test_y, pred)
        self.r_squared = utils.compute_r_squared(test_y, pred)


def test_todaystreams():
    '''
    This function is used to test streams prediction.
    '''
    print "Computing streams prediction in Italy."
    ts = TodayStreams()
    print "Filtering region..."
    ts.filter_region('it')
    res = dict()
    res['dtree-rmses'] = list()
    res['dtree-r2'] = list()
    res['randfor-rmses'] = list()
    res['randfor-r2'] = list()
    res['neural-rmses'] = list()
    res['neural-r2'] = list()
    for train_ratio in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        print "Initializing training and test sets with ratio {}...".format(train_ratio)
        ts.set_regressor('dtree')
        ts.initialize_train_test(train_ratio=train_ratio)
        ts.fit_regressor()
        ts.test_regressor()
        print "Decision tree:"
        print "    RMSE:       {}".format(ts.rmse)
        res['dtree-rmses'].append(ts.rmse)
        print "    R squared:  {}".format(ts.r_squared)
        res['dtree-r2'].append(ts.r_squared)
        ts.set_regressor('randforest')
        ts.fit_regressor()
        ts.test_regressor()
        print "Random forest:"
        print "    RMSE:       {}".format(ts.rmse)
        res['randfor-rmses'].append(ts.rmse)
        print "    R squared:  {}".format(ts.r_squared)
        res['randfor-r2'].append(ts.r_squared)
        ts.set_regressor('neural')
        ts.fit_regressor()
        ts.test_regressor()
        print "Neural Network:"
        print "    RMSE:       {}".format(ts.rmse)
        res['neural-rmses'].append(ts.rmse)
        print "    R squared:  {}".format(ts.r_squared)
        res['neural-r2'].append(ts.r_squared)
    import json
    with open("it-streams.json", "w") as fp:
        json.dump(res, fp, indent=4, sort_keys=True)


test_todaystreams()

# def add_previous_streams(data, verbosity_level = 0):
#     '''
#     This function adds to the dataset a new column containing the previous recorded number
#     of streams. Specifically, in number of streams in the new column is referred to the
#     number of streams registered in the last day when the song has achieved a position in
#     the Spotify's top 200 in the same country.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     if verbosity_level > 0:
#         print "Adding {} column...".format(STREAMSPRED_PREV_STREAMS)
#     # Add the empty column
#     data[STREAMSPRED_PREV_STREAMS] = 0
#     # Get the list of countries
#     countries = data[datasetinfo.REGION_COLUMN].unique()
#     for country in countries:
#         # Get the tracks for that country
#         tracks = data[data[datasetinfo.REGION_COLUMN] == country][datasetinfo.TRACKNAME_COLUMN]
#         tracks = tracks.unique()
#         for track in tracks:
#             subdata = data[data[datasetinfo.REGION_COLUMN] == country]
#             subdata = subdata[subdata[datasetinfo.TRACKNAME_COLUMN] == track]
#             # Shift the streams back of one day
#             streams = subdata[datasetinfo.STREAMS_COLUMN]
#             data.loc[subdata.index, STREAMSPRED_PREV_STREAMS] = streams.shift(1)
#         if verbosity_level > 1:
#             print "\rPrevious streams of tracks for country {} computed.".format(country)
#     # Drop rows with NaN values. We cannot know the number of streams in the next day
#     dropindex = data[data[STREAMSPRED_PREV_STREAMS].isna()].index
#     data = data.drop(index = dropindex)
#     return data


# def initialize_dataset(dataset = None, regions = 'global', verbosity_level = 0):
#     '''
#     This function initializes the given songs' dataset. If no dataset is supplied, then the
#     function initializes the default one.
#     The initialization procedure consists in imputing the NaN values and encoding the
#     categorical features.
#     Notice that the initialization procedure also drops the URL and the Position columns and
#     every row that does not belongs to one of the given regions. Furthermore, the Region
#     column is dropped.
#     If no region is supplied, then the global region is assumed.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     if verbosity_level > 0:
#         print "Starting the initialization procedure..."
#     # Read the dataset
#     data = dataset
#     if data is None:
#         data = utils.read_dataset(verbosity_level)
#     # Drop the URL and Position columns
#     data = data.drop(columns = [datasetinfo.URL_COLUMN, datasetinfo.POSITION_COLUMN])
#     # Compute the regions' list
#     regs = None
#     if type(regions) is str:
#         regs = [regions]
#     elif type(regions) in [list, set, tuple]:
#         regs = list(regions)
#     else:
#         raise ValueError("Regions must be a string or an iterable object.")
#     # Exclude rows not belonging to the given regions
#     data = data[data[datasetinfo.REGION_COLUMN].isin(regs)]
#     # If no data remains, then the region code are invalid or no informations about
#     # the regions are available
#     if len(data) == 0:
#         errstr = "There's nothing in the dataset for the following regions:{0}"
#         errstr = errstr.format(os.linesep)
#         for reg in regs:
#             errstr += "    {0}".format(reg)
#         raise IndexError(errstr)
#     if verbosity_level > 1:
#         print "Successfully dropped rows not belonging to the following regions:"
#         for reg in regs:
#             print "    {0}".format(reg)
#     elif verbosity_level > 0:
#         print "Successfully dropped rows not belonging to interesting regions."
#     data = utils.impute_categorical_nans(data, inplace = True,
#                                                verbosity_level = verbosity_level)
#     # Add shifted streams
#     data = add_previous_streams(data, verbosity_level)
#     # Drop the region column
#     data = data.drop(columns = [datasetinfo.REGION_COLUMN])
#     if verbosity_level > 0:
#         print "Successfully dropped Region column."
#     # Label encodes all the categorical features
#     data = utils.label_encode_columns(data, inplace = True, verbosity_level = verbosity_level)
#     # Return the dataset
#     if verbosity_level > 0:
#         print "Initialization procedure has been completed succesffully."
#     return data

# def fit_classifier(training_set, verbosity_level = 0):
#     '''
#     This functions returns a Sci-Kit regressor trained with the given training set.
#     The type and the parameters of the regressor are hard-coded in the function, according to
#     which settings resulted to be the optimal ones. The features' list is given by the global
#     variable STREAMSPRED_FEATURES.
#     The function operates splitting the training set into features and target sets and then
#     fitting the regressor. Once the fitting operation is completed, the regressor is
#     returned.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     # Create the features and target sets
#     global STREAMSPRED_FEATURES
#     train_x = training_set[STREAMSPRED_FEATURES]
#     train_y = training_set[datasetinfo.STREAMS_COLUMN]
#     # Train the regressor
#     if verbosity_level > 0:
#         print "Training the regressor for \"Streams\" prediction..."
#     clf = RandomForestRegressor()
#     clf.fit(train_x, train_y)
#     if verbosity_level > 0:
#         print "The classifier has been trained successfully."
#     return clf

# def predict_data(regressor, data, verbosity_level = 0):
#     '''
#     This function applies the prediction of the given regressor on the given data and
#     returns the computed predicted regression.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     if verbosity_level > 0:
#         print "Applying the prediction..."
#     pred = regressor.predict(data[STREAMSPRED_FEATURES])
#     if verbosity_level > 0:
#         print "Prediction completed successfully."
#     return pred

# def predict_test_data(regressor, test, verbosity_level = 0):
#     '''
#     This function applies the prediction of the given regressor to the given test set and
#     returns the RMSE between the predicted values and the real ones.
#     The test set is splitted by the function into features set and target set.
#     The features' list is given by the global variable STREAMSPRED_FEATURES.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     # Split into features and target sets
#     global STREAMSPRED_FEATURES
#     test_x = test[STREAMSPRED_FEATURES]
#     test_y = test[datasetinfo.STREAMS_COLUMN]
#     # Apply the prediction
#     pred = predict_data(regressor, test_y, verbosity_level)
#     # Compute and return the RMSE
#     return utils.compute_rmse(test_y, pred)

# def streamsprediction(trainig_set, data, verbosity_level = 0):
#     '''
#     This function performs the "Streams" prediction.
#     The prediction consists in estimating the number of streams of a certain song in a given
#     date.
#     The function trains a regressor using the given training_set and then computes the
#     prediction on the given dataset.
#     The results of the prediction are then returned by the function.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     # Train a classifier on the training set
#     clf = fit_classifier(trainig_set, verbosity_level)
#     # Apply and return the prediction
#     return predict_data(clf, data, verbosity_level)

# def streamsprediction_test():
#     '''
#     This is a testing function for the "Streams" prediction procedure.
#     '''
#     # Use the same dataset for all the tests. Simply write many times different
#     # subsets of him
#     main_data = pd.read_csv(os.path.join(utils.DATA_DIRECTORY, utils.DATASET_NAME))

#     # Get the relevand part of the dataset
#     regions = ['it']
#     data = main_data[main_data['Region'].isin(regions)].copy()
#     data = initialize_dataset(data, regions, verbosity_level = 2)
#     # Split into train and test
#     train, test = utils.split_dataset_sample(data)
#     print "Starting computation..."
#     pred = streamsprediction(train, test, 0) 
#     real_y = test[datasetinfo.STREAMS_COLUMN]
#     print "RMSE:      {}".format(utils.compute_rmse(real_y, pred))
#     print "R squared: {}".format(utils.compute_r_squared(real_y, pred))