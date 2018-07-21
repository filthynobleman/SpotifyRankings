'''
This module provides a class that can be used to predict if a certain song of a
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
from sklearn.avm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import utils
import datasetinfo as dsinfo
import os
from customexceptions import *

TOPSONGS_FEATURES = [dsinfo.ARTIST_COLUMN,
                     dsinfo.DATE_COLUMN,
                     dsinfo.TRACKNAME_COLUMN]
'''
This list represents the features list that is used in the "Top Songs" prediction.
It cannot be supplied by the user.
'''

class TopSongs(object):
    '''
    This class is used to perform the "Top Songs" prediction.
    The class uses a SciKit classifier to compute the prediction. Which classifier have to
    be used is tunable, according to which results are obtained during the tests.
    The class also provides a set of useful features to filter the dataset according to
    the needs.
    Because the class is used also for testing, it is possible to keep informations about
    a training set and a test set, and also to obtain the results of testing.
    It is possible to filter the dataset (and also the training and test set) in the
    following ways:
        - Date range
        - Artists' list
        - Tracks names' list
        - Regions' list
    Every parameter of the class can be changed in any moment after the creation.
    '''

    def set_classifier(self, classifier_type):
        '''
        This method sets the classifier used for the prediction to the one corresponding 
        to the given classifier type.
        The allowed classifiers are:
            - 'dtree' for DecisionTreeClassifier
            - 'randforest' for RandomForestClassifier
            - 'svc' for SVC (Support Vector Classifier)
            - 'adaboost' for AdaBoostClassifier
        A classifier not in this list is invalid and it will raise an InvalidTypeException.
        Notice that the classifier name is case insensitive.
        Remember to fit again the classifier, once the method is executed, before it
        reinitializes the classifier and every information about the previous fits are
        lost.
        '''
        if classifier_type.lower() == 'dtree':
            self.clf = DecisionTreeClassifier()
        elif classifier_type.lower() == 'randforest':
            self.clf = RandomForestClassifier()
        elif classifier_type.lower() == 'svc':
            self.clf = SVC()
        elif classifier_type.lower() == 'adaboost':
            self.clf = AdaBoostClassifier()
        else:
            errstring = "The type {} is not allowed.".format(classifier_type.lower())
            raise InvalidTypeException(errstring)

    def __init__(self, ds_name = utils.DATASET_FILE, top_length = 10,
                 classifier_type = 'dtree'):
        '''
        The initialization method creates the TopSongs object reading the dataset and
        dropping the useless features (URL and Streams).
        The length of the top list top predict is set to the input parameter top_length
        and the classifier is set to the SciKit classifier identified by the input
        parameter classifier_type.
        The allowed classifier type are defined in the description of the method
        'set_classifier()'.
        Default values for the input parameters are 10 for the length of the top list and
        'dtree' for DecisionTreeClassifier.
        Notice that the training and the test sets are initialized to None. The only way to
        set them is to call the methods 'initialize_train_test()'.
        '''
        super(TopSongs, self).__init__()
        # Setting the top length
        self.top_length = top_length
        # Setting the classifier
        self.clf = None
        self.set_classifier(classifier_type)
        # Reading the dataset
        self.dataset = pd.read_csv(ds_name)
        # Drop URL and Streams
        drop_columns = [dsinfo.URL_COLUMN, dsinfo.STREAMS_COLUMN]
        self.dataset.drop(columns = drop_columns, inplace = True)
        # Initialize train and test
        self.train = None
        self.test = None

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
        # Check if the given artist is a string or a collection
        ltracks = None
        if isinstance(tracks, str):
            ltracks = [tracks]
        else:
            for collecttype in [list, set, tuple]:
                if isinstance(tracks, collecttype):
                    ltracks = list(tracks)
        # Apply the filter
        ds = ds[ds[dsinfo.TRACKNAME_COLUMN].isin(ltracks)]




# def initialize_dataset(dataset = None, region = 'global', verbosity_level = 0):
#     '''
#     This function preprocess the given songs' dataset. If no dataset is given, then the
#     function reads a default dataset and applies the preprocessing procedure to this one.
#     The preprocessing procedure consists in imputing the NaN values for the categorical
#     features and encoding those lasts.
#     Notice that the preprocessing procedure also drops the URL and Streams columns and every
#     row that does not belongs to the given region, or to one of the given regions.
#     In no region is given, then the 'global' region is assumed.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     if verbosity_level > 0:
#         print "The initialization procedure has been started..."
#     # Read the dataset and drop the URL and Streams columns
#     if dataset is None:
#         data = utils.read_dataset(verbosity_level)
#     else:
#         data = dataset
#     data = data.drop(columns = [datasetinfo.URL_COLUMN, datasetinfo.STREAMS_COLUMN])
#     if verbosity_level > 0:
#         print "Successfully dropped URL and Streams columns."
#     # Drop every line that does not belongs to the given regions
#     regs = None
#     if type(region) is str:
#         regs = [region]
#     else:
#         regs = list(region)
#     data = data[data[datasetinfo.REGION_COLUMN].isin(regs)]
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
#     # The region is no more needed. Drop it
#     data = data.drop(columns = [datasetinfo.REGION_COLUMN])
#     if verbosity_level > 0:
#         print "Successfully dropped Region column."
#     # Impute missing values
#     data = utils.impute_categorical_nans(data, inplace = True,
#                                                verbosity_level = verbosity_level)
#     # Label encodes all the categorical features
#     data = utils.label_encode_columns(data, inplace = True, verbosity_level = verbosity_level)
#     # If some other kind of encoding is required for artist and track name, use it
#     # if not labels:
#     #     if verbosity_level > 1:
#     #         print "Label encoding does not fit well the following columns:"
#     #         print "    {0}".format(datasetinfo.ARTIST_COLUMN)
#     #         print "    {0}".format(datasetinfo.TRACKNAME_COLUMN)
#     #     elif verbosity_level > 0:
#     #         print "Some other encoding method is required for some features."
#     # Return the dataset
#     if verbosity_level > 0:
#         print "Initialization procedure has been completed succesffully."
#     return data

# def fit_classifier(training_set, top_length = 10, verbosity_level = 0):
#     '''
#     This function returns a Sci-Kit classifier trained with the given training set.
#     The type and the parameters of the classifier are hard-coded in the function, according
#     to which settings resulted to be the optimal ones. The features' list is given by the
#     global variable TOPSONGS_FEATURES.
#     The top_length parameter defines the length of the list of the top songs on which the
#     classifier is going to be trained. Default value is 10.
#     The function operates splitting the training set into features and target sets and then
#     fitting the classifier. Once the fitting operation is completed, the classifier is
#     returned.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     # Creates the features set and the target sets
#     global TOPSONGS_FEATURES
#     train_x = training_set[TOPSONGS_FEATURES]
#     train_y = training_set[datasetinfo.POSITION_COLUMN] <= top_length
#     # Create the classifier
#     if verbosity_level > 0:
#         print "Training the classifier for \"Top Songs\" prediction..."
#     clf = DecisionTreeClassifier(criterion = 'entropy')
#     clf.fit(train_x, train_y)
#     if verbosity_level > 0:
#         print "The classifier has been trained successfully."
#     if verbosity_level > 1:
#         print "The decision path of the trained decision tree over the training set is:"
#         print clf.decision_path(train_x)
#     return clf

# def predict_data(classifier, data, verbosity_level = 0):
#     '''
#     This function applies the prediction of the given classifier to the given data and
#     returns the computed predicted classification.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     if verbosity_level > 0:
#         print "Applying the prediction..."
#     pred = classifier.predict(data[TOPSONGS_FEATURES])
#     if verbosity_level > 0:
#         print "Prediction completed successfully."
#     return pred

# def predict_test_data(classifier, test, top_length = 10, verbosity_level = 0):
#     '''
#     This function applies the prediction of the given classifier to the given test set and
#     returns a 3-tuple containing the following informations:
#         - Accuracy score
#         - Precision score
#         - Recall score
#     The test set is splitted by the function into features set and target set.
#     The features' list is given by the global variable TOPSONGS_FEATURES.
#     The top_length parameter defines the length of the list of the top songs on which the
#     classifier is going to be trained. Default value is 10.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     # Split the test set into features and target
#     global TOPSONGS_FEATURES
#     test_x = test[TOPSONGS_FEATURES]
#     test_y = test[datasetinfo.POSITION_COLUMN] <= top_length
#     # Apply the prediction
#     pred = predict_data(classifier, test_x, verbosity_level)
#     # Compute the scores and return them
#     return utils.compute_classification_scores(test_y, pred)

# def topsongs(training_set, data, top_length = 10, verbosity_level = 0):
#     '''
#     This function performs the "Top Songs" prediction.
#     The prediction consists in finding if a cartain song of a certain artist will be, in a
#     given date, in the list of the top songs. The length of the list of the top songs is
#     defined by the top_length parameter, whose default value is 10.
#     The function trains a classifier using the given training_set and then computes the 
#     prediction on the given dataset.
#     The results of the prediction are then returned by the function.
#     The input parameter verbosity_level indicates which informations about the execution of
#     the procedure must be printed on the standard output. Default value is 0, that means 
#     nothing has to be printed.
#     '''
#     # Train a classifier over the training set
#     clf = fit_classifier(training_set, top_length, verbosity_level)
#     # Compute the prediction and return the result
#     return predict_data(clf, data, verbosity_level)
    

# def topsongs_test():
#     '''
#     This is a testing function for the "Top Songs" prediction procedure.
#     '''
#     # Use the same dataset for all the tests. Simply write many times different
#     # subsets of him
#     main_data = pd.read_csv(os.path.join(utils.DATA_DIRECTORY, utils.DATASET_NAME))


#     print "********** PREDICTION OF THE TOP 10 IN ITALY **********"
#     print "Description:"
#     print "The training set is a sampled 75% of the original dataset."
#     print "The test set is the remaining 25%."
#     print "Trying to predict if, knowing the song and its artist, the"
#     print "song will occupy a position in the Italian Spotify's top 10"
#     print "in a certain day."
#     print ""
#     # Get the relevand part of the dataset
#     data = main_data[main_data['Region'] == 'it'].copy()
#     data = initialize_dataset(data, 'it')
#     # Split into train and test
#     train, test = utils.split_dataset_sample(data)
#     pred = topsongs(train, test, 10, 0) 
#     real_y = test[datasetinfo.POSITION_COLUMN] <= 10
#     scores = utils.compute_classification_scores(real_y, pred)
#     print "Scores for this prediction:"
#     print "Accuracy:  {}".format(scores[0])
#     print "Precision: {}".format(scores[1])
#     print "Recall:    {}".format(scores[2])
#     print ""
#     print ""
#     print ""


#     print "********** PREDICTION OF NEXT WEEK TOP 25 IN IRELAND **********"
#     print "Description:"
#     print "The training set is the fraction of the dataset that goes"
#     print "from 2017-04-01 to 2017-04-30."
#     print "The test set is the fraction of the dataset that goes from"
#     print "2017-05-01 to 2017-05-07."
#     print "Trying to predict if, knowing the song and its artist, the"
#     print "song will occupy a position in the Irish Spotify's top 25"
#     print "in a certain day."
#     print ""
#     data = main_data[main_data['Region'] == 'ie'].copy()
#     data = data[(data['Date'] >= '2017-04-01') & (data['Date'] <= '2017-05-07')]
#     data = initialize_dataset(data, 'ie')
#     train = data[data['Date'] < 30]
#     test = data[data['Date'] >= 30]
#     pred = topsongs(train, test, 25, 0)
#     real_y = test[datasetinfo.POSITION_COLUMN] <= 25
#     scores = utils.compute_classification_scores(real_y, pred)
#     print "Scores for this prediction:"
#     print "Accuracy:  {}".format(scores[0])
#     print "Precision: {}".format(scores[1])
#     print "Recall:    {}".format(scores[2])
#     print ""
#     print ""
#     print ""


#     print "********** PREDICTION OF THE TOP 3 IN EUROPE **********"
#     print "Description:"
#     print "The training set is a sampled 75% of the original dataset."
#     print "The test set is the remaining 25%."
#     print "Trying to predict if, knowing the song and its artist, the"
#     print "song will occupy a position in the Euro Spotify's top 3 in"
#     print "a certain day."
#     print ""
#     # Get the relevand part of the dataset
#     regions = ['al', 'at', 'be', 'bg', 'ch', 'cz', 'de', 'dk', 'es',
#                'fi', 'fr', 'gb', 'gr', 'hr', 'hu', 'ie', 'is', 'it',
#                'lt', 'lu', 'lv', 'mc', 'md', 'mk', 'mt', 'nl', 'no',
#                'pl', 'ro', 'rs', 'se', 'si', 'sk', 'sm', 'ua']
#     data = main_data[main_data['Region'].isin(regions)].copy()
#     data = initialize_dataset(data, regions)
#     # Split into train and test
#     train, test = utils.split_dataset_sample(data)
#     pred = topsongs(train, test, 3, 0) 
#     real_y = test[datasetinfo.POSITION_COLUMN] <= 3
#     scores = utils.compute_classification_scores(real_y, pred)
#     print "Scores for this prediction:"
#     print "Accuracy:  {}".format(scores[0])
#     print "Precision: {}".format(scores[1])
#     print "Recall:    {}".format(scores[2])
#     print ""
#     print ""
#     print ""


#     print "********** PREDICTION OF NEXT WEEK TOP 5 IN USA AND CANADA **********"
#     print "Description:"
#     print "The training set is the fraction of the dataset that goes"
#     print "from 2017-07-01 to 2017-07-31."
#     print "The test set is the fraction of the dataset that goes from"
#     print "2017-08-01 to 2017-08-07."
#     print "Trying to predict if, knowing the song and its artist, the"
#     print "song will occupy a position in the Spotify's top 5 in the"
#     print "USA and Canada in a certain day."
#     print ""
#     regions = ['us', 'ca']
#     data = main_data[main_data['Region'].isin(regions)].copy()
#     data = data[(data['Date'] >= '2017-07-01') & (data['Date'] <= '2017-08-07')]
#     data = initialize_dataset(data, regions)
#     train = data[data['Date'] < 31]
#     test = data[data['Date'] >= 31]
#     pred = topsongs(train, test, 5, 0)
#     real_y = test[datasetinfo.POSITION_COLUMN] <= 5
#     scores = utils.compute_classification_scores(real_y, pred)
#     print "Scores for this prediction:"
#     print "Accuracy:  {}".format(scores[0])
#     print "Precision: {}".format(scores[1])
#     print "Recall:    {}".format(scores[2])
#     print ""
#     print ""
#     print ""


#     print "********** PREDICT AUGUST ITALIAN TOP 25 FROM JUNE-JULY SPANISH TOP 25 **********"
#     print "Description:"
#     print "The training set is the fraction of the dataset for the"
#     print "Spanish region from 2017-06-01 to 2017-07-31."
#     print "The test set is the fraction of the dataset for the Italian"
#     print "region from 2017-08-01 to 2017-08-31."
#     print "Trying to predict if, knowing the song and its artist, the"
#     print "song will occupy a position in the Italian Spotify's top 25"
#     print "in a certain day."
#     regions = ['it', 'es']
#     data = main_data[main_data['Region'].isin(regions)].copy()
#     data = data[(data['Date'] >= '2017-06-01') & (data['Date'] <= '2017-08-31')]
#     itindex = data.index[(data['Region'] == 'it') & (data['Date'] >= '2017-08-01')]
#     esindex = data.index[(data['Region'] == 'es') & (data['Date'] <= '2017-07-31')]
#     data = initialize_dataset(data, regions)
#     train = data.loc[esindex]
#     test = data.loc[itindex]
#     pred = topsongs(train, test, 25, 0)
#     real_y = test[datasetinfo.POSITION_COLUMN] <= 25
#     scores = utils.compute_classification_scores(real_y, pred)
#     print "Scores for this prediction:"
#     print "Accuracy:  {}".format(scores[0])
#     print "Precision: {}".format(scores[1])
#     print "Recall:    {}".format(scores[2])
#     print ""
#     print ""
#     print ""