'''
This module contains the definition of the class Predictor.
This class is used as base class for the various predictors used in this project.
In the Predictor class there are the initialization of the object and some preprocessing
methods. The class also contains some interface for fitting, predicting and testing data.
'''

import pandas as pd
import utils
import datasetinfo as dsinfo
import preprocessing as pp
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from customexceptions import *


class Predictor(Object):
    '''
    This is the base class for every predictor in the project.
    The base class initializes the properties commons to every predictor, such as the
    dataset and the backend predictor to use.
    Furthermore, the base class provides some useful methods such as data filtering,
    encoding and train-test splitting.
    '''

    def set_predictor(self, backend_predictor):
        '''
        This method sets the backend predictor to the one corresponding to the given
        predictor type.
        The allowed predictors are:
            - 'dtree' for Decision Tree
            - 'randforest' for Random Forest
            - 'svc' for Support Vector 
            - 'adaboost' for AdaBoost
        A predictor not in this list is invalid and it will raise an InvalidTypeException.
        Notice that the predictor name is case insensitive.
        Remember to fit again the predictor, once the method is executed, before it
        reinitializes the predictor and every information about the previous fits are
        lost.
        '''
        if backend_predictor.lower() == 'dtree':
            if self.pred_task == 'classification':
                self.clf = DecisionTreeClassifier()
            else:
                self.clf = DecisionTreeRegressor()
        elif backend_predictor.lower() == 'randforest':
            if self.pred_task == 'classification':
                self.clf = RandomForestClassifier()
            else:
                self.clf = RandomForestRegressor()
        elif backend_predictor.lower() == 'svc':
            if self.pred_task == 'classification':
                self.clf = SVC()
            else:
                self.clf = SVR()
        elif backend_predictor.lower() == 'adaboost':
            if self.pred_task == 'classification':
                self.clf = AdaBoostClassifier()
            else:
                self.clf = AdaBoostRegressor()
        else:
            errstring = "The type {} is not allowed.".format(backend_predictor.lower())
            raise InvalidTypeException(errstring)
    
    def __init__(self, data = None, datafile = utils.DATASET_FILE,
                       backend_predictor = 'dtree', prediction_type = 'classification'):
        '''
        The initialization methods initializes the dataset, the training set and the test set.
        It also initializes the backend predictor to be used.
        Finally, the initialization method performs some preprocessing, dropping the useless
        columns and the rows with missing data.

        The backend predictor is set using the rules described in the method 'set_predictor()'.
        The dataset is initialized as follows:
            - If data is given, then the dataset is initialized as a copy of data
            - If data is None, then the dataset is initialized reading the file datafile
        Notice that the training set and the test set are initialized to None. They will be
        defined at the first call to 'initialize_train_test'.

        The input parameter prediction_type determines if the Predictor object is used for
        classification or regression tasks. The allowed values for the parameter are:
            - 'classification'
            - 'regression'

        The following exception are going to be raised:
            - TypeError: if data is not a pandas DataFrame
            - InvalidTypeException: if backend_predictor is not a valid one
            - InvalidTypeException: if prediction_type is not a valid one
            - UndefinedDatasetException: if data is None and datafile does not represents a
                                         valid filename for a file containing a dataset
        '''
        super(Predictor, self).__init__()
        # At first, initialize the prediction type and the classifier
        if prediction_type.lower() not in ['classification', 'regression']:
            errstring = "The prediction task {} is not allowed."
            errstring = errstring.format(prediction_type.lower())
            raise InvalidTypeException(errstring)
        self.pred_task = prediction_type.lower()
        self.clf = None
        self.set_predictor(backend_predictor)
        # Now, initialize the dataset
        self.dataset = None
        # If data is None, try to read a dataset
        if data is None:
            try:
                self.dataset = pd.read_csv(datafile)
            except:
                pass # TODO raise exception as defined in docstring
        # Otherwise, copy the given dataset
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("The dataset must be a pandas DataFrame.")
            self.dataset = data.copy()
        # Drop the URL column and the missing rows
        self.dataset.drop(columns=[dsinfo.URL_COLUMN], inplace=True)
        for col in set(self.dataset.columns):
            self.dataset = self.dataset[~self.dataset[col].isna()]
        # Initialize other useful properties such as training/test sets and their
        # encoded versions
        self.train = None
        self.test = None
        self.enc_train = None
        self.enc_test = None
        # Initialize the list of features and the label for testing
        # They should be defined by the subclasses
        self.features = None
        self.label = None

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
    
    def fit(self):
        '''
        This method fits the predictor associated with this object with the encoded training
        dataset.
        If the encoded training dataset is not updated to the current training set or it is
        None, then the method raises a EncodingNotUpToDateException.
        If the backend predictor is None, then the method raises an
        UndefinedPredictorException.
        '''
        if self.enc_train is None or self.enc_train.index != self.train.index:
            pass # TODO raise the proper exception as defined in documentation
        if self.clf is None:
            pass # TODO raise the proper exception as defined in documentation
        train_x = self.train[self.features]
        train_y = self.train[self.label]
        self.clf.fit(train_x, train_y)
    
    def predict(self, set_x):
        '''
        This method uses the predictor associated with this object to predict the given
        dataset. The results of the prediction are then returned.
        Notice that the given dataset must be encoded or the prediction will fail.
        '''
        return self.clf.predict(set_x[self.features])
    
    @abstractmethod
    def test_prediction(self):
        '''
        This method tests the quality of the prediction using the encoded test set associated
        with this object.
        In the base class, the methods returns the prediction, in order to allow to the sub
        classes to compute some statistics about the results.
        In the sub classes, the method should return nothing.

        If the encoded test set is not updated to the current test set or it is None, the the
        method raises a EncodingNotUpToDateException.
        '''
        if self.enc_test is None or self.enc_test.index != self.test.index:
            pass # TODO raise the proper exception as defined in documentation
        self.fit()
        return self.predict(self.enc_test[self.features])
    
    def encode_dataset(self, data):
        '''
        This method computes the encoding of the given dataset, according to the content of
        the training set.
        Notice that, once this method is called, if the predictor was previously fitted, its
        predictions will not be siginficant, because the encoding of the test set could be
        different.
        The encoding of the test set is saved in this object, while the encoding of the
        given dataset is returned.

        This method raises the following exceptions:
            - TypeError: if data is not a pandas DataFrame
            - NoneReferenceException: if training set has not yet been defined
        '''
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if self.train is None:
            raise NoneReferenceException("Training set is undefined.")
        # Compute the union of training set and test set
        union = self.train.append(data)
        pp.encode_dates(union, inplace=True)
        pp.encode_artists(union, inplace=True)
        pp.encode_tracks(union, inplace=True)
        # Save the encoded training set
        self.enc_train = union.loc[self.train.index,]
        # Update the feature set
        self.features = set(self.enc_train.columns)
        self.features -= set([dsinfo.POSITION_COLUMN])
        self.features -= set([dsinfo.REGION_COLUMN])
        self.features -= set([dsinfo.STREAMS_COLUMN])
        self.features = list(self.features)
        # Return the encoded dataset
        return union.loc[data.index,]
    
    def encode_train_and_test(self):
        '''
        This method encodes the traing set and the test set and save the encodings in this
        object.
        Notice that, once this method is called, if the predictor was previously fitted, its
        predictions will not be siginficant, because the encoding of the test set could be
        different.

        This method raises the following exceptions:
            - NoneReferenceException: if training set has not yet been defined
            - NoneReferenceException: if test set has not yet been defined
        '''
        if self.train is None:
            raise NoneReferenceException("Training set is undefined.")
        if self.test is None:
            raise NoneReferenceException("Test set is undefined.")
        # Save the encoding in the encoded test set
        self.enc_test = self.encode_dataset(self.test)

