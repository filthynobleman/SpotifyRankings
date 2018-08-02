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
from predictor import *
from sklearn.metrics import accuracy_score, precision_score, recall_score
import utils
import datasetinfo as dsinfo
import preprocessing as pp
import os
from customexceptions import *

class TopSongs(Predictor):
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
        - Streams range
        - Date range
        - Artists' list
        - Tracks names' list
        - Regions' list
    Every parameter of the class can be changed in any moment after the creation.
    '''

    def __init__(self, data = None, datafile = utils.DATASET_FILE, top_length = 10,
                       backend_predictor = 'dtree'):
        '''
        The initialization method creates the TopSongs object.
        For a more detailed documentation, see the constructor of the superclass Predictor.

        The initialization method saves the length of the top and initializes the scores
        for evaluating the quality of the prediction.
        The available metrics are:
            - Accuracy
            - Precision
            - Recall
        Furthermore, the constructor initialzes the features' set and the label to defined
        values, adding to the dataset the new target column 'Is In Top'.
        '''
        super(TopSongs, self).__init__(data, datafile, backend_predictor, 'classification')
        # Setting the top length
        self.top_length = top_length
        # Initialize accuracy, precision and recall
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        # Initialize features and label
        self.label = dsinfo.ISINTOP_COLUMN
        self.dataset[self.label] = self.dataset[dsinfo.POSITION_COLUMN] <= self.top_length
        self.features = set(self.dataset.columns)
        self.features -= set([dsinfo.POSITION_COLUMN])
        self.features -= set([dsinfo.REGION_COLUMN])
        self.features -= set([dsinfo.STREAMS_COLUMN])
        self.features -= set([dsinfo.ISINTOP_COLUMN])
        self.features = list(self.features)
    
    def set_top_length(self, top_length):
        '''
        This method safely update the top length associated to this object.
        The method recomputes the target column using the new value for the top length.
        The method also updates the target column in training and test set, without changing
        their content. Same is for their encoded version.
        Notice that a new fitting is required, because previous one could be no more
        significant, if top length changes.
        '''
        self.top_length = top_length
        self.dataset[self.label] = self.dataset[dsinfo.POSITION_COLUMN] <= self.top_length
        if self.train is not None:
            self.train[self.label] = self.train[dsinfo.POSITION_COLUMN] <= self.top_length
        if self.test is not None:
            self.test[self.label] = self.test[dsinfo.POSITION_COLUMN] <= self.top_length
        if self.enc_train is not None:
            self.enc_train[self.label] = self.enc_train[dsinfo.POSITION_COLUMN] <= self.top_length
        if self.enc_test is not None:
            self.enctest[self.label] = self.enc_test[dsinfo.POSITION_COLUMN] <= self.top_length

    def test_prediction(self):
        '''
        For a more complete documentation, read the docstring of this method in the 
        superclass Predictor.

        This method computes the accuracy score, the precision score and the recall score
        of the prediction computed on the test set.
        '''
        pred = super(TopSongs, self).test_prediction()
        y_true = self.enc_test[self.label]
        self.accuracy = accuracy_score(y_true, pred)
        self.precision = precision_score(y_true, pred)
        self.recall = recall_score(y_true, pred)