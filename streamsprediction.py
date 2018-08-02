'''
This module contains a set of functions that can be used to predict the number of streams
of a certain song in a certain date.
Notice that, even if the module has been written to be used to perform the same prediction on
any dataset with a similar structure, it also has been written with in mind the "Spotify's top
200 songs" dataset from Kaggle.
The source of the dataset is:
https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking
'''

from predictor import *
import utils
import datasetinfo as dsinfo
import preprocessing as pp
import os, sys

class TodayStreams(Predictor):
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

    def __init__(self, data = None, datafile = utils.DATASET_FILE,
                       backend_predictor = 'dtree'):
        '''
        The initialization method creates the TodayStreams object.
        For a more detailed documentation, see the constructor of the superclass Predictor.

        The initialization method initializes the scores for evaluating the quality of the
        prediction.
        The available metrics are:
            - Root of the Mean Squared Error
            - R Squared
        Furthermore, the constructor initialzes the features' set and the label to defined
        values.
        '''
        super(TodayStreams, self).__init__(data, datafile, backend_predictor)
        # Initialize the RMSE and the R squared
        self.rmse = float('inf')
        self.r_squared = 0
        # Initialize features and label
        self.label = dsinfo.STREAMS_COLUMN
        self.base_features = [dsinfo.DATE_COLUMN, dsinfo.PREVSTREAMS_COLUMN]
    
    def add_previous_streams(self):
        '''
        This method adds to the dataset the 'Previous Streams' column.
        In this column are defined the number of streams achieved by the track in the last
        day where it has obtained a position in the Spotify's top 200 of the same country.
        '''
        # Initialize the column
        self.dataset[dsinfo.PREVSTREAMS_COLUMN] = 0
        # Get the list of tracks
        tracks = self.dataset[dsinfo.TRACKNAME_COLUMN].unique()
        # For each track, compute the Previous Streams values it has and add it to
        # the new column
        for track in tracks:
            local = self.dataset[self.dataset[dsinfo.TRACKNAME_COLUMN] == track]
            prevs = local[dsinfo.STREAMS_COLUMN].shift(1)
            self.dataset.loc[local.index, dsinfo.PREVSTREAMS_COLUMN] = prevs
        # Drop all the rows with missing values
        to_drop = self.dataset[self.dataset[dsinfo.PREVSTREAMS_COLUMN].isna()].index
        self.dataset.drop(index=to_drop, inplace=True)
    
    def test_prediction(self):
        pred = super(TodayStreams, self).test_prediction()
        y_true = self.enc_test[self.label]
        self.rmse = utils.compute_rmse(y_true, pred)
        self.r_squared = utils.compute_r_squared(y_true, pred)