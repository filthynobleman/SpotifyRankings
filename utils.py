'''
This module provides some useful functions that can be used to preprocess, handle or
manage the data.
Notice that, even if some functions can be used in general with any dataset, they have been
written with in mind the "Spotify's top 200 songs" dataset from Kaggle.
The dataset's source is:
https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking
'''

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATA_DIRECTORY = "."
'''
This string represents the relative path to the directory containing the dataset.
It can be supplied by the user. Default value is current working directory.
'''
DATASET_NAME = "data.csv"
'''
This string represents the base name of the filename containing the dataset.
It can be supplied by the user. Default value is "data.csv".
'''

def read_dataset(verbosity_level = 0):
    '''
    This function automatically reads the dataset and returns it as a pandas DataFrame.
    The file is readed by the directory identified by the global variable DATA_DIRECTORY and
    its base name is identified by the global variable DATASET_NAME.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    global DATA_DIRECTORY
    global DATASET_NAME
    filename = os.path.join(DATA_DIRECTORY, DATASET_NAME)
    if verbosity_level > 1:
        print "Reading dataset from file {0}...".format(filename)
    return pd.read_csv(filename)
    if verbosity_level > 0:
        print "Dataset successfully readed from file {0}.".format(filename)

def split_dataset(dataset, frac = 0.75):
    '''
    This function splits the dataset into two parts that, if concatenated, will result back
    in the original dataset.
    The function returns a 2-tuple containing the following objects in this order:
        - The first m rows of the dataset, where m = floor(frac * n), n is the length of the
          whole dataset and frac is an input parameter for this function.
        - The remaining rows of the original dataset.
    The default value for frac is 0.75.
    '''
    frac_len = int(np.floor(len(ds) * frac))
    return (dataset[:frac_len], dataset[frac_len:])

def split_dataset_sample(dataset, frac = 0.75):
    '''
    This function splits the dataset into two parts that, if contatenated, will result in a
    dataset containing the same rows of the original dataset, eventually in a different order.
    The function returns a 2-tuple containing the following objects in this order:
        - m rows from the original dataset, where m = floor(frac * n), n is the length of the
          whole dataset and frac is an input parameter for this function.
        - The other rows of the original dataset.
    The default value for frac is 0.75.
    '''
    first = dataset.sample(frac = frac)
    second = pd.concat([dataset, first]).drop_duplicates(keep = False)
    return (first, second)

def label_encode_columns(dataset, columns = None, inplace = False, verbosity_level = 0):
    '''
    This method applies the label encoding method to the given columns of the given dataset.
    If the input parameter columns is not supplied or it is None, then the procedure is
    applied to every column in the dataset whose type is 'object'.
    If the input parameter inplace is True, then the given dataset is directly modified and
    then returned. Otherwise, a deep copy of the dataset is made and the modifications are
    applied only to that copy, which is then returned.
    Notice that label encoding provides the values some kind of importance, giving them an
    order that they may don't have implicitly. This function encodes the values using their
    natural order, so, as an example, the label encoding of two strings follows the natural
    lexicographic ordering and the label encoding of two dates follows the natural time
    ordering.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    ds = dataset
    # Check if in place modification is required
    if not inplace:
        if verbosity_level > 1:
            print "The label encoding procedure has been executed without in place option."
        ds = dataset.copy()
        if verbosity_level > 1:
            print "The dataset has been succesffully copied."
    else:
        if verbosity_level > 1:
            print "The label encoding procedure has been executed with in place option."
    # Get the columns
    _columns = None
    if columns is None:
        if verbosity_level > 1:
            string = "No column has been supplied to the label encoding procedure. "
            string += "Searching for categorical columns in the dataset..."
        _columns = list()
        for col in set(ds.columns):
            if ds[col].dtype == 'object':
                _columns.append(col)
    elif type(columns) is str:
        _columns = [columns]
    else:
        _columns = list(columns)
    # For each column, apply the label encoding
    for col in _columns:
        le = LabelEncoder()
        # Fit the encoder with the unique values in the dataset.
        # Assuming order is important, the values are sorted
        values = ds[col].unique()
        values.sort()
        le.fit(values)
        # Replace the column with the encoded values
        ds[col] = le.transform(ds[col])
        if verbosity_level > 0:
            print "Column {0} has been succesffully encoded.".format(col)
    return ds

def impute_categorical_nans(dataset, columns = None, value = '', inplace = False, 
                            verbosity_level = 0):
    '''
    This function imputes NaN values for cathegorical features in the given dataset.
    If the input parameter columns is None, then any column of the given dataset whose
    type is 'object' or 'str' is analyzed. Otherwise, the parameter specifies which
    columns must be analyzed.
    The input parameter value defines which is the value that must be used to replace
    a NaN, if found. Default is the empty string.
    The input parameter inplace defines if the modifications to the dataset must be
    applied to the given dataset itself or to a copy of it. If its value is True, then
    the modifications are applied to the dataset, otherwise to a deep copy of it.
    The input parameter verbosity_level indicates which informations about the execution of
    the procedure must be printed on the standard output. Default value is 0, that means 
    nothing has to be printed.
    '''
    ds = dataset
    # Check if in place modification is required
    if not inplace:
        if verbosity_level > 1:
            string = "The categorical NaNs imputation procedure has been executed without"
            string += "in place option."
            print string
        ds = dataset.copy()
        if verbosity_level > 1:
            print "The dataset has been succesffully copied."
    else:
        if verbosity_level > 1:
            string = "The categorical NaNs imputation procedure has been executed with"
            string += "in place option."
            print string
    # Build the list of columns to analyze
    cols = None
    if columns is None:
        if verbosity_level > 1:
            string = "No column has been supplied to the categorical NaNs imputation procedure"
            string += ". Searching for categorical columns in the dataset..."
        cols = list()
        for col in set(ds.columns):
            if ds[col].dtype == 'object':
                cols.append(col)
    elif type(columns) is str:
        cols = [col]
    else:
        cols = list(columns)
    # For each column, search for NaNs and replace them with the given value
    for col in cols:
        nans = ds[col].isna().sum()
        ds.loc[ds[col].isna(), col] = value
        if verbosity_level > 0:
            print "Replaced {} NaN{} in column {}".format(nans, "s" if nans != 1 else "", col)
    # Return the dataset
    return ds


def compute_classification_scores(y_true, pred):
    '''
    This function computes the main scores for the given prediction pred with respect to
    the real values y_true.
    The function returns a 3-tuple containing the following informations:
        - Accuracy score
        - Precision score
        - Recall score
    '''
    acc = accuracy_score(y_true, pred)
    pre = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    return (acc, pre, rec)

def compute_rmse(y_true, pred):
    '''
    This function computes the RMSE for the given prediction pred with respect to the real
    values y_true.
    The RMSE for two vector A, B of length n is computed as
        SQRT( SUM_{i = 1}^{n} (((A_i - B_i) / A_i)^2 / n) )
    '''
    n = len(y_true)
    diff = y_true - pred
    rel_diff = diff / y_true
    mse = (rel_diff**2).sum() / n
    return np.sqrt(mse)