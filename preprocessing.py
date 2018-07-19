'''
This module contains a set of functions that can be used to preprocess the data in order
to perform some kinds of predictions.
'''

import pandas as pd
import numpy as np
import encoder
from sklearn.preprocessing import LabelEncoder

def label_encode(dataset, columns = None, inplace = False):
    '''
    This function computes the label encoding of the given columns in the given dataset and
    replaces the old column with the encoded ones.
    If the input parameter columns is a string, then the column with that name is encoded.
    If columns is a list, then the encoding is computed for all the columns with the name in
    the list.
    If columns is None, then every column of type 'object' is encoded.
    If the input parameter inplace is True, then the modifications of the dataset are performed
    directly on the dataset itself. Otherwise, a deep copy of the dataset is produced and the
    modifications are applied to the copy, which is then returned.
    The function raises the following errors and exceptions:
        TypeError: If dataset is not a pandas dataframe.
    '''
    #Check if dataset is a pandas dataframe
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("The input dataset must be a pandas dataframe.")
    # Rename or copy the dataset, depending on inplace
    ds = dataset
    if not inplace:
        ds = dataset.copy()
    # Compute the list of columns to encode
    cols = None
    if isinstance(columns, str):
        cols = [columns]
    elif columsn is None:
        cols = list()
        for col in list(ds.columns):
            if ds[col].dtype == 'object':
                cols.append(col)
    else:
        cols = list(columns)
    # For each column, apply the label encoding
    for col in cols:
        le = LabelEncoder()
        ds[col] = le.fit_transform(ds[col])
    # Return the encoded dataset
    return ds

def binary_encode(dataset, columns = None, inplace = False):
    '''
    This function computes the binary encoding of the given columns in the given dataset and
    replaces the old column with the encoded ones.
    If the input parameter columns is a string, then the column with that name is encoded.
    If columns is a list, then the encoding is computed for all the columns with the name in
    the list.
    If columns is None, then every column of type 'object' is encoded.
    If the input parameter inplace is True, then the modifications of the dataset are performed
    directly on the dataset itself. Otherwise, a deep copy of the dataset is produced and the
    modifications are applied to the copy, which is then returned.
    The function raises the following errors and exceptions:
        TypeError: If dataset is not a pandas dataframe.
    '''
    #Check if dataset is a pandas dataframe
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("The input dataset must be a pandas dataframe.")
    # Rename or copy the dataset, depending on inplace
    ds = dataset
    if not inplace:
        ds = dataset.copy()
    # Compute the list of columns to encode
    cols = None
    if isinstance(columns, str):
        cols = [columns]
    elif columsn is None:
        cols = list()
        for col in list(ds.columns):
            if ds[col].dtype == 'object':
                cols.append(col)
    else:
        cols = list(columns)
    # For each column, apply the binary encoding
    for col in cols:
        newcol = encoder.binarizer(ds[col], prefix = col[:int(np.max([3, len(col)]))])
        newcol.index = ds.index
        ds = ds.drop(columns = [col])
        for ncol in list(newcol.columns):
            ds[ncol] = newcol[ncol]
    # Return the encoded dataset
    return ds
    

