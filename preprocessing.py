'''
This module contains a set of functions that can be used to preprocess the data in order
to perform some kinds of predictions.
'''

import pandas as pd
import numpy as np
import encoder
import datasetinfo as dsinfo
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
        newcol = encoder.binarizer(ds[col], prefix = col[:int(np.min([3, len(col)]))])
        newcol.index = ds.index
        ds = ds.drop(columns = [col])
        for ncol in list(newcol.columns):
            ds[ncol] = newcol[ncol]
    # Return the encoded dataset
    return ds
    

def encode_dates(dataset, inplace = False):
    '''
    This function computes the encoding of the Date column in the dataset. The encoding of the
    Date column is a label encoding, in order to preserve the ordering between the dates.
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
    # Encode the Date column with label encoding
    ds = label_encode(ds, dsinfo.DATE_COLUMN, True)
    return ds

def encode_artists(dataset, inplace = False):
    '''
    This function computes the encoding of the Artist column in the dataset. The encoding of
    the Artist column is a binary encoding, in order to preserve the absence of ordering
    between the artists.
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
    # Encode the Artist column with label encoding
    ds = binary_encode(ds, dsinfo.ARTIST_COLUMN, True)
    return ds

def encode_tracks(dataset, inplace = False):
    '''
    This function computes the encoding of the Track Name column in the dataset. The encoding
    of the Track Name column is a binary encoding, in order to preserve the absence of ordering
    between the track names.
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
    # Encode the Track Name column with label encoding
    ds = binary_encode(ds, dsinfo.TRACKNAME_COLUMN, True)
    return ds

def encode_region(dataset, inplace = False):
    '''
    This function computes the encoding of the Region column in the dataset. The encoding of
    the Region column is a binary encoding, in order to preserve the absence of ordering
    between the region codes.
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
    # Encode the Region column with label encoding
    ds = binary_encode(ds, dsinfo.REGION_COLUMN, True)
    return ds

def add_prev_streams(dataset, inplace = False):
    '''
    This function adds to the given dataset another column representing the number of streams
    recorded for a certain track in the last day when it has obtained a position in the
    Spotify's top 200 of the same country.
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
    # Add the previous streams column
    ds[dsinfo.PREVSTREAMS_COLUMN] = 0
    
