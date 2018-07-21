'''
This module provides a set of functions used to encode categorical features.
'''

import numpy as np
import pandas as pd
import numbers

from sklearn.preprocessing import LabelEncoder

def int2binarray(val, length = None):
    '''
    This functions returns a numpy array containing the binary representation of the given
    value. In the resulting array, each component is a digit of the binary representation of
    the value.
    The length of the array is determined by the input parameter length as follows:
        - If length is None, then the length of the array is the exact number of digits
          required to represent the value.
        - If length is greater than the digits required to represent the value, then the
          length of the array will be equals to the parameter length.
    The function raises the following errors:
        TypeError: If val is not convertible to an integer.
                   If length is not an integer or None.
        ArithmeticError: If length is an integer, but it is lower than the number of digits
                         required to represent the value. Namely, length < ceil(log2(value)).
                         If val is lower than zero.
    '''
    # Convert val to integer
    lval = 0
    try:
        lval = int(val)
    except:
        raise TypeError("The input value must be castable to integer.")
    # Check if lower than zero
    if lval < 0:
        raise ArithmeticError("The input value must be non-negative.")
    # Compute the length
    llength = int(np.ceil(np.log2(np.max([lval, 1]))))
    if length is not None:
        if not isinstance(length, numbers.Integral):
            raise TypeError("The length must be an integer.")
        if length < llength:
            raise ArithmeticError("The length cannot be less than the required digits.")
        llength = length
    # Compute the representation
    lst = list()
    while lval > 0:
        lst.insert(0, lval % 2)
        lval = lval / 2
    # Add the zeroes to fill the array
    while len(lst) < llength:
        lst.insert(0, 0)
    return np.array(lst)


def binarizer(series, prefix = None, suffix = None):
    '''
    This function transforms the given pandas series (or numpy array) into a dataframe where
    each row is an array that represents the binary encoding of the element in the 
    corresponding row of the series. Notice that the given series is firtly label encoded,
    and then the binarization is applied. Also, notice that the length of the rows are equals
    to the length of the binary representation of the largest integer in the encoding.
    The columns of the dataframe are renamed using the following convention:
        <prefix>i<suffix>
    where prefix and suffix are the inputs and i is the column index.
    The function raises the following errors and exceptions:
        TypeError: If series is not a pandas series or a numpy array.
                   If prefix is not a string or None.
                   If suffix is not a sring or None.
    '''
    # Check the validity of the type of series
    if not isinstance(series, pd.Series) and not isinstance(series, np.array):
        raise TypeError("The input series must be a pandas series or a numpy array.")
    lseries = pd.Series(series)
    # Validate prefix
    lprefix = prefix
    if prefix is None:
        lprefix = ''
    if not isinstance(lprefix, str):
        raise TypeError("The input prefix must be a string or None.")
    lprefix = str(lprefix)
    # Validate suffix
    lsuffix = suffix
    if suffix is None:
        lsuffix = ''
    if not isinstance(lsuffix, str):
        raise TypeError("The input suffix must be a string or None.")
    lsuffix = str(lsuffix)
    # Label encode the series
    # Sort the values according to their natural ordering
    le = LabelEncoder()
    vals = lseries.unique()
    vals.sort()
    le.fit(vals)
    lseries = pd.Series(le.transform(lseries))
    # Compute the length
    length = int(np.max([1, np.ceil(np.log2(lseries.max()))]))
    # Create the dataframe from the series of array
    frame = lseries.apply(lambda x: int2binarray(x, length)).apply(lambda x: pd.Series(x))
    # Add prefix and suffix and return
    frame = frame.add_prefix(lprefix).add_suffix(lsuffix)
    return frame