'''
This module implements a set of custom errors and exceptions for the program.
'''

class InvalidTypeException(Exception):
    '''
    This exception is raised when a method or a function is called with a string that does not
    identify a valid object type.
    '''

    def __init__(self, errstring):
        super(InvalidTypeException, self).__init__(errstring)

class NoneReferenceException(Exception):
    '''
    This exception is raised when a method or a function is executed on an object that is
    supposed to be already initialized to a value that is not None.
    '''

    def __init__(self, errstring):
        super(NoneReferenceException, self).__init__(errstring)

class UndefinedPredictorException(Exception):
    '''
    This exception is raised when a method or a function tries to fit a predictor that is
    None or tries to predict with a predictor that is None.
    '''

    def __init__(self, errstring):
        super(UndefinedPredictorException, self).__init__(errstring)

class UndefinedDatasetException(Exception):
    '''
    This exception is raised when a method or a function tries to interact with an undefined
    dataset.
    '''

    def __init__(self, errstring):
        super(UndefinedDatasetException, self).__init__(errstring)

class EncodingNotUpToDateException(Exception):
    '''
    This exception is raised when a method or a function tries to operate with an encoded
    dataset that is None or whose the rows does not match those of the original dataset.
    '''

    def __init__(self, errstring):
        super(EncodingNotUpToDateException, self).__init__(errstring)
        