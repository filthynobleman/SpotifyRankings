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