#!/usr/bin/env python
""" test namedtuple
"""
import sys
print('e.g.: [python] %s' % sys.argv[0])

import numpy as np
import collections

#----------

def namedtuple(typename, field_names, default_values=()):
    """
    Overwriting namedtuple class to use default arguments for variables not passed in at creation of object
    Can manually set default value for a variable; otherwise None will become default value
    """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)

    #if isinstance(default_values, collections.Mapping):
    if isinstance(default_values, collections.abc.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)

    T.__new__.__defaults__ = tuple(prototype)
    return T

 
o = namedtuple('ROIMetrics',
    ['xN', #Size of the image in X   
    'x0',  #Position of the first pixel in x
    'yN',  #Size of the image in Y 
    'y0',  #Position of the first pixel in y
    'x',   #X vector
    'y',   #Y vector
    ], 
    {'xN': 1024,                      
     'x0': 0, 
     'yN': 1024, 
     'y0': 0,
     'x': np.arange(0, 1024),
     'y': np.arange(0, 1024)})

#----------

def test_namedtuple() :
    print('\n object ROIMetrics:', o)
    print('\n dir(o)', dir(o))
    print('\n o.__dict__:', o.__dict__)
    print('\n _fields:', o._fields)
    print('\n type(o.x):', type(o.x))
    print('\n ttt:', dir(o.x))
    print('\n __slots__:', o.__slots__)
    print('\n getattr(...):', getattr(o, 'xN'))
    print('\n _field_defaults', o._field_defaults)
    #print('\n __defaults__:', o.__defaults__)
    #print('\n ttt:', o._asdict['x'])
    #print('\n o._asdict:', o._asdict())
    #print('\n _fields_defaults', o._fields_defaults)
    #print('\n count()', o.count())



#----------

if __name__ == "__main__":
    test_namedtuple()

#----------
