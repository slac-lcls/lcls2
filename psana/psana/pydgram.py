#!/usr/bin/env python
#

import sys

from psana import dgram
import numpy as np

class container:
    pass

class PyDgram:
    """Stores variables and arrays for a single event, loaded from an XTC source.

       The sole purpose of the Event class is to hold variables and numpy
       arrays that were created by a dGram object.  Ideally the variables
       and arrays would stay in the dGram object, but this creates cyclic
       references that Python is unable to garbage collect.  Moving the
       variables and arrays to an Event object breaks this cycle,
       allowing Python to reclaim array memory once it is no longer
       referenced by users.

       The basic problem is array memory is created and managed by the
       dGram object, meaning all arrays have references (i.e. pointers)
       to the dGram object (where their data lives).  The cyclic
       references exist as long as dGram objects also hold references
       back to these arrays.  Simply moving the array references to a
       different place breaks this cycle.

       A better solution would be to add support for cyclic garbage
       collection to the dGram class, but at the time of this writing
       numpy arrays do not support this.  We may revisit this issue
       when numpy arrays support cyclic garbage collection.
    """
    def __init__(self, d):
        #d=dgram.Dgram(config=config)
        #for key in sorted(d.__dict__.keys()):
        #    setattr(self, key, getattr(d, key))
        self.buf = np.asarray(d)
        keys = sorted(d.__dict__.keys())
        for k in keys:
            fields = k.split('_')
            currobj = self
            for f in fields[:-1]:
                if not hasattr(currobj,f):
                    setattr(currobj,f,container())
                currobj = getattr(currobj,f)
            val = getattr(d,k)
            setattr(currobj,fields[-1],val)

