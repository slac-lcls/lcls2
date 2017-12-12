#!/usr/bin/env python
#

import sys, os

sys.path.append('../build/psana')
import dgram

class container:
    pass

class Event:
    """Stores variables and arrays for a single event, loaded from an XTC source.\n"""
    def __init__(self, d):
        #d=dgram.Dgram(config=config)
        #for key in sorted(d.__dict__.keys()):
        #    setattr(self, key, getattr(d, key))
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
