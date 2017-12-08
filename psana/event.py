#!/usr/bin/env python
#

import sys, os

sys.path.append('../build/psana')
import dgram

class Event:
    """Stores variables and arrays for a single event, loaded from an XTC source.\n"""
    def __init__(self, d):
        #d=dgram.Dgram(config=config)
        for key in sorted(d.__dict__.keys()):
            setattr(self, key, getattr(d, key))

