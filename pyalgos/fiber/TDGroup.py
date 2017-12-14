#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  class TDGroup
#
#------------------------------------------------------------------------

"""TDGroup - text data event information holder/accessor class.

Works together with TDFileContainer and TDPeakRecord classes.


This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@see TDFileContainer - loads/holds text data from class and provides per-event-indexed access. 
@see TDPeakRecord - holds a list of records associated with a single event.

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
__version__ = "$Revision$"
# $Source$
##-----------------------------

#import os
#from time import time
#from pyimgalgos.TDPeakRecord import TDPeakRecord

##-----------------------------
##-----------------------------

class TDGroup :
    """Gets in constructor a list of text data records and converts them in a list of objects 
    """
    def __init__(self, recs, objtype, pbits=0) :
        """Constructor
           Args:
           recs    - list of text data records from file associated with this event
           objtype - object type used for text data record processing/access
           pbits   - print control bit-word; pbits & 256 - tracking
        """
        if pbits & 256 : print 'c-tor of class %s' % self.__class__.__name__
        self.objtype = objtype
        self.pbits = pbits
        self.lst_of_objs = [objtype(rec) for rec in recs]
        
##-----------------------------

    def __call__(self) :
        """Alias of get_objs()
        """
        if self.pbits & 256 : print '__call__() of class %s' % self.__class__.__name__
        return self.get_objs()
        
##-----------------------------

    def get_objs(self) :
        """Returns list of objs in event
        """
        if self.pbits & 256 : print 'get_objs() of class %s' % self.__class__.__name__
        return self.lst_of_objs
    
##-----------------------------

    def print_attrs(self) :
        #print 'Attributes of the class %s object' % self.__class__.__name__
        if self.lst_of_objs == [] : print 'List of objects in %s is empty' % self.__class__.__name__

        print 'List of objects in the %s:' % self.__class__.__name__
        for obj in self.lst_of_objs : obj.print_short()

##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------
