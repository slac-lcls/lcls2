#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`DCVersion` for the Detector Calibration (DC) project
=====================================================================

Usage::

    # Import
    from PSCalib.DCVersion import DCVersion

    # Initialization
    o = DCVersion(vnum, tsprod=None, arr=None, cmt=None)

    # Methods
    o.set_vnum(vnum)            # sets (int) version 
    o.set_tsprod(tsprod)        # sets (double) time stamp of the version production
    o.add_data(data)            # sets (str or np.array) calibration data
    vnum   = o.vnum()           # returns (int) version number
    s_vnum = o.str_vnum()       # returns (str) version number
    tsvers = o.tsprod()         # returns (double) time stamp of the version production
    data   = o.data()           # returns (np.array) calibration array
    o.save(group)               # saves object content under h5py.group in the hdf5 file. 
    o.load(group)               # loads object content from the h5py.group of hdf5 file. 
    o.print_obj()               # print info about this object.

    # and all methods inherited from PSCalib.DCBase

See:
    * :class:`DCStore`
    * :class:`DCType`
    * :class:`DCRange`
    * :class:`DCVersion`
    * :class:`DCBase`
    * :class:`DCInterface`
    * :class:`DCUtils`
    * :class:`DCDetectorId`
    * :class:`DCConfigParameters`
    * :class:`DCFileName`
    * :class:`DCLogger`
    * :class:`DCMethods`
    * :class:`DCEmail`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2016 by Mikhail Dubrovin
"""
#------------------------------

import os
import sys
import numpy as np
#from time import time
from math import floor
from PSCalib.DCInterface import DCVersionI
from PSCalib.DCLogger import log
from PSCalib.DCUtils import sp, get_subgroup, save_object_as_dset

#------------------------------

def version_int_to_str(vnum) : return ('v%04d' % vnum) if vnum is not None else 'None'

def version_str_to_int(vstr) : return int(vstr.lstrip('v').lstrip('0'))    

class DCVersion(DCVersionI) :

    """Class for the Detector Calibration (DC) project

    Parameters
    \n vnum : int - version number
    \n tsprod : double - time in sec
    \n arr : numpy.array - array of constants to save
    \n cmt : str - comment
    """

    def __init__(self, vnum, tsprod=None, arr=None, cmt=None) : # int, double, np.array
        DCVersionI.__init__(self, vnum, tsprod, arr, cmt)
        self._name = self.__class__.__name__

        self.set_vnum(vnum)
        self.set_tsprod(tsprod)
        self.add_data(arr)
        log.debug('In c-tor for version: %s' % vnum, self._name)

    def set_vnum(self, vnum)       : self._vnum = vnum     # int

    def set_tsprod(self, tsprod)   : self._tsprod = tsprod # double or None

    def add_data(self, data)       : self._data = data     # np.array, str or None

    def vnum(self)                 : return self._vnum     # int

    def str_vnum(self)             : return version_int_to_str(self._vnum) # str

    def tsprod(self)               : return self._tsprod   # double

    def data(self)                 : return self._data      # np.array

    def save(self, group) :
        grp = get_subgroup(group, self.str_vnum())                      # (str)
        ds1 = save_object_as_dset(grp, 'version', data=self.vnum())     # dtype='int'
        ds2 = save_object_as_dset(grp, 'tsprod',  data=self.tsprod())   # dtype='double'
        ds3 = save_object_as_dset(grp, 'data',    data=self.data())     # dtype='str' or 'np.array'

        msg = '==== save(), group %s object for version %d' % (grp.name, self.vnum())
        log.debug(msg, self._name)

        self.save_base(grp)


    def load(self, grp) :
        msg = '==== load data from group %s and fill object %s' % (grp.name, self._name)
        log.debug(msg, self._name)

        for k,v in dict(grp).iteritems() :
            #subgrp = v
            #print '    ', k , v

            if isinstance(v, sp.dataset_t) :                    
                log.debug('load dataset "%s"' % k, self._name)
                #t0_sec = time()
                if   k == 'version': self.set_vnum(v[0])
                elif k == 'tsprod' : self.set_tsprod(v[0])
                elif k == 'data'   :
                    d = v.value
                    if str(d.dtype)[:2] == '|S' :
                        d=d.tostring() # .split('\n')
                        #print 'XXX: d, type(d): %s'%d, type(d)
                    self.add_data(d)

                else : log.warning('group "%s" has unrecognized dataset "%s"' % (grp.name, k), self._name)
                #print 'TTT %s dataset "%s" time (sec) = %.6f' % (sys._getframe().f_code.co_name, k, time()-t0_sec)

            elif isinstance(v, sp.group_t) :
                if self.is_base_group(k,v) : continue
            #    print 'XXX: ', self._name, k,v
            #    log.debug('load group "%s"' % k, self._name)
            #    o = self.add_version(v['version'][0])
            #    o.load(v)


    def print_obj(self) :
        offset = 4 * self._offspace
        self.print_base(offset)
        print '%s version    %s' % (offset, self.vnum())

        tsec = self.tsprod()
        msg = '%d: %s' % (floor(tsec), self.tsec_to_tstr(tsec)) if tsec is not None else 'None'
        print '%s tsprod     %s' % (offset, msg)

        data = self.data()
        if isinstance(data, np.ndarray) :
           print '%s data.shape %s  dtype %s' % (offset, str(data.shape), str(data.dtype))
        else :
           print '%s data' % (offset)
           for s in data.split('\n') : print '%s     %s' % (offset, s)

        #for k,v in self.versions().iteritems() :
        #    v.print_obj()

    def __del__(self) :
        pass

#---- TO-DO

#    def get(self, p1, p2, p3)  : return None

#------------------------------

def test_DCVersion() :

    o = DCVersion(None)

    o.set_vnum(5)
    o.set_tsprod(None)
    o.add_data(None)

    r = o.vnum()
    r = o.str_vnum()
    r = o.tsprod()
    r = o.data()

    #o.save(None)
    #o.load(None)

    r = o.get(None, None, None)    

#------------------------------

def test() :
    log.setPrintBits(0377) 

    if len(sys.argv)==1 :
        print 'For test(s) use command: python %s <test-number=1-4>' % sys.argv[0]
        test_DCVersion()
    elif(sys.argv[1]=='1') : test_DCVersion()        
    else : print 'Non-expected arguments: sys.argv = %s use 1,2,...' % sys.argv

#------------------------------

if __name__ == "__main__" :
    test()
    sys.exit( 'End of %s test.' % sys.argv[0])

#------------------------------
