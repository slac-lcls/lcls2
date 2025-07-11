#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`DCRange` for the Detector Calibration (DC) project
===================================================================

Usage::

    # Import
    from PSCalib.DCRange import DCRange

    # Initialization
    o = DCRange(begin, end=None, cmt=None)

    # Methods
    str_range   = o.range()               # (str) of the time stamp validity range
    t_sec       = o.begin()               # (double) time stamp beginning validity range
    t_sec       = o.end()                 # (double) time stamp ending validity range or (str) 'end'
    dico        = o.versions()            # (list of uint) versions of calibrations
    v           = o.vnum_def()            # returns default version number
    v           = o.vnum_last()           # returns last version number 
    vo          = o.version(vnum=None)    # returns version object for specified version
    ts_in_range = o.tsec_in_range(tsec)   # (bool) True/False if tsec is/not in the validity range
    evt_in_range= o.evt_in_range(evt)     # (bool) True/False if evt is/not in the validity range
    o.set_begin(tsbegin)                  # set (int) time stamp beginning validity range
    o.set_end(tsend)                      # set (int) time stamp ending validity range
    o.add_version(vnum=None, tsec_prod=None, nda=None, cmt=None) # add object for new version of calibration data
    o.set_vnum_def(vnum=None)             # set default version number, if available. vnum=None - use last available.
    vd = o.mark_version(vnum=None)        # mark version for deletion, returns version number or None if nothing was deleted
    o.mark_versions()                     # mark all registered versions for deletion

    o.save(group)                         # saves object content under h5py.group in the hdf5 file. 
    o.load(group)                         # loads object content from the hdf5 file. 
    o.print_obj()                         # print info about this object and its children

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

Created: 2016-09-23 at Mikhail Dubrovin
"""
#------------------------------

import os
import sys
from math import floor, ceil
#import math
#import numpy as np
#from time import time
#from PSCalib.DCConfigParameters import cp
from PSCalib.DCInterface import DCRangeI
from PSCalib.DCLogger import log
from PSCalib.DCVersion import DCVersion, version_int_to_str
from PSCalib.DCUtils import sp, evt_time, get_subgroup, save_object_as_dset, delete_object

#------------------------------

def key(begin, end=None) :
    """ Return range as a string,
    ex.: 1471285222-1471285555 or 1471285222-end from double time like 1471285222.123456
    """
    str_begin = ('%d' % floor(begin)) if begin is not None else 0
    str_end = 'end' if (end is None or end=='end') else ('%d' % ceil(end))
    return '%s-%s' % (str_begin, str_end)

#------------------------------

class DCRange(DCRangeI) :

    """Class for the Detector Calibration (DC) project

    Parameters
    
    - begin : double - time in sec
    - end   : double - time in sec or None meaning infinity
    - cmt   : str - comment
    """

    def __init__(self, begin, end=None, cmt=None) : # double, double/None
        DCRangeI.__init__(self, begin, end, cmt)
        self._name = self.__class__.__name__
        self.set_begin(begin)
        self.set_end(end)
        self._dicvers = {}
        self._vnum_def = 0 # 0 = use last
        self._str_range = key(begin, end)
        log.debug('In c-tor for range: %s' % self._str_range, self._name)

    def range(self)                : return self._str_range

    def begin(self)                : return self._begin

    def end(self)                  : return self._end

    def versions(self)             : return self._dicvers

    def vnum_def(self) :
        #return self.vnum_last()
        if self._vnum_def == 0 or self._vnum_def is None :
            return self.vnum_last()
        return self._vnum_def 

    def vnum_last(self) :
        keys = self._dicvers.keys()
        return keys[-1] if len(keys) else 0

    def version(self, vnum=None)   :
        v = vnum if vnum is not None else self.vnum_def()
        return self._dicvers.get(v, None) if v is not None else None

    def set_begin(self, begin)     : self._begin = begin

    def set_end(self, end=None)    : self._end = 'end' if end is None else end

    def set_str_range(self, str_range) : self._str_range = str_range


    def add_version(self, vnum=None, tsec_prod=None, nda=None, cmt=False) :
        vn = self.vnum_last() + 1 if vnum is None else vnum
        if vn in self._dicvers.keys() :
            return self._dicvers[vn]
        o = self._dicvers[vn] = DCVersion(vn, tsec_prod, nda)

        rec = self.make_record('add version', str(vn), cmt) 
        if cmt is not False : self.add_history_record(rec)
        log.info(rec, self.__class__.__name__)
        return o


    def set_vnum_def(self, vnum=None) :
        if vnum is None or vnum == 0 :
            self._vnum_def = 0 # will use last
        elif vnum in self._dicvers.keys() :
            self._vnum_def = vnum
            self.add_history_record('WARNING: set_vnum_defdef sets default version %d' % vnum)
        else :
            msg = 'Attemt to set non-existent version %d as default' % vnum
            log.warning(msg, self._name)


    def mark_version(self, vnum=None, cmt=False) :
        """Marks child object for deletion in save()"""
        vers = self.vnum_last() if vnum is None else vnum

        if vers in self._dicvers.keys() :
            self._lst_del_keys.append(vers)

            rec = self.make_record('del version', str(vers), cmt) 
            if cmt is not False : self.add_history_record(rec)
            log.info(rec, self.__class__.__name__)
            return vers
        else :
            msg = 'Marking of non-existent version %s' % str(vers)
            log.warning(msg, self._name)
            return None


    def mark_versions(self) :
        """Marks all child objects for deletion in save()"""
        for vers in self._dicvers.keys() :
            self.mark_version(vers)
            #self._lst_del_keys.append(vers)


    def __del__(self) :
        for vers in self._dicvers.keys() :
            del self._dicvers[vers]


    def clear_versions(self) :
        self._dicvers.clear()


    def tsec_in_range(self, tsec) :
        if tsec < self.begin() : return False 
        if self.end() == 'end' : return True 
        if tsec > self.end()   : return False 
        return True


    def evt_in_range(self, evt) :
        return self.tsec_in_range(evt_time(evt))


    def __cmp__(self, other) :
        """for comparison in sorted()
        """
        if self.begin() <  other.begin() : return -1
        if self.begin() >  other.begin() : return  1
        if self.begin() == other.begin() : 
            if self.end() == other.end() : return  0
            if self.end()  == 'end'      : return -1 # inverse comparison for end
            if other.end() == 'end'      : return  1
            if self.end()  > other.end() : return -1
            if self.end()  < other.end() : return  1


    def save(self, group) :

        grp = get_subgroup(group, self.range())

        ds1 = save_object_as_dset(grp, 'begin',   data=self.begin())    # dtype='double'
        ds2 = save_object_as_dset(grp, 'end',     data=self.end())      # dtype='double'
        ds3 = save_object_as_dset(grp, 'range',   data=self.range())    # dtype='str'
        ds4 = save_object_as_dset(grp, 'versdef', data=self._vnum_def)  # dtype='int'

        msg = '=== save(), group %s object for %s' % (grp.name, self.range())
        log.debug(msg, self._name)

        #print 'ZZZ: self.versions()', self.versions() 

        # save/delete objects in/from hdf5 file
        for k,v in self._dicvers.iteritems() :
            if k in self._lst_del_keys : delete_object(grp, version_int_to_str(k))
            else : v.save(grp)

        # deletes items from dictionary
        for k in self._lst_del_keys :
            del self._dicvers[k]

        self._lst_del_keys = []

        self.save_base(grp)


    def load(self, grp) :
        msg = '=== load data from group %s and fill object %s' % (grp.name, self._name)
        log.debug(msg, self._name)

        #print  'XXX load grp, keys:', grp, grp.keys()
        for k,v in dict(grp).iteritems() :
            #subgrp = v
            if isinstance(v, sp.dataset_t) :                    
                log.debug('load dataset "%s"' % k, self._name)
                if   k == 'begin'   : self.set_begin(v[0])
                elif k == 'end'     : self.set_end(v[0])
                elif k == 'range'   : self.set_str_range(v[0])
                elif k == 'versdef' : self.set_vnum_def(v[0]) # self._vnum_def = v[0]
                else : log.warning('group "%s" has unrecognized dataset "%s"' % (grp.name, k), self._name)

            elif isinstance(v, sp.group_t) :
                #print '  YYY:group v.name, v.keys():', v.name, v.keys()
                if self.is_base_group(k,v) : continue
                log.debug('load group "%s"' % k, self._name)
                version = v.get('version')
                if version is None :
                    msg = 'corrupted file structure - group "%s" does not contain key "version", keys: "%s"' % (v.name, v.keys())
                    log.error(msg, self._name)
                    print 'ERROR:', self._name, msg
                    continue
                o = self.add_version(version[0], cmt=False)
                o.load(v)


    def print_obj(self) :
        offset = 3 * self._offspace
        self.print_base(offset)
        print '%s begin     %s' % (offset, self.begin()),
        print ': %s'            % self.tsec_to_tstr(self.begin())
        print '%s end       %s' % (offset, self.end()),
        print ' %s'             % ('' if (self.end() in (None,'end')) else self.tsec_to_tstr(self.end()))
        print '%s range     %s' % (offset, self.range())
        print '%s versdef   %s' % (offset, self.vnum_def())
        print '%s N vers    %s' % (offset, len(self.versions()))
        print '%s versions  %s' % (offset, str(self.versions().keys()))

        for k,v in self.versions().iteritems() :
            v.print_obj()

#------------------------------

def test_DCRange() :

    o = DCRange(None, None)

    r = o.begin()
    r = o.end()
    r = o.versions()
    r = o.vnum_def()
    r = o.version(None)
    o.set_begin(None)
    o.set_end(None)
    o.add_version()
    o.set_vnum_def(None)
    v = o.del_version(None)
    o.del_versions()
    o.clear_versions()

    r = o.get(None, None, None)    
    #o.save(None)
    o.load(None)

#------------------------------

def test() :
    log.setPrintBits(0377) 

    if len(sys.argv)==1 : print 'For test(s) use command: python %s <test-number=1-4>' % sys.argv[0]
    elif(sys.argv[1]=='1') : test_DCRange()        
    else : print 'Non-expected arguments: sys.argv = %s use 1,2,...' % sys.argv

#------------------------------

if __name__ == "__main__" :
    test()
    sys.exit( 'End of %s test.' % sys.argv[0])

#------------------------------
