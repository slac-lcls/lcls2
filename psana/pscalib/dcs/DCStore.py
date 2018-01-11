#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`DCStore` for the Detector Calibration (DC) project
===================================================================

Usage::

    # Import
    from PSCalib.DCStore import DCStore

    # Initialization
    o = DCStore(fpath)

    # Methods
    tscfile     = o.tscfile()               # (double) time stamp of the file creation
    dettype     = o.dettype()               # (str) detector type
    detid       = o.detid()                 # (str) detector id
    detname     = o.detname()               # (str) detector name of self object
    predecessor = o.predecessor()           # (str) detname of predecessor or None
    successor   = o.successor()             # (str) detname of successor or None
    ctypes      = o.ctypes()                # (list) calibration types in the file
    cto         = o.ctypeobj(ctype)         # (DCType ~ h5py.Group) calibration type object
    o.set_tscfile(tsec)                     # set (double) time stamp of the file creation 
    o.set_dettype(dettype)                  # set (str) detector type
    o.set_detid(detid)                      # set (str) detector id
    o.set_detname(detname)                  # set (str) detector name of self object
    o.set_predecessor(pred)                 # set (str) detname of predecessor or None
    o.set_successor(succ)                   # set (str) detname of successor or None
    o.add_ctype(ctype)                      # add (str) calibration type to the DCStore object
    ctype = o.mark_ctype(ctype)             # delete ctype (str) from the DCStore object, returns ctype or None
    o.mark_ctypes()                         # delete all ctypes (str) from the DCStore object
    o.clear_ctype()                         # clear all ctypes (str) from the DCStore object dictionary

    o.save(group, mode='r+')                # saves object in hdf5 file. mode='r+'/'w' update/rewrite file.
    o.load(group)                           # loads object content from the hdf5 file. 
    o.print_obj()                           # print info about this object and its children

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
from time import time

#import math
#import numpy as np
#from PSCalib.DCConfigParameters import cp
from PSCalib.DCInterface import DCStoreI
from PSCalib.DCType import DCType
from PSCalib.DCLogger import log
from PSCalib.DCUtils import gu, sp, save_object_as_dset, evt_time, delete_object

#------------------------------

def print_warning(obj, metframe) :
    wng = 'INFO: %s.%s - abstract interface method needs to be re-implemented in derived class.' \
          % (obj.__class__.__name__, metframe.f_code.co_name)
    log.warning(wng, obj.__class__.__name__)
    #print wng
    #raise NotImplementedError(wng)

#------------------------------
#------------------------------

class DCStore(DCStoreI) :
    
    """Class for the Detector Calibration (DC) project

    Parameters
    
    path : str - path to the hdf5 file with calibration info
    cmt  : str - comment
    """

#------------------------------

    def __init__(self, path, cmt=None) :
        DCStoreI.__init__(self, path, cmt)
        self._name = self.__class__.__name__
        self._set_file_name(path)
        self._tscfile = None
        self._predecessor = None
        self._successor = None
        self._dicctypes = {}
        log.debug('In c-tor for path: %s' % path, self._name)
        
#------------------------------

    def _set_file_name(self, path) :
        self._fpath = path if isinstance(path, str) else None
        if self._fpath is None : return
        root, ext = os.path.splitext(path)
        detname = os.path.basename(root)
        dettype, detid = detname.split('-',1)
        self.set_dettype(dettype)
        self.set_detid(detid)

#------------------------------

    def tscfile(self)               : return self._tscfile

    def dettype(self)               : return self._dettype

    def detid(self)                 : return self._detid

    def detname(self)               :
        if self._dettype is None : return None
        if self._detid is None : return None
        return '%s-%s' % (self._dettype, self._detid)

    def predecessor(self)           : return self._predecessor

    def successor(self)             : return self._successor

    def ctypes(self)                : return self._dicctypes

    def ctypeobj(self, ctype)       : return self._dicctypes.get(ctype, None) if ctype is not None else None

    def set_tscfile(self, tsec=None): self._tscfile = time() if tsec is None else tsec

    def set_dettype(self, dettype)  : self._dettype = str(dettype)

    def set_detid(self, detid)      : self._detid = str(detid)

    def set_detname(self, detname) :
        if not isinstance(detname, str) :
            self._dettype, self._detid = None, None
            return

        fields = detname.split('-',1)
        self._dettype, self._detid = fields[0], fields[1]

    def set_predecessor(self, pred=None) : self._predecessor = pred 

    def set_successor(self, succ=None)   : self._successor = succ


    def add_ctype(self, ctype, cmt=False) :
        if not (ctype in gu.calib_names) : 
            msg = 'ctype "%s" is not in the list of known types:\n  %s' % (ctype, gu.calib_names)
            log.error(msg, self.__class__.__name__)
            return None
            
        if ctype in self._dicctypes.keys() :
            return self._dicctypes[ctype]
        o = self._dicctypes[ctype] = DCType(ctype)

        rec = self.make_record('add ctype', ctype, cmt) 
        if cmt is not False : self.add_history_record(rec)
        log.info(rec, self.__class__.__name__)
        return o


    def mark_ctype(self, ctype, cmt=False) :
        """Marks child object for deletion in save()"""
        if ctype in self._dicctypes.keys() :
            self._lst_del_keys.append(ctype)

            rec = self.make_record('del ctype', ctype, cmt) 
            if cmt is not False : self.add_history_record(rec)
            log.info(rec, self.__class__.__name__)
            return ctype
        else :
            msg = 'Marking of non-existent ctype "%s"' % str(ctype)
            log.warning(msg, self._name)
            return None


    def mark_ctypes(self) :
        """Marks all child objects for deletion in save()"""
        for ctype in self._dicctypes.keys() :
            self.mark_ctype(ctype)
            #self._lst_del_keys.append(ctype)


    def __del__(self) :
        for ctype in self._dicctypes.keys() :
            del self._dicctypes[ctype] 


    def clear_ctypes(self) :
        self._dicctypes.clear()     


    def save(self, path=None, mode='r+') :
        if path is not None : self._fpath = path
        if not isinstance(self._fpath, str) :
            msg = 'Invalid file name: %s' % str(self._fpath)
            log.error(msg, self.__class__.__name__)
            raise ValueError(msg)

        mode_rw = mode if os.path.exists(self._fpath) else 'w'
        
        with sp.File(self._fpath, mode_rw) as grp :

            msg = '= save(), group %s object for %s' % (grp.name, self.detname())
            log.debug(msg, self._name)

            ds1 = save_object_as_dset(grp, 'dettype',     data=self.dettype())     # 'str'
            ds2 = save_object_as_dset(grp, 'detname',     data=self.detname())     # 'str'
            ds3 = save_object_as_dset(grp, 'detid',       data=self.detid())       # 'str'
            ds4 = save_object_as_dset(grp, 'tscfile',     data=self.tscfile())     # 'double'
            ds5 = save_object_as_dset(grp, 'predecessor', data=self.predecessor()) # 'str'       
            ds6 = save_object_as_dset(grp, 'successor',   data=self.successor())   # 'str'

            # save/delete objects in/from hdf5 file
            for k,v in self._dicctypes.iteritems() :
                if k in self._lst_del_keys : delete_object(grp, k)
                else : v.save(grp)
                       #self._dicctypes[k].save(grp)

            # deletes items from dictionary
            for k in self._lst_del_keys :
                del self._dicctypes[k]

            self._lst_del_keys = []

            self.save_base(grp)

            grp.close()
            log.info('File %s is updated/saved' % self._fpath, self._name)


    def load(self, path=None) : 

        with sp.File(self._fpath, 'r') as grp :
            
            #msg = 'Load data from file %s and fill %s object for group "%s"' % (self._fpath, self._name, grp.name)
            #log.info(msg, self._name)
            log.info('Load data from file %s' % self._fpath, self._name)

            for k,v in dict(grp).iteritems() :
                #subgrp = v
                #print '    ', k # , "   ", subg.name #, val, subg.len(), type(subg),

                if isinstance(v, sp.dataset_t) :                    
                    log.debug('load dataset "%s"' % k, self._name)
                    if   k == 'dettype'     : self.set_dettype(v[0])
                    elif k == 'detid'       : self.set_detid(v[0])
                    elif k == 'detname'     : self.set_detname(v[0])
                    elif k == 'tscfile'     : self.set_tscfile(v[0])
                    elif k == 'predecessor' : self.set_predecessor(v[0])
                    elif k == 'successor'   : self.set_successor(v[0])
                    else : log.warning('hdf file has unrecognized dataset "%s"' % k, self._name)

                elif isinstance(v, sp.group_t) :
                    if self.is_base_group(k,v) : continue
                    log.debug('load group "%s"' % k, self._name)                    
                    o = self.add_ctype(k, cmt=False)
                    o.load(v)
 

    def print_obj(self) :
        offset = 1 * self._offspace
        self.print_base(offset)
        tsec = self.tscfile()
        print '%s dettype     %s' % (offset, self.dettype())
        print '%s detid       %s' % (offset, self.detid())
        print '%s detname     %s' % (offset, self.detname())
        print '%s tscfile     %s' % (offset, ('%.9f: %s' % (tsec, self.tsec_to_tstr(tsec)))\
                                     if tsec is not None else str(tsec))
        print '%s predecessor %s' % (offset, self.predecessor())
        print '%s successor   %s' % (offset, self.successor())
        print '%s ctypes'         % (offset)

        print '%s N types     %s' % (offset, len(self.ctypes()))
        print '%s types       %s' % (offset, str(self.ctypes().keys()))

        for k,v in self.ctypes().iteritems() :
        #    #msg='Add type %s as object %s' % (k, v.ctype())
        #    #log.info(msg, self._name)
            v.print_obj()

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

def test_DCStore() :

    o = DCStore('cspad-654321.h5')

    r = o.tscfile()
    r = o.dettype()
    r = o.detid()
    r = o.detname()
    r = o.predecessor()
    r = o.successor()
    r = o.ctypes()
    r = o.ctypeobj(None)
    r = o.get(None, None, None)
    o.set_tscfile(None)
    o.set_dettype(None)
    o.set_detid(None)
    o.set_detname(None)
    o.set_predecessor(None)
    o.set_successor(None)
    o.add_ctype(None)
    o.mark_ctype(None)
    o.mark_ctypes()
    o.clear_ctypes()
    #o.save(None)
    o.load(None)

#------------------------------

def test_DCStore_save() :

    import numpy as np

    o = DCStore('cspad-654321.h5')
    o.set_dettype('cspad')
    o.set_detid('654321')
    o.set_tscfile(tsec=None)
    o.set_predecessor('cspad-654320')
    o.set_successor('cspad-654322')
    o.add_history_record('Some record 1 to commenet DCStore')
    o.add_history_record('Some record 2 to commenet DCStore')
    o.add_par('par-1-in-DCStore', 1)
    o.add_par('par-2-in-DCStore', 'some string 1')
    o.add_par('par-3-in-DCStore', 1.1)

    o.add_ctype('pixel_rms')
    o.add_ctype('pixel_status')
    o.add_ctype('pixel_mask')
    o.add_ctype('pixel_gain')
    o.add_ctype('geometry')
    po = o.add_ctype('pedestals')
    po.add_history_record('Some record 1 to commenet DCType')
    po.add_history_record('Some record 2 to commenet DCType')
    po.add_par('par-1-in-DCType', 2)
    po.add_par('par-2-in-DCType', 'some string 2')
    po.add_par('par-3-in-DCType', 2.2)

    t1 = time();
    t2 = t1+1;
    ro1 = po.add_range(t1, end=t1+1000)
    ro2 = po.add_range(t2, end=t2+1000)
    ro1.add_history_record('Some record 1 to commenet DCRange')
    ro1.add_history_record('Some record 2 to commenet DCRange')
    ro1.add_par('par-1-in-DCRange', 3)
    ro1.add_par('par-2-in-DCRange', 'some string 3')
    ro1.add_par('par-3-in-DCRange', 3.3)

    vo1 = ro2.add_version()
    vo2 = ro2.add_version()
    vo1.add_history_record('Some record 1 to commenet DCVersion')
    vo1.add_history_record('Some record 2 to commenet DCVersion')
    vo1.add_history_record('Some record 3 to commenet DCVersion')
    vo1.add_history_record('Some record 4 to commenet DCVersion')
    vo1.add_par('par-1-in-DCVersion', 4)
    vo1.add_par('par-2-in-DCVersion', 'some string 4')
    vo1.add_par('par-3-in-DCVersion', 4.4)

    #ro2.set_vnum_def(vo2.vnum())

    vo1.set_tsprod(time())
    vo1.add_data(np.zeros((32,185,388)))

    vo2.set_tsprod(time())
    vo2.add_data(np.ones((32,185,388)))

    o.print_obj()
    o.save()

#------------------------------

def test_DCStore_load() :

    import numpy as np

    o = DCStore('cspad-654321.h5')
    o.load()

    print 50*'_','\ntest o.print()' 
    o.print_obj()

#------------------------------

def test_DCStore_load_and_save() :

    import numpy as np

    o = DCStore('cspad-654321.h5')
    o.load()

    print 50*'_','\ntest o.print()' 
    o.print_obj()

    print 50*'_','\n test o.save(fname)' 
    o.save('cspad-re-loaded.h5')

#------------------------------

def test() :
    log.setPrintBits(0377) 
    if len(sys.argv)==1    : print 'For test(s) use command: python %s <test-number=1-4>' % sys.argv[0]
    elif(sys.argv[1]=='1') : test_DCStore()        
    elif(sys.argv[1]=='2') : test_DCStore_save()        
    elif(sys.argv[1]=='3') : test_DCStore_load()        
    elif(sys.argv[1]=='4') : test_DCStore_load_and_save()        
    else : print 'Non-expected arguments: sys.argv = %s use 1,2,...' % sys.argv

#------------------------------

if __name__ == "__main__" :
    test()
    sys.exit( 'End of %s test.' % sys.argv[0])

#------------------------------
