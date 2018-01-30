####!/usr/bin/env python
#------------------------------
"""
Class :py:class:`DCBase` is a base class for the Detector Calibration (DC) project
==================================================================================

Usage::

    # Import
    from PSCalib.DCBase import DCBase

    o = DCBase()

    # Dictionary of parameters
    # ========================
    
    o.set_pars_dict(d)                   # set (dict) dictionary of pars.
    o.add_par(k,v)                       # add (k,v) par to the dictionary of pars.
    o.del_par(k)                         # delete par with key k. 
    o.clear_pars()                       # delete all pars from the dictionary.
    d = o.pars_dict()                    # returns (dict) dictionary of pars.
    p = o.par(k)                         # returns par value for key k.
    t = o.pars_text()                    # returns (str) text of all pars.
    
    # History records
    # ===============
    
    o.set_history_dict(d)                # set (dict) dictionary of history from specified dictionary
    o.add_history_record(rec, tsec=None) # add (str) record with (int) time[sec] to the history dictionary of (tsec:rec).
                                         # If tsec is None - current time is used as a key.
    o.del_history_record(tsec)           # Delete one history record from the dictionary by its time tsec.
    o.clear_history()                    # Delete all history records from the dictionary.
    d = o.history_dict()                 # returns (dict) history dictionary associated with current object .
    r = o.history_record(tsec)           # returns (str) history record for specified time tsec.
    t = o.history_text(tsfmt=None)       # returns (str) all history records preceded by the time stamp as a text.
    
    # Save and Load
    # =============
    
    o.save_history_file(path='history.txt', verb=False) # save history in the text file
    o.load_history_file(path='history.txt', verb=False) # load history from the text file
    
    o.save_base(grp)                     # save everything in hdf5 group
    o.load_base(name, grp)               # load from hdf5 group
    
    # Time convertors
    # ===============
    
    t_str = o.tsec_to_tstr(tsec, tsfmt=None) # converts (float) time[sec] to the (str) time stamp
    t_sec = o.tstr_to_tsec(tstr, tsfmt=None) # converts (str) time stamp to (float) time[sec]

See:
    * :py:class:`DCStore`
    * :py:class:`DCType`
    * :py:class:`DCRange`
    * :py:class:`DCVersion`
    * :py:class:`DCBase`
    * :py:class:`DCInterface`
    * :py:class:`DCUtils`
    * :py:class:`DCDetectorId`
    * :py:class:`DCConfigParameters`
    * :py:class:`DCFileName`
    * :py:class:`DCLogger`
    * :py:class:`DCMethods`
    * :py:class:`DCEmail`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""
#------------------------------

from time import time, sleep, localtime, gmtime, strftime, strptime, mktime
from math import floor
from PSCalib.DCLogger import log
from PSCalib.DCUtils import get_subgroup, save_object_as_dset

#------------------------------

#class DCBase() :
class DCBase(object) :
    """Base class for the Detector Calibration (DC) project.

       Parameters
       
       cmt : str - string of comment associated with derived class object.
    """
    _tsfmt = '%Y-%m-%dT%H:%M:%S'
    _offspace = '    '


    def __init__(self, cmt=None) :
        self._name = 'DCBase'
        self._dicpars = {}
        self._dichist = {}
        msg = 'In c-tor %s' % self._name
        log.debug(msg, self._name)
        self._grp_pars_name = '_parameters'
        self._grp_history_name = '_history'
        self._tsec_old = None
        self._lst_del_keys = []

        if cmt is not None : self.add_history_record(cmt)


    def __del__(self) :
        self._dicpars.clear()


    def set_pars_dict(self, d) :
        self._dicpars.clear()
        for k,v in d.items() :
            self._dicpars[k] = v
    

    def add_par(self, k, v) :
        self._dicpars[k] = v


    def del_par(self, k) :
        if k in self._dicpars.keys() : del self._dicpars[k]
        

    def clear_pars(self) :
        self._dicpars.clear()


    def pars_dict(self) :
        return self._dicpars if len(self._dicpars)>0 else None


    def par(self, k ) :
        return self._dicpars.get(k, None)


    def pars_text(self) :
        return ', '.join(['(%s : %s)' % (str(k), str(v)) for k,v in self._dicpars.items()])


    def set_history_dict(self, d) :
        self._dichist.clear()
        for k,v in d.items() :
            self._dichist[k] = v


    def add_history_record(self, rec, tsec=None) :
        t_sec = tsec
        if t_sec is None :
            t_sec = time()
            if t_sec == self._tsec_old : # make sure that all records have different keys
                t_sec += 0.001
                self._tsec_old = t_sec
        self._dichist[t_sec] = rec

        #sleep(1) # wait 1ms in order to get unique timestamp
        #print 'add recod in time = %.6f' % t_sec
        #log.debug('Add recod: %s with time:  %.6f' % (rec, t_sec), self._name)


    def del_history_record(self, k) :
        if k in self._dichist.keys() : del self._dichist[k]
        

    def clear_history(self) :
        self._dichist.clear()


    def history_dict(self) :
        return self._dichist


    def history_record(self, tsec) :
        return self._dichist.get(tsec)


    def history_text(self, tsfmt=None) :
        """Returns (str) history records preceded by the time stamp as a text"""
        fmt = self._tsfmt if tsfmt is None else tsfmt
        return '\n'.join(['%s %s' % (self.tsec_to_tstr(ts), str(rec)) for ts,rec in sorted(self._dichist.items())])


    def save_history_file(self, path='history.txt', verb=False) :
        """Save history in the text file"""
        f = open(path,'w')
        f.write(self.history_text())
        f.close()
        if verb : 
            #print 'History records are saved in the file: %s' % path
            log.debug('History records are saved in the file: %s' % path, self._name)

    
    def load_history_file(self, path='history.txt', verb=False) :
        """Load history from the text file"""
        f = open(path,'r')
        lines = f.readlines()
        f.close()
        for line in lines :
            tsstr, rec = line.rstrip('\n').split(' ',1)
            #print 'XXX:', tsstr, rec            
            self._dichist[self.tstr_to_tsec(tsstr)] = rec
        if verb : 
            #print 'Read history records from the file: %s' % path
            log.debug('Read history records from the file: %s' % path, self._name)


    def _save_pars_dict(self, grp) :
        """Saves _dicpars in the h5py group"""
        if not self._dicpars : return # skip empty dictionary

        #grpdic = grp.create_group(self._grp_pars_name)
        grpdic = get_subgroup(grp, self._grp_pars_name)
        for k,v in self._dicpars.items() :
            ds = save_object_as_dset(grpdic, name=k, data=v)


    def _save_hystory_dict(self, grp) :
        """Saves _dichist in the h5py group"""
        if not self._dichist : return # skip empty dictionary

        #grpdic = grp.create_group(self._grp_history_name)
        grpdic = get_subgroup(grp, self._grp_history_name)
        for k,v in self._dichist.items() :
            #tstamp = str(self.tsec_to_tstr(k))
            tstamp = str('%.6f' % k)
            #print 'XXX:', tstamp, v
            ds = save_object_as_dset(grpdic, tstamp, data=v)
        #print 'In %s.save_hystory_dict(): group name=%s TBD: save parameters and hystory' % (self._name, grp.name)


    def save_base(self, grp) :
        self._save_pars_dict(grp)
        self._save_hystory_dict(grp)


    def group_name(self, grp) : 
        return grp.name.rsplit('/')[-1]


    def is_base_group(self, name, grp) : 
        return self.load_base(name, grp)


    def load_base(self, name, grp) :
        grpname = name # self.group_name()

        if grpname == self._grp_pars_name :
            self._load_pars_dict(grp)
            return True
        elif grpname == self._grp_history_name :
            self._load_hystory_dict(grp)
            return True
        return False
        

    def _load_pars_dict(self, grp) :
        log.debug('_load_pars_dict for group %s' % grp.name, self._name)
        self.clear_pars()
        for k,v in dict(grp).iteritems() :
            log.debug('par: %s = %s' % (k, str(v[0])), self._name)
            self.add_par(k, v[0])


    def _load_hystory_dict(self, grp) :
        log.debug('_load_hystory_dict for group %s' % grp.name, self._name)
        self.clear_history()
        
        for k,v in zip(grp.keys(), grp.values()) :
            #print '             YYY:k,v:', k,v 
            tsec = float(k)
            rec = 'None' if v is None else v[0]
            log.debug('t: %.6f rec: %s' % (tsec, rec), self._name)
            self.add_history_record(rec, tsec) # tsec=self.tstr_to_tsec(k)


    def tsec_to_tstr(self, tsec, tsfmt=None, addfsec=True) :
        """converts float tsec like 1471035078.908067 to the string 2016-08-12T13:51:18.908067"""
        fmt = self._tsfmt if tsfmt is None else tsfmt
        itsec = floor(tsec)
        strfsec = ('%.6f' % (tsec-itsec)).lstrip('0') if addfsec else ''
        return '%s%s' % (strftime(fmt, localtime(itsec)), strfsec)


    def tstr_to_tsec(self, tstr, tsfmt=None) :
        """converts string tstr like 2016-08-12T13:51:18.908067 to the float time in seconds 1471035078.908067"""
        #t0_sec=time()
        fmt = self._tsfmt if tsfmt is None else tsfmt
        ts, fsec = tstr.split('.')
        return mktime(strptime(ts, fmt)) + 1e-6*int(fsec)
        #print 'tstr_to_tsec consumed time (sec) =', time()-t0_sec
        #return t_sec


    def print_base(self, offset='  ') :
        """Print content of dictionaries of parameters and history"""
        print '%s %s'             % (offset, self._name)

        if len(self._dicpars) : print '%s Parameters:' % offset
        for k,v in self._dicpars.items() :
            print '  %s par: %20s  value: %s' % (offset, k, str(v))

        if len(self._dichist) : print '%s History:' % offset
        for k,v in sorted(self._dichist.items()) :
            #print '%s t[sec]: %d: %s rec: %s' % (offset, floor(k), self.tsec_to_tstr(k), str(v))
            print '  %s %s %s' % (offset, self.tsec_to_tstr(k, addfsec=False), str(v))

#------------------------------

    def make_record(self, action='', key='', cmt=False) :
        """Returns string record combined with comment.
        
        Parameters
        
        action : str - description of method action,
        key    : str - key for hdf5 group or dataset name,
        cmt    : str/None/False - additional comment or no-comment: False is used to turn off history record, None - no-comment.
        """
        if cmt is None or cmt is False : return '%s %s' % (action, key)
        return '%s %s: %s' % (action, key, cmt)

#------------------------------

def test_pars() :
    o = DCBase()
    d = {1:'10', 2:'20', 3:'30'}
    o.set_pars_dict(d)
    print '\nTest pars: %s' % o.pars_text()
    o.del_par(2)
    print '\nAfter del_par(2): %s' % o.pars_text()
    print '\npar(3): %s' % o.par(3)

#------------------------------

def test_history() :
    o = DCBase()
    o.add_history_record('rec 01')
    o.add_history_record('rec 02')
    o.add_history_record('rec 03')
    o.add_history_record('rec 04')
    o.add_history_record('rec 05')
    o.add_history_record('rec 06')
    print '\nTest history records:\n%s' % o.history_text()
    o.save_history_file('history-test.txt', verb=True)
    o.add_history_record('rec 07')

    o.load_history_file('history-test.txt', verb=True)
    print '\nTest history records:\n%s' % o.history_text()

#------------------------------

def test_time_converters() :
    o = DCBase()
    t_sec  = time()
    t_str  = o.tsec_to_tstr(t_sec, tsfmt=None) 
    t_sec2 = o.tstr_to_tsec(t_str, tsfmt=None)
    print 'convert time     %.6f to time stamp: %s' % (t_sec,  t_str)
    print 'and back to time %.6f' % (t_sec2)

#------------------------------

def test_make_record() :
    o = DCBase()
    print o.make_record(action='test make_record for cmt=False', key='keyword', cmt=False)
    print o.make_record(action='test make_record for cmt=None', key='keyword', cmt=None)
    print o.make_record(action='test make_record for cmt="my comment"', key='keyword', cmt="my comment")

#------------------------------

if __name__ == "__main__" :
    import sys
    test_pars()
    test_history()
    test_time_converters()
    test_make_record()
    sys.exit('End of %s test.' % sys.argv[0])

#------------------------------
