#------------------------------

"""Class helps to save text file with information about peaks found in peak-finders. 

Usage::

    # Imports
    from pyimgalgos.PeakStore import PeakStore

    # Usage with psana types env and evt

    pstore = PeakStore(env, 5, prefix='xxx', add_header='TitV1 TitV2 TitV3 ...', pbits=255)
    for peak in peaks :
        rec = '%s %d %f ...' % (peak[0], peak[5], peak[7],...)
        pstore.save_peak(evt, rec)
    pstore.close() # is done by default in destructor


    # Usage without psana types

    pstore = PeakStore('expNNNNN', 5, prefix='xxx', add_header='TitV1 TitV2 TitV3 ...', pbits=255)
    for peak in peaks :
        rec = '%s %d %f ...' % (peak[0], peak[5], peak[7],...)
        pstore.save_peak(peak_rec=rec)

    # Print methods
    pstore.print_attrs()

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------

import numpy as np
from time import strftime, localtime #, gmtime

#------------------------------

class PeakStore :

    def __init__(self, par='cxiNNNNN', runnum=0, prefix=None, header=None, add_header='Evnum etc...' , pbits=0) :  
        """Constructor parameters are used to generate file name, set header and object verbosity.

        @param par - (str) experiment name or (psana) env to form default output file name. Ex.; 'cxiNNNNN' or env 
        @param runnum - (int) run number to form default output file name. 
        @param prefix - (str) output file name prefix
        @param header - (str) initial part of the record header (w/o #), by default it is
                        '# Exp     Run  Date       Time      time(sec)   time(nsec) fiduc'
                        and these values are retreived from (psana) evt
        @param add_header - (str) additional part of the record header 
        @param pbits  - (int) print control bit-word
        """
        self.fout   = None
        self.pbits  = pbits
        self.exp    = par if isinstance(par,str) else par.experiment()
        self.runnum = runnum 
        self.set_file_name(prefix)
        self.set_header(header, add_header)
        self.open_file()
        self.counter = 0
        self.psana  = None

##-----------------------------

    def __del__(self) :
        """Destructor closes file
        """
        self.close_file()

##-----------------------------

    def print_attrs(self) :
        msg = 'Attributes of %s' % self.__class__.__name__ +\
              '\n prefix: %s' % str(self.prefix) +\
              '\n fname: %s' % str(self.fname) +\
              '\n title: %s' % self.header
        print msg

##-----------------------------

    def set_file_name(self, prefix=None, fmt='%Y-%m-%d') : # fmt='%Y-%m-%dT%H:%M:%S'
        """Sets the name of the file with peak info
        """
        self.prefix = prefix        
        if prefix is None : prefix='peaks'
        tstamp = strftime(fmt, localtime())
        self.fname = '%s-%s-r%04d-%s.txt' % (self.prefix, self.exp, self.runnum, tstamp)
 
##-----------------------------

    def open_file(self) :  
        if self.fname is not None :
            self.fout = open(self.fname,'w')
            self.fout.write('%s\n' % self.header)
            if self.pbits & 1 : print 'Open output file with peaks: %s' % self.fname

##-----------------------------

    def close_file(self) :  
        self.fout.close()
        if self.pbits & 1 : print 'Close file %s with %d peaks' % (self.fname, self.counter)

##-----------------------------

    def set_header(self, header=None, add_header='add_header: Evnum etc...') :  
        """Returns a string of comments for output file with list of peaks
        """
        if header is None :
            self.header = '%s  %s' %\
                ('# Exp     Run  Date       Time      time(sec)   time(nsec) fiduc', add_header )
        else :
            self.header = '# %s %s' % (header, add_header )

        if self.pbits & 2 : print 'Hdr : %s' % (self.header)

##-----------------------------

    def rec_evtid(self, evt) :
        """Returns a string with event identidication info: exp, run, evtnum, tstamp, etc.
        """
        if self.psana is None :
            import psana
            self.psana = psana
 
        evtid = evt.get(self.psana.EventId)
        time_sec, time_nsec = evtid.time()
        tstamp = strftime('%Y-%m-%d %H:%M:%S', localtime(time_sec))
        return '%8s  %3d  %s  %10d  %9d  %6d' % \
               (self.exp, evtid.run(), tstamp, time_sec, time_nsec, evtid.fiducials())

#------------------------------
    
    def save_peak(self, evt=None, peak_rec='') :
        """Save event id with peak record (string) in the text file 
        """
        rec = peak_rec if evt is None else '%s %s' % (self.rec_evtid(evt), peak_rec)
        if self.fout is not None : self.fout.write('%s\n' % rec)
        self.counter += 1
        #self.evt_peaks.append(peak)
        if self.pbits & 2 : print '%7d: %s' % (self.counter, rec)
    
#------------------------------
    
    def save_comment(self, cmt='# default comment') :
        """Save (str) comment in the text file.  
        """
        rec = cmt if cmt[0] == '#' else '# %s' % (cmt)
        if self.fout is not None : self.fout.write('%s\n' % rec)
        if self.pbits & 2 : print rec

#------------------------------
#------------------------------
#----------  TEST  ------------
#------------------------------
#------------------------------

def test_PeakStore() :

    peaks = ((11,12,13,14,15),
             (21,22,23,24,25),
             (31,32,33,34,35))
 
    print '\nEXAMPLE #1'
    ps1 = PeakStore('expNNNNN', 5, prefix='xxx', header=None, add_header='TitV1 TitV2 TitV3 ...', pbits=255)
    for peak in peaks :
        rec = '%d %d %f ...' % (peak[0], peak[2], peak[3])
        ps1.save_peak(peak_rec=rec)


    print '\nEXAMPLE #2'
    ps2 = PeakStore('expNNNNN', 5, prefix='xxxxxx', header='V1 V2 V3 V4 V5', add_header='', pbits=255)
    for peak in peaks :
        rec = '%d %d %f ...' % (peak[0], peak[2], peak[3])
        ps2.save_peak(peak_rec=rec)

#------------------------------

if __name__ == "__main__" :

    test_PeakStore()

#------------------------------

