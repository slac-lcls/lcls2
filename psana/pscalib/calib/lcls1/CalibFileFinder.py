#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`CalibFileFinder` is a python version of CalibFileFinder.cpp - finds calibration file
=====================================================================================================

Usage::

    from PSCalib.CalibFileFinder import CalibFileFinder

    cdir  = '/reg/d/psdm/CXI/cxi83714/calib/'
    group = 'CsPad::CalibV1'   # optional parameter, if not available will be set for src from dict 
    src   = 'CxiDs1.0:Cspad.0'
    type  = 'pedestals'
    rnum  = 137

    cff = CalibFileFinder(cdir, group, pbits=0377)
    #OR
    cff = CalibFileFinder(cdir)
    fname = cff.findCalibFile(src, type, rnum)

    fname_new = cff.makeCalibFileName(src, type, run_start, run_end=None)

    #-----------------------------------------------
    # ALTERNATIVE usage of direct access methods

    from PSCalib.CalibFileFinder import find_calib_file, make_calib_file_name

    fname_existing = find_calib_file(cdir, src, type, rnum, pbits=1)
    fname_new      = make_calib_file_name(cdir, src, type, run_start, run_end=None, pbits=1)

    #-----------------------------------------------
    # Deploy file or numpy array as a file in the calibration store

    # use optional dictionary of comments to save in the HISTORY and file
    arr = np.ones((32,185,388))
    cmts = {'exp':'cxi12345', 'ifname':'input-file-name', 'app':'my-app-name', 'comment':'my-comment'}
    deploy_calib_array(cdir, src, type, run_start, run_end, arr, cmts, fmt='%.1f', pbits=1)

    cmts = {'exp':'cxi12345', 'app':'my-app-name', 'comment':'my-comment'}
    ifname='path-to-my-own-calibtation-file/file-name.txt'
    deploy_calib_file(cdir, src, type, run_start, run_end, ifnameq, cmts, pbits=1)

See :py:class:`GlobalUtils`, :py:class:`NDArrIO`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""
#------------------------------

import os
import sys
import PSCalib.GlobalUtils as gu
from PSCalib.NDArrIO import save_txt
import tempfile

#------------------------------

class CalibFile :

    rnum_max = 9999

    def __init__(self, path='', pbits=1) :
        self.path = path
        self.pbits = pbits
        
        fname = os.path.basename(path)
        basename = os.path.splitext(fname)[0]

        if not ('-' in basename) :
            self.set_invalid('WARNING! INVALID CALIBRATION FILE NAME "%s" - missing dash' % basename)
            return
            
        fields = basename.split('-')
        if len(fields) != 2 :
            self.set_invalid('WARNING! INVALID CALIBRATION FILE NAME "%s" - wrong number of dases' % basename)
            return

        begin, end = fields

        if begin.isdigit() :
            self.begin = int(begin)
            if self.begin>self.rnum_max : 
                self.set_invalid('WARNING! INVALID CALIBRATION FILE NAME "%d" - begin value is too big' % begin)
                return
        else : 
            self.valid = False
            return

        if end.isdigit() :
            self.end = int(end)
            if self.end>self.rnum_max :
                self.set_invalid('WARNING! INVALID CALIBRATION FILE NAME "%d" - end value is too big' % begin)
                return
        elif end == 'end' :
            self.end = self.rnum_max
        else :
            self.set_invalid('WARNING! INVALID CALIBRATION FILE NAME "%d" - end value is not recognized' % begin)
            return

        self.valid = True

    def get_path(self) :
        return self.path

    def get_begin(self) :
        return self.begin

    def get_end(self) :
        return self.end

    def set_invalid(self, msg) :
        if self.pbits : print msg
        self.valid = False

    def __cmp__(self, other) :        
        #if self.begin != other.begin : return self.begin < other.begin
        #return self.end > other.end

        if   self.begin < other.begin : return -1
        elif self.begin > other.begin : return  1
        else :
            if   self.end > other.end : return -1
            elif self.end < other.end : return  1
            else : return 0

    def str_attrs(self) : 
        return 'begin: %4d  end: %4d  path: %s' % (self.begin, self.end, self.path)

#------------------------------

def find_calib_file(cdir, src, type, rnum, pbits=1) :
    return CalibFileFinder(cdir, pbits=pbits).findCalibFile(src, type, rnum)

#------------------------------

def make_calib_file_name(cdir, src, type, run_start, run_end=None, pbits=1) :
    return CalibFileFinder(cdir, pbits=pbits).makeCalibFileName(src, type, run_start, run_end=None)

#------------------------------

def deploy_calib_array(cdir, src, type, run_start, run_end=None, arr=None, dcmts={}, fmt='%.1f', pbits=1) :
    """Deploys array in calibration file

       - makes the new file name using make_calib_file_name(...)
       - if file with this name already exists - rename it with current timestamp in the name
       - save array in file
       - add history record
    """

    fname = make_calib_file_name(cdir, src, type, run_start, run_end, pbits)
    path_history = '%s/HISTORY'%os.path.dirname(fname)

    if os.path.exists(fname) :
        fname_bkp = '%s-%s'%(fname, gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S'))
        os.system('cp %s %s'%(fname, fname_bkp))
        if pbits & 1 :
            print 'Existing file %s\nis backed-up  %s' % (fname, fname_bkp)

    # extend dictionary for other parameters
    d = dict(dcmts)
    d['run']   = run_start
    d['fname'] = os.path.basename(fname)
    d['src']   = src
    d['ctype'] = type

    # make list of comments
    cmts=['%s %s'%(k.upper().ljust(11),v) for k,v in d.iteritems()]
    
    # save n-dimensional numpy array in the tmp text file
    fntmp = tempfile.NamedTemporaryFile(mode='r+b',suffix='.data')
    if pbits & 2 : print 'Save constants in tmp file: %s' % fntmp.name
    save_txt(fntmp.name, arr, cmts, fmt='%.1f')

    if pbits & 1 : print 'Deploy constants in file: %s' % fname
    # USE cat in stead of cp and move in order to create output file with correct ACL permissions
    cmd_cat = 'cat %s > %s' % (fntmp.name, fname)    
    #os.system(cmd_cat)
    stream = os.popen(cmd_cat)
    resp = stream.read()
    msg = 'Command: %s\n - resp: %s' % (cmd_cat, resp)
    if pbits & 2 : print msg

    # add record to the HISTORY file
    hrec = _history_record(d)
    if pbits & 1 : print 'Add record: %sto the file: %s' % (hrec, path_history)
    gu.save_textfile(hrec, path_history, mode='a')

#------------------------------

def deploy_calib_file(cdir, src, type, run_start, run_end=None, ifname='', dcmts={}, pbits=1) :
    """Deploys calibration file

       - makes the new file name using make_calib_file_name(...)
       - if file with this name already exists - rename it with current timestamp in the name
       - save array in file
       - add history record
    """

    fname = make_calib_file_name(cdir, src, type, run_start, run_end, pbits)
    path_history = '%s/HISTORY'%os.path.dirname(fname)

    if os.path.exists(fname) :
        fname_bkp = '%s-%s'%(fname, gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S'))
        os.system('cp %s %s'%(fname, fname_bkp))
        if pbits & 1 :
            print 'Existing file %s\nis backed-up  %s' % (fname, fname_bkp)

    # extend dictionary for other parameters
    d = dict(dcmts)
    d['run']   = run_start
    d['fname'] = os.path.basename(fname)
    d['ifname']= ifname
    d['src']   = src
    d['ctype'] = type

    if pbits & 1 : print 'Deploy constants in file: %s' % fname
    # USE cat in stead of cp and move in order to create output file with correct ACL permissions
    cmd_cat = 'cat %s > %s' % (ifname, fname)    
    #os.system(cmd_cat)
    stream = os.popen(cmd_cat)
    resp = stream.read()
    msg = 'Command: %s\n - resp: %s' % (cmd_cat, resp)
    if pbits & 2 : print msg

    # add record to the HISTORY file
    hrec = _history_record(d)
    if pbits & 1 : print 'Add record: %sto the file: %s' % (hrec, path_history)
    gu.save_textfile(hrec, path_history, mode='a')

#------------------------------

def _history_record(dcmts) :
    """Returns history record made of dictionary comments and system info
    """
    user   = gu.get_login()
    host   = gu.get_hostname()
    tstamp = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S  zone:%Z')
    rnum   = '%04d' % dcmts.get('run')
    exp    = '%s' % dcmts.get('exp')
    ifname = '%s' % dcmts.get('ifname')
    ofname = '%s' % dcmts.get('fname')
    app    = '%s' % dcmts.get('app')
    cmt    = '%s' % dcmts.get('comment')

    return 'file:%s  copy_of:%s  exp:%s  run:%s  app:%s  user:%s  host:%s  cptime:%s  comment:%s\n' % \
          (ofname.ljust(14),
           ifname,
           exp.ljust(8),
           rnum.ljust(4),
           app.ljust(10),
           user,
           host,
           tstamp.ljust(29),
           cmt)
   
#------------------------------

class CalibFileFinder :

    def __init__(self, cdir='', group='', pbits=1) :
        self.cdir  = cdir
        self.group = group
        self.pbits = pbits


    def _setGroup(self, src) :
        """If not available, sets group from source.
        """
        if self.group == '' or self.group is None :
            dettype = gu.det_type_from_source(src)
            self.group = gu.dic_det_type_to_calib_group.get(dettype)
            if self.group is None :
                if self.pbits & 1 : print 'WARNING! CALIBRATION GROUP IS NOT FOUND FOR SOURCE %s' % src
                return False
        return True


    def makeCalibFileName(self, src, type, run_start, run_end=None) :
        """Returns calibration file name.
        """
        if os.path.basename(self.cdir.rstrip('/')) != 'calib' :
            if self.pbits & 1  : print 'WARNING! NOT calib DIRECTORY: %s' % self.cdir
            return None

        # there have been problems with calib-dir mounts on the mon nodes.
        # raise an exception here to try to detect this problem
        #assert os.path.isdir(self.cdir), 'psana calib-dir must exist: '+self.cdir

        if not os.path.isdir(self.cdir) :
            print 'WARNING! psana calib-dir is not found: %s' % self.cdir
            return None

        if not self._setGroup(src) :
            return None

        if run_start < 0 :
            if self.pbits & 1  : print 'WARNING! START RUN NUMBER IS NEGATIVE: %d' % run_start
            return None

        if run_start > 9999 :
            if self.pbits & 1  : print 'WARNING! START RUN NUMBER EXCEEDS 4-DIGITS: %d' % run_start
            return None

        if run_end is None :
            self.cfname = '%d-end.data' % (run_start)
            
        else :

          if run_end < 0 :
            if self.pbits & 1  : print 'WARNING! END RUN NUMBER IS NEGATIVE: %d' % run_end
            return None

          if run_end > 9999 :
            if self.pbits & 1  : print 'WARNING! END RUN NUMBER IS TOO BIG: %d' % run_end
            return None

          if run_end < run_start :
            if self.pbits & 1  : print 'WARNING! END RUN:%d < START RUN:%d' % (run_end, run_start)
            return None

          self.cfname = '%d-%d.data' % (run_start, run_end) 

        dir = self.cdir
        for subdir in (self.group, src, type) :
            dir = os.path.join(dir, subdir)
            gu.create_directory(dir, self.pbits)

        return os.path.join(dir, self.cfname)


    def findCalibFile(self, src, type, rnum0) :
        """Find calibration file.
        """
        rnum = rnum0 if rnum0 <= CalibFile.rnum_max else CalibFile.rnum_max

        # there have been problems with calib-dir mounts on the mon nodes.
        # raise an exception here to try to detect this problem
        #assert os.path.isdir(self.cdir), 'psana calib-dir must exist: '+self.cdir
        if not os.path.isdir(self.cdir) :
            print 'WARNING! psana calib-dir is not found: %s' % self.cdir
            return None

        if not self._setGroup(src) : return ''

        dir_name = os.path.join(self.cdir, self.group, src, type)
        if not os.path.exists(dir_name) :
            if self.pbits & 1  : print 'WARNING! NON-EXISTENT DIR: %s' % dir_name
            return ''

        fnames = os.listdir(dir_name)
        files = [os.path.join(dir_name,fname) for fname in fnames]
        return self.selectCalibFile(files, rnum) 


    def selectCalibFile(self, files, rnum) :
        """Selects calibration file from a list of file names
        """
        if self.pbits & 1024 : print '\nUnsorted list of *.data files in the calib directory:'
        list_cf = []
        for path in files : 
           fname = os.path.basename(path)

           if fname is 'HISTORY' : continue
           if os.path.splitext(fname)[1] != '.data' : continue

           cf = CalibFile(path)
           if cf.valid :
               if self.pbits & 1024 : print cf.str_attrs()
               list_cf.append(cf)
           
        # sotr list
        list_cf_ord = sorted(list_cf)
        
        # print entire sorted list
        if self.pbits & 4 :
            print '\nSorted list of *.data files in the calib directory:'
            for cf in list_cf_ord[::-1] :
                if self.pbits & 4 : print cf.str_attrs()

        # search for the calibration file
        for cf in list_cf_ord[::-1] :
            if cf.get_begin() <= rnum and rnum <= cf.get_end() :
                if self.pbits & 8 :
                    print 'Select calib file: %s' % cf.get_path()
                return cf.get_path()

        # if no matching found
        return ''

#----------------------------------------------

def test01() :

    # assuming /reg/d/psdm/CXI/cxid2714/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/pedestals/15-end.data

    #cdir  = '/reg/d/psdm/CXI/cxid2714/calib/'
    #cdir  = '/reg/d/psdm/CXI/cxi80410/calib/'
    cdir  = '/reg/d/psdm/CXI/cxi83714/calib/'

    group = 'CsPad::CalibV1'
    src   = 'CxiDs1.0:Cspad.0'
    type  = 'pedestals'
    rnum  = 134
    #rnum  = 123456789

    #--------------------------

    print 80*'_', '\nTest 1'
    print 'Finding calib file for\n  dir = %s\n  grp = %s\n  src = %s\n  type= %s\n  run = %d' % \
          (cdir, group, src, type, rnum)

    cff = CalibFileFinder(cdir, group, 0377)
    fname = cff.findCalibFile(src, type, rnum)

    #--------------------------

    print 80*'_', '\nTest 2'
    print 'Test methods find_calib_file and make_calib_file_name'
    fname_existing = find_calib_file(cdir, src, type, rnum, pbits=1)
    print '  fname_existing : %s' % fname_existing

    cdir = './calib'
    run_start = 134
    gu.create_directory(cdir, True)
    fname_new      = make_calib_file_name(cdir, src, type, run_start, run_end=None, pbits=0)
    print '  fname_new      : %s' % fname_new

#--------------------------

def test_deploy_calib_array() :
    print 80*'_', '\nTest deploy_calib_array'

    cdir  = './calib'
    if not os.path.exists(cdir) : gu.create_directory(cdir, verb=True)
    #cdir  = '/reg/d/psdm/CXI/cxi83714/calib'

    src   = 'CxiDs1.0:Cspad.0'
    type  = 'pedestals'
    run_start  = 9991
    run_end    = None
    arr= gu.np.ones((32,185,388))
    cmts = {'exp':'cxi83714', 'ifname':'input-file-name', 'app':'my-app-name', 'comment':'my-comment'}
    deploy_calib_array(cdir, src, type, run_start, run_end, arr, cmts, fmt='%.1f', pbits=3)

#--------------------------

def test_deploy_calib_file() :
    print 80*'_', '\nTest deploy_calib_file'
    cdir  = './calib'
    if not os.path.exists(cdir) : gu.create_directory(cdir, verb=True)
    #cdir  = '/reg/d/psdm/CXI/cxi83714/calib'
    
    src   = 'CxiDs1.0:Cspad.0'
    type  = 'geometry'
    run_start  = 9992
    run_end    = None
    fname = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-camera1-2014-09-24/2016-06-15-geometry-cxil0216-r150-camera1-z95mm.txt'
    cmts = {'exp':'cxi83714', 'app':'my-app-name', 'comment':'my-comment'}
    deploy_calib_file(cdir, src, type, run_start, run_end, fname, cmts, pbits=3)

#--------------------------

if __name__ == "__main__" :

    if len(sys.argv)<2    : test01()
    elif sys.argv[1]=='1' : test01() 
    elif sys.argv[1]=='2' : test01() 
    elif sys.argv[1]=='3' : test_deploy_calib_array() 
    elif sys.argv[1]=='4' : test_deploy_calib_file() 
    else : print 'Non-expected arguments: sys.argv=', sys.argv

    sys.exit('End of %s' % sys.argv[0])

#----------------------------------------------
