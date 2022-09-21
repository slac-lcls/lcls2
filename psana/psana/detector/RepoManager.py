
"""
:py:class:`RepoManager`
=======================

Usage::

      from psana.detector.UtilsCalib import RepoManager
      repoman = RepoManager(dirrepo, dirmode=0o777, filemode=0o666)
      d = repoman.dir_logs()
      d = repoman.makedir_logs()
      f = repoman.logname_at_start(fname)

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2022-01-20 by Mikhail Dubrovin
"""
import os
import psana.detector.Utils as ut
from psana.detector.dir_root import DIR_LOG_AT_START
# DIR_LOG_AT_START = DIR_ROOT + '/detector/logs/atstart'


class RepoManager(object):
    """Supports repository directories/files naming structure
       <dirrepo>/<panel_id>/<constant_type>/<files-with-constants>
       <dirrepo>/logs/<year>/<log-files>
       <dirrepo>/logs/log-<fname>-<year>.txt # file with log_rec_at_start()
       e.g.: dirrepo = DIR_ROOT + '/detector/gains/epix10k/panels'
       DIR_LOG_AT_START/<year>/<year>_lcls2_<procname>.txt
    """

    def __init__(self, dirrepo, **kwa):
        self.dirrepo = dirrepo.rstrip('/')
        self.dirmode     = kwa.get('dirmode',  0o777)
        self.filemode    = kwa.get('filemode', 0o666)
        self.umask       = kwa.get('umask', 0o0)
        self.dirname_log = kwa.get('dirname_log', 'logs')
        self.year        = kwa.get('year', ut.str_tstamp(fmt='%Y'))
        self.tstamp      = kwa.get('tstamp', ut.str_tstamp(fmt='%Y-%m-%dT%H%M%S'))
        self.dettype     = kwa.get('dettype', None)
        self.addname     = kwa.get('addname', 'lcls2')  # 'logrec'
        if self.dettype is not None: self.dirrepo += '/%s' % self.dettype
        self.dir_log_at_start = kwa.get('dir_log_at_start', DIR_LOG_AT_START)


    def makedir(self, d):
        """create and return directory d with mode defined in object property"""
        ut.create_directory(d, self.dirmode, umask=self.umask)
        return d


    def dir_in_repo(self, name):
        """return directory <dirrepo>/<name>"""
        return os.path.join(self.dirrepo, name)


    def makedir_in_repo(self, name):
        """create and return directory <dirrepo>/<name>"""
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_in_repo(name))


    def dir_logs(self):
        """return directory <dirrepo>/logs"""
        return self.dir_in_repo(self.dirname_log)


    def makedir_logs(self):
        """create and return directory <dirrepo>/logs"""
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_logs())


    def dir_logs_year(self):
        """return directory <dirrepo>/logs/<year>"""
        return os.path.join(self.dir_logs(), self.year)


    def makedir_logs_year(self):
        """create and return directory <dirrepo>/logs/<year>"""
        d = self.makedir_logs()
        return self.makedir(self.dir_logs_year())


    def dir_merge(self, dname='merge_tmp'):
        d = self.makedir(self.dirrepo)
        return self.dir_in_repo(dname)


    def makedir_merge(self, dname='merge_tmp'):
        return self.makedir(self.dir_merge(dname))


    def dir_panel(self, panel_id):
        """returns path to panel directory like <dirrepo>/<panel_id>"""
        return os.path.join(self.dirrepo, panel_id)


    def makedir_panel(self, panel_id):
        """create and returns path to panel directory like <dirrepo>/<panel_id>"""
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_panel(panel_id))


    def dir_type(self, panel_id, ctype): # ctype='pedestals'
        """returns path to the directory like <dirrepo>/<panel_id>/<ctype>
        """
        return '%s/%s' % (self.dir_panel(panel_id), ctype)


    def makedir_type(self, panel_id, ctype): # ctype='pedestals'
        """create and returns path to the directory like <dirrepo>/<panel_id>/<ctype>"""
        d = self.makedir_panel(panel_id)
        return self.makedir(self.dir_type(panel_id, ctype))


    def dir_types(self, panel_id, subdirs=('pedestals', 'rms', 'status', 'plots')):
        """define structure of subdirectories in calibration repository under <dirrepo>/<panel_id>/..."""
        return ['%s/%s'%(self.dir_panel(panel_id), name) for name in subdirs]


    def makedir_types(self, panel_id, subdirs=('pedestals', 'rms', 'status', 'plots')):
        """create structure of subdirectories in calibration repository under <dirrepo>/<panel_id>/..."""
        dp = self.makedir_panel(panel_id)
        dirs = self.dir_types(panel_id, subdirs=subdirs)
        for d in dirs: self.makedir(d)
        return dirs


    def dir_constants(self, dname='constants'):
        """returns path to the directory like <dirrepo>/<logs>/<year>/<constants>"""
        return os.path.join(self.dirrepo, dname)


    def makedir_constants(self, dname='constants'):
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_constants(dname))


    def logname(self, procname):
        return '%s/%s_log_%s.txt' % (self.makedir_logs_year(), self.tstamp, procname)


    def dir_log_at_start_year(self):
        """return directory <dirlog_at_start>/<year>"""
        return os.path.join(self.dir_log_at_start, self.year)


    def makedir_log_at_start_year(self):
        """create and return directory"""
        return self.makedir(self.dir_log_at_start_year())


    def logname_at_start(self, procname):
        """ex.: <DIR_ROOT>/detector/logs/atstart/2022/2022_lcls2_calibman.txt"""
        return '%s/%s_%s_%s.txt' % (self.makedir_log_at_start_year(), self.year, self.addname, procname)


    def save_record_at_start(self, procname, tsfmt='%Y-%m-%dT%H:%M:%S%z'):
        ut.save_record_at_start(self, procname, tsfmt=tsfmt)


if __name__ == "__main__":
    dirrepo = './work'
    fname = 'testfname'
    procname = 'testproc-%s' % ut.get_login()
    repoman = RepoManager(dirrepo, dirmode=0o777, filemode=0o666)
    print('makedir_logs %s' % repoman.makedir_logs())
    print('logname %s' % repoman.logname(procname))
    print('makedir_constants %s' % repoman.makedir_constants())
    print('logname_at_start %s' % repoman.logname_at_start(fname))
    repoman.save_record_at_start('test_of_RepoManager')

# EOF
