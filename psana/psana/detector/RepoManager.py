
"""
:py:class:`RepoManager`
=======================

Usage::

      from psana.detector.UtilsCalib import RepoManager
      repoman = RepoManager(dirrepo, dirmode=0o775, filemode=0o664)
      d = repoman.dir_logs()
      d = repoman.makedir_logs()
      f = repoman.logname_on_start(fname)

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2022-01-20 by Mikhail Dubrovin
"""

import os
from psana.pyalgos.generic.Utils import str_tstamp, logging, get_login
logger = logging.getLogger(__name__)


def file_mode(fname) :
    """Returns file mode
    """
    from stat import ST_MODE
    return os.stat(fname)[ST_MODE]


def create_directory(dir, mode=0o777, **kwa):
    """Creates directory and sets its mode"""
    if os.path.exists(dir):
        logger.debug('Exists: %s mode(oct): %s' % (dir, oct(file_mode(dir))))
    else:
        os.makedirs(dir)
        os.chmod(dir, mode)
        logger.debug('Created: %s, mode(oct)=%s' % (dir, oct(mode)))


class RepoManager(object):
    """Supports repository directories/files naming structure
       <dirrepo>/<panel_id>/<constant_type>/<files-with-constants>
       <dirrepo>/logs/<year>/<log-files>
       <dirrepo>/logs/log-<fname>-<year>.txt # file with log_rec_on_start()
       e.g.: dirrepo = '/reg/g/psdm/detector/gains/epix10k/panels'
    """

    def __init__(self, dirrepo, **kwa):
        self.dirrepo = dirrepo.rstrip('/')
        self.dirmode     = kwa.get('dirmode',  0o775)
        self.filemode    = kwa.get('filemode', 0o664)
        self.dirname_log = kwa.get('dirname_log', 'logs')


    def makedir(self, d):
        """create and return directory d with mode defined in object property
        """
        create_directory(d, self.dirmode)
        return d


    def dir_in_repo(self, name):
        """return directory <dirrepo>/<name>
        """
        return os.path.join(self.dirrepo, name)


    def makedir_in_repo(self, name):
        """create and return directory <dirrepo>/<name>
        """
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_in_repo(name))


    def dir_logs(self):
        """return directory <dirrepo>/logs
        """
        return self.dir_in_repo(self.dirname_log)


    def makedir_logs(self):
        """create and return directory <dirrepo>/logs
        """
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_logs())


    def dir_logs_year(self, year=None):
        """return directory <dirrepo>/logs/<year>
        """
        _year = str_tstamp(fmt='%Y') if year is None else year
        return os.path.join(self.dir_logs(), _year)


    def makedir_logs_year(self, year=None):
        """create and return directory <dirrepo>/logs/<year>
        """
        d = self.makedir_logs()
        return self.makedir(self.dir_logs_year(year))


    def dir_merge(self, dname='merge_tmp'):
        d = self.makedir(self.dirrepo)
        return self.dir_in_repo(dname)


    def makedir_merge(self, dname='merge_tmp'):
        return self.makedir(self.dir_merge(dname))


    def dir_panel(self, panel_id):
        """returns path to panel directory like <dirrepo>/<panel_id>
        """
        return os.path.join(self.dirrepo, panel_id)


    def makedir_panel(self, panel_id):
        """create and returns path to panel directory like <dirrepo>/<panel_id>
        """
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_panel(panel_id))


    def dir_type(self, panel_id, ctype): # ctype='pedestals'
        """returns path to the directory like <dirrepo>/<panel_id>/<ctype>
        """
        return '%s/%s' % (self.dir_panel(panel_id), ctype)


    def makedir_type(self, panel_id, ctype): # ctype='pedestals'
        """create and returns path to the directory like <dirrepo>/<panel_id>/<ctype>
        """
        d = self.makedir_panel(panel_id)
        return self.makedir(self.dir_type(panel_id, ctype))


    def dir_types(self, panel_id, subdirs=('pedestals', 'rms', 'status', 'plots')):
        """define structure of subdirectories in calibration repository under <dirrepo>/<panel_id>/...
        """
        return ['%s/%s'%(self.dir_panel(panel_id), name) for name in subdirs]


    def makedir_types(self, panel_id, subdirs=('pedestals', 'rms', 'status', 'plots')):
        """create structure of subdirectories in calibration repository under <dirrepo>/<panel_id>/...
        """
        dp = self.makedir_panel(panel_id)
        dirs = self.dir_types(panel_id, subdirs=subdirs)
        for d in dirs: self.makedir(d)
        return dirs


    def logname_on_start(self, scrname, year=None):
        _year = str_tstamp(fmt='%Y') if year is None else str(year)
        return '%s/%s_log_%s.txt' % (self.makedir_logs(), _year, scrname)


    def logname(self, scrname):
        tstamp = str_tstamp(fmt='%Y-%m-%dT%H%M%S')
        return '%s/%s_log_%s.txt' % (self.makedir_logs_year(), tstamp, scrname)


    def dir_constants(self, dname='constants'):
        """returns path to the directory like <dirrepo>/<logs>/<year>/<constants>
        """
        return os.path.join(self.dirrepo, dname)


    def makedir_constants(self, dname='constants'):
        d = self.makedir(self.dirrepo)
        return self.makedir(self.dir_constants(dname))


if __name__ == "__main__":

    dirrepo = './work'
    fname = 'testfname'
    scrname = 'testscrname-%s' % get_login()
    repoman = RepoManager(dirrepo, dirmode=0o775, filemode=0o664)
    print('makedir_logs %s' % repoman.makedir_logs())
    print('logname_on_start %s' % repoman.logname_on_start(fname))
    print('logname %s' % repoman.logname(scrname))
    print('makedir_constants %s' % repoman.makedir_constants())

# EOF
