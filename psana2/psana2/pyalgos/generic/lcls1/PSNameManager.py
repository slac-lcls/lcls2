#------------------------------
"""
@version $Id: PSNameManager.py 13157 2017-02-18 00:05:34Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin

Usage ::
    see method test_all()
"""
#------------------------------
#import sys
#import os
#------------------------------

class PSNameManager :
    """Contains a set of methods returning LCLS/psana unique names for directories, dataset, etc.
    """
    _name = 'PSNameManager'

    def __init__(self, cp=None, log=None) :
        #log.info('object is init-ed', self._name)
        self.set_cp_and_log(cp, log)
        
    def set_cp_and_log(self, cp, log) :
        self.cp  = cp
        self.log = log

    def set_config_pars(self, cp) :
        #log.info('config pars object is set from file: %s' % cp.fname_cp, self._name)
        self.cp  = cp

    def set_logger(self, log) :
        #log.info('config pars object is set from file: %s' % cp.fname_cp, self._name)
        self.log = log 
    
    def cpars(self) :
        if self.cp is None :
            from expmon.PSConfigParameters import PSConfigParameters
            self.cp = PSConfigParameters()
        return self.cp

#    def logger(self) :
#        if self.log is None :
#            from expmon.Logger import log
#            self.log = log
#        return self.log

#------------------------------

    def dir_exp(self) :
        """Returns directory of experiments, e.g.: /reg/d/psdm/CXI"""
        return '%s/%s' % (self.cpars().instr_dir.value(), self.cpars().instr_name.value())

#------------------------------

    def dir_xtc(self) :
        """Returns xtc directory, e.g.: /reg/d/psdm/CXI/cxi02117/xtc"""
        return '%s/%s/%s/xtc' % (self.cpars().instr_dir.value(), self.cpars().instr_name.value(), self.cpars().exp_name.value())
    
#------------------------------

    def dir_ffb(self) :
        """Returns ffb xtc directory, e.g.: /reg/d/ffb/cxi/cxi02117/xtc"""
        return '%s/%s/%s/xtc' % (self.cpars().instr_dir.value(), self.cpars().instr_name.value().lower(), self.cpars().exp_name.value())
    
#------------------------------

    def dir_calib(self) :
        """Returns calib directory, e.g.: /reg/d/psdm/CXI/cxi02117/calib"""
        return '%s/%s/%s/calib' % (self.cpars().instr_dir.value(), self.cpars().instr_name.value(), self.cpars().exp_name.value())

#------------------------------

    def dsname(self) :
        """Returns string like exp=cxi12316:run=1234:..."""
        cp = self.cpars()
        ext = cp.dsextension.value()

        if ext == 'shmem' :
            return 'shmem=psana.0:stop=no'

        if cp.exp_name.is_default() or cp.str_runnum.is_default() : return None
        base = 'exp=%s:run=%s' % (cp.exp_name.value(), cp.str_runnum.value().lstrip('0'))

        if ext == 'None' :
            return base

        if ext == 'idx' or ext == 'smd' :
            return '%s:%s' % (base, ext)

        if ext == 'smd:live' :
            return '%s:%s:dir=%s' % (base, ext, self.dir_ffb())

        return base

#------------------------------

    def log_file_repo(self) :
        import os
        import expmon.PSUtils as psu

        if None in (self.cp, self.log) : return None
        # Returns name like /reg/g/psdm/logs/emon/2017/07/2016-05-17-10:16:00-log-dubrovin-562.txt
        fname = self.log.getLogFileName()     # 2016-07-19-11:53:02-log.txt
        year, month = fname.split('-')[:2]  # 2016, 07
        name, ext = os.path.splitext(fname) # 2016-07-19-11:53:02-log, .txt   
        return '%s/%s/%s/%s-%s-%s%s' % (self.cp.dir_log_repo.value(), year, month, name, psu.get_login(), psu.get_pid(), ext)

#------------------------------

nm = PSNameManager()

#------------------------------

def test_all() :

    #from expmon.PSNameManager import nm
    from expmon.EMConfigParameters import cp
    nm.set_config_pars(cp)

    print 'dir_exp   :', nm.dir_exp()
    print 'dir_xtc   :', nm.dir_xtc()
    print 'dir_ffb   :', nm.dir_ffb()
    print 'dir_calib :', nm.dir_calib()
    print 'dsname    :', nm.dsname()

#------------------------------

if __name__ == "__main__" :
    test_all()

#------------------------------
