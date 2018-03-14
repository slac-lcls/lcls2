#------------------------------
"""

from psana.pyalgos.generic.PSNameManager import nm

"""
#------------------------------
#import sys
#import os
#------------------------------

from psana.pyalgos.generic.PSConstants import INSTRUMENTS, DIR_INS, DIR_FFB # , DIR_LOG

#------------------------------

class PSNameManager :
    """Contains a set of methods returning LCLS/psana unique names for directories, dataset, etc.
    """
    _name = 'PSNameManager'

#------------------------------

    def __init__(self, instr_dir=DIR_INS) :
        self.set_instrument_directory(instr_dir)

#------------------------------

    def set_instrument_directory(self, instr_dir=DIR_INS) :
        self.instr_dir = instr_dir

#------------------------------

    def instrument(self, exp_name) :
        """Returns 3-letter uppercase instrument for experiment name, e.g.: CXI for exp_name=cxi12345"""
        for ins in INSTRUMENTS :
            if ins.lower() in exp_name : return ins
        raise IOError('Specified experiment %s does not belong to any instrument: %s' % exp_name, str(INSTRUMENTS))

#------------------------------

    def dir_exp(self, exp_name) :
        """Returns directory of experiments, e.g.: /reg/d/psdm/CXI"""
        return '%s/%s' % (self.instr_dir, self.instrument(exp_name))

#------------------------------

    def dir_xtc(self, exp_name) :
        """Returns xtc directory, e.g.: /reg/d/psdm/CXI/cxi02117/xtc"""
        return '%s/%s/xtc' % (self.dir_exp(exp_name), exp_name)

#------------------------------

    def dir_ffb(self, exp_name) :
        """Returns ffb xtc directory, e.g.: /reg/d/ffb/cxi/cxi02117/xtc"""
        return '%s/%s/%s/xtc' % (DIR_FFB, self.instrument(exp_name).lower(), exp_name)

#------------------------------

    def dir_calib(self, exp_name) :
        """Returns calib directory, e.g.: /reg/d/psdm/CXI/cxi02117/calib"""
        return '%s/%s/calib' % (self.dir_exp(exp_name), exp_name)

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
'''
    def dsname(self, exp='cxi12316', run=0, ext='None') :
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
'''

#------------------------------

nm = PSNameManager()

#------------------------------

if __name__ == "__main__" :
  def test_all() :

    #from psana.pyalgos.generic.PSNameManager import nm

    exp = 'cxix25615'
    print('instrument:', nm.instrument(exp))
    print('dir_exp   :', nm.dir_exp(exp))
    print('dir_xtc   :', nm.dir_xtc(exp))
    print('dir_ffb   :', nm.dir_ffb(exp))
    print('dir_calib :', nm.dir_calib(exp))
    #print('dsname    :', nm.dsname(exp))

#------------------------------

if __name__ == "__main__" :
    test_all()

#------------------------------
