
"""Class :py:class:`UtilsSubproc` - utilities for subprocess operations
=======================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/UtilsSubproc

    import psana.graphqt.UtilsSubproc as usp

Created on 2021-08-18 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

#import os
import select
import subprocess # for subprocess.Popen
from time import time, sleep

def subproc_open(command_seq, logname=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=None, shell=False, executable='/bin/bash'): 
    """ if shell=False command is a str: '/bin/bash -l -c "source /reg/g/psdm/etc/psconda.sh; echo $PYTHONPATH"'
        else command is a list: ['/bin/bash', '-l', '-c', '. /reg/g/psdm/etc/psconda.sh; echo "PATH: $PATH";\
                                 echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"; detnames exp=xppx44719:run=11']
    """    
    logger.debug('executable: %s' % executable)
    logger.debug('shell: %s' % str(shell))
    logger.debug('env: %s' % str(env))
    logger.debug('command:%s' % str(command_seq))

    if logname:
        log = open(logname, 'w')
        stderr = stdout = log
    return subprocess.Popen(command_seq, stdout=stdout, stderr=stderr, env=env, shell=shell, executable=executable)


class SubProcess:
    """SubProcess - access to one sub-process"""

    def __init__(self, **kwa):
        self.subproc = None
        self.selpoll = None

    def subprocs_open(self, command, logname=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=None, shell=False, executable='/bin/bash'):
        self.command = command
        logger.debug('command: %s' % str(command))
        logger.debug('shell: %s' % str(shell))
        assert isinstance(command, str if shell else list) #cmd = cmd if shell else command.split(' ')
        self.subproc = subproc_open(command, logname, stdout, stderr, env, shell, executable)
        self.selpoll = select.poll()
        self.selpoll.register(self.subproc.stdout, select.POLLIN)

    def __call__(self, *args, **kwargs): return self.subprocs_open(*args, **kwargs)

    def is_compleated(self): return self.subproc.poll()==0

    def kill(self): self.subproc.kill()

    def stdout_incriment(self):
        sp = self.subproc
        sp.stdout.flush()
        buf = ''
        while self.selpoll.poll(1):
            s = sp.stdout.readline()
            if not s: break
            buf += s.decode('ascii')
        return buf


class SubProcManager:
    """SubProcManager - multi sub-processes manager"""

    def __init__(self, **kwa):
        parent=kwa.get('parent', None)
        subprocs = []

    def subprocs_open(self, command, logname=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=None, shell=False, executable='/bin/bash'):
        osp = SubProcess()
        osp(command, logname, stdout, stderr, env, shell, executable)
        subprocs.append(osp)


spm = SubProcManager() # singleton


if __name__ == "__main__":

  def dump_line(s): print(s)

  def test_SubProcess(time_proc_sec=10):
      osp = SubProcess()
      dt_sec = 2
      t0_sec = time()
      cmd = 'python test-long-job.py %d' % time_proc_sec

      osp(cmd, stdout=subprocess.PIPE, env=None, shell=False)
      print('\n== creates subprocess for command: %s' % cmd)

      print('\n')
      counter = 0
      while True:
          counter +=1
          print('==== check %02d after %d sec - stdout_incriment:' % (counter, time()-t0_sec))
          print(osp.stdout_incriment())
          if osp.is_compleated():
              break
          sleep(dt_sec)

      print('\nSUBPROCESS IS COMPLETED')


  def test_subproc(time_proc_sec=10):
      """direct test of all methods
      """
      dt_sec = 2
      t0_sec = time()
      cmd = 'python test-long-job.py %d' % time_proc_sec
      #cmd = 'ls -ltra'
      #sp = subproc_open(cmd.split(' '), logname='test-log.txt', env=None, shell=False)
      sp = subproc_open(cmd.split(' '), stdout=subprocess.PIPE, env=None, shell=False)
      print('\n== creates subprocess for command: %s' % cmd)

      selpoll = select.poll()
      selpoll.register(sp.stdout, select.POLLIN)

      print('\n')
      counter = 0
      while True:
          counter +=1
          pstatus = sp.poll()
          print('==== check %02d after %d sec - poll status: %s' % (counter, time()-t0_sec, str(pstatus)))#, end='\r')

          sp.stdout.flush()

          while True:
            if selpoll.poll(1):
              s = sp.stdout.readline()
              if not s: break
              dump_line(s.decode('ascii').rstrip('\n'))
            else:
              print("---- stdout buffer is empty")
              break
 
          if pstatus==0:
              print('\nALL SUBPROCESSES COMPLETED')
              break
          sleep(dt_sec)


if __name__ == "__main__":
    import sys
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '2'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_subproc(time_proc_sec=10)
    elif tname == '1': test_subproc(time_proc_sec=20)
    elif tname == '2': test_SubProcess(time_proc_sec=10)
    else: print('test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

# EOF
