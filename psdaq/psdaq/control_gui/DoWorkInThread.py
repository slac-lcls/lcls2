
"""
Class :py:class:`DoWorkInThread` runs worker in the thread, access status and results 
========================================================================================

Usage ::
    from psdaq.control_gui.DoWorkInThread import DoWorkInThread

See:
    - test example below
    - :class:`CGDaqControl`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2019-02-04 by Mikhail Dubrovin
"""

#import logging
#logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s (%(threadName)-2s) %(message)s',

import threading

#----------

class DoWorkInThread :
    """Live long and prosper - Peace and long life"""
    def __init__(self, worker, **kwargs) :
        """Run worker in the thread. Input and output parameters are passed using dicio"""
        self.dicio = kwargs.get('dicio', {})
        self.t = threading.Thread(target=worker, args=(self.dicio,))
        self.t.start()

    def is_running(self) :
        return self.t.isAlive()

    def is_compleated(self) :
        return not self.is_running(self)

    def dict_io(self) :
        return self.dicio

#----------
#----------
#----------

if __name__ == "__main__" :

  from random import randint
  from time import time, sleep

  def worker_example(dicio) :
      print('input:', dicio['input'])
      pause = randint(1,5)
      print('sleep in worker_example for random %d sec'%pause)
      dicio['output'] = pause
      sleep(pause)

  def test_DoWorkInThread() :
      t0_sec = time()
      #kwargs = {'dicio': {'input':123, 'output':None}}
      #o = DoWorkInThread(worker_example, **kwargs)
      o = DoWorkInThread(worker_example, dicio={'input':123, 'output':None})
      print('worker_example is execution in thread. Submission time = %.6f sec' %(time()-t0_sec))

      for i in range(10) :
          if o.is_running() :
             print('%2d process is still running...' % i)
             sleep(1)
          else :
             print('results:', o.dict_io()['output'])
             break

#----------

  test_DoWorkInThread()

#----------
