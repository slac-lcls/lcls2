#!/usr/bin/env python

from time import time
t0_sec = time()
from psana.detector.NDArrUtils import *
print('import psana.detector.NDArrUtils time = %.6f sec' % (time()-t0_sec))
print('available methods:\n', dir())
import sys
sys.exit('END OF %s' % sys.argv[0].rsplit('/')[-1])

# EOF
