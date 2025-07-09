
""" for test purpose - emmitation of the very long process"""

import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(message)s', level=logging.DEBUG)
#logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

from time import sleep

# input pareameter - time of the process in sec
time_proc_sec = int(sys.argv[1]) if len(sys.argv) > 1 else 10
dt_sec = 0.5
nloops = int(float(time_proc_sec)/dt_sec)

for i in range(nloops):
    s = 'loop %04d' % i
    print(s)
    logger.info(s)
    sys.stdout.flush()
    sleep(dt_sec)

print('End of %s' % sys.argv[0].split('/')[-1])
