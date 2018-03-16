#------------------------------
# nosetests -sv psana/psana/tests
# 
#------------------------------

import subprocess
doPlot = 0

#------------------------------

def test_peakFinder():
    subprocess.call(['python','psalgPeakFinder.py', str(doPlot)])

#------------------------------

def psalg() :
    test_peakFinder()

#------------------------------

if __name__ == '__main__':
    psalg()

#------------------------------
