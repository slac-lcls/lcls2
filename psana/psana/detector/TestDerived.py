#------------------------------
# run it as:
# python lcls2/psana/psana/detector/TestBase.py
#------------------------------

from psana.detector.TestBase import TestBase

#------------------------------

class TestDerived(TestBase) :
    def __init__(self) :
        TestBase.__init__(self,5,6)
        print('In Derived')

#------------------------------

if __name__ == "__main__" :
    print('%s in TestDerived.py' % __name__)
    a = TestDerived()

#------------------------------
