#------------------------------
# run it as:
# python lcls2/psana/psana/detector/TestBase.py
#------------------------------

class TestBase :
    def __init__(self, p1=1, p2=2) :
        print('In TestBase p1, p2:', p1, p2)

#------------------------------

if __name__ == "__main__" :
    print('%s in TestBase.py' % __name__)
    TestBase()
    TestBase(3,4)

#------------------------------
