

from psana.graphqt.GWView import *

logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)#DEBUG)

def test_fwview(tname):
    #print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w=GWView(rscene=QRectF(-10, -10, 30, 30),\
             scale_ctl=('HV', 'H', 'V', '')[int(tname)],\
             show_mode=0o377)
    w.setWindowTitle("TestGWView")
    w.setGeometry(20, 20, 600, 600)
    w.show()
    app.exec_()
    app.quit()
    del w
    del app

def usage(tname):
    scrname = sys.argv[0].split('/')[-1]
    s = '\nUsage: python %s <tname [0-4]>' %scrname\
      + '\n   where tname stands for scale_ctl=("HV", "H", "V", "")[int(tname)]'
    return s

if __name__ == "__main__":
    import sys
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_fwview(tname)
    print(usage(tname))
    sys.exit('End of Test %s' % tname)

# EOF
