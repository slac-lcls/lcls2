
import sys
from psana.graphqt.FWViewHist import *

def test_guiview(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None

    rs = QRectF(0, 0, 100, 1000)

    if   tname ==  '0': w=FWViewHist(None, rs, origin='UL', scale_ctl='V', fgcolor='white', bgcolor='gray')
    elif tname ==  '1': w=FWViewHist(None, rs, origin='DL', scale_ctl='H', fgcolor='black', bgcolor='yellow')
    elif tname ==  '2': w=FWViewHist(None, rs, origin='DR')
    elif tname ==  '3': w=FWViewHist(None, rs, origin='UR')
    elif tname ==  '4': w=FWViewHist(None, rs, origin='DR', scale_ctl='V', fgcolor='yellow', bgcolor='gray', orient='V')
    elif tname ==  '5': w=FWViewHist(None, rs, origin='DR', scale_ctl='V', fgcolor='white', orient='V')
    else:
        print('test %s is not implemented' % tname)
        return

    w.print_attributes()

    #w.connect_axes_limits_changed(w.test_axes_limits_changed_reception)
    #w.disconnect_axes_limits_changed(w.test_axes_limits_changed_reception)
    w.show()
    app.exec_()

    del w
    del app


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_guiview(tname)
    sys.exit('End of Test %s' % tname)

# EOF
