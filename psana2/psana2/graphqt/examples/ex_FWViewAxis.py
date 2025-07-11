
from psana2.graphqt.FWViewAxis import *

import sys

def test_guiview(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None
    rs=QRectF(0, 0, 1000, 10)
    if   tname ==  '0': w=FWViewAxis(None, rs, side='D', origin='UL', fgcolor='red', bgcolor='yellow')
    elif tname ==  '1': w=FWViewAxis(None, rs, side='U', origin='UL')
    elif tname ==  '2': w=FWViewAxis(None, rs, side='L', origin='UL')
    elif tname ==  '3': w=FWViewAxis(None, rs, side='R', origin='UL')

    elif tname == '10': w=FWViewAxis(None, rs, side='D', origin='DL')
    elif tname == '11': w=FWViewAxis(None, rs, side='U', origin='DL')
    elif tname == '12': w=FWViewAxis(None, rs, side='L', origin='DL')
    elif tname == '13': w=FWViewAxis(None, rs, side='R', origin='DL')

    elif tname == '20': w=FWViewAxis(None, rs, side='D', origin='DR')
    elif tname == '21': w=FWViewAxis(None, rs, side='U', origin='DR')
    elif tname == '22': w=FWViewAxis(None, rs, side='L', origin='DR')
    elif tname == '23': w=FWViewAxis(None, rs, side='R', origin='DR')

    elif tname == '30': w=FWViewAxis(None, rs, side='D', origin='UR')
    elif tname == '31': w=FWViewAxis(None, rs, side='U', origin='UR')
    elif tname == '32': w=FWViewAxis(None, rs, side='L', origin='UR')
    elif tname == '33': w=FWViewAxis(None, rs, side='R', origin='UR')
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
