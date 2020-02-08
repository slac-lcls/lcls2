#!/usr/bin/env python
""" 
"""
import sys

#----------

def data_file(tname='0') :
    dname = '/reg/g/psdm/detector/data2_test/xtc/'
    fname = 'data-amox23616-r0104-e000010-xtcav.xtc2' if tname == '0' else\
            'data-amox23616-r0104-e000400-xtcav.xtc2' if tname == '1' else\
            'data-amox23616-r0131-e000200-xtcav.xtc2' if tname == '2' else\
            'data-amox23616-r0137-e000100-xtcav.xtc2'
    fname = dname+fname
    print('data file: %s' % fname)
    return fname

#----------

if __name__ == "__main__":
    data_file(sys.argv[1] if len(sys.argv) > 1 else '0')

#----------
