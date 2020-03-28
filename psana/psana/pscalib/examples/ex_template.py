#!/usr/bin/env python
"""
"""

#----------

def ex_01(tname) : 
    print('ex_01 %s' % tname)

#----------

if __name__ == "__main__" :
    import sys
    scrname = sys.argv[0]
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s:' % tname)
    if   tname == '0' : ex_01(tname)
    elif tname == '1' : ex_01(tname)
    else : print('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s %s' % (scrname, tname))
 
#----------
