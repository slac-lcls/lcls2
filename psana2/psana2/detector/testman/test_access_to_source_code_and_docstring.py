#!/usr/bin/env python
"""test access to source code and docstring"""
import sys
import inspect
SCRNAME = sys.argv[0].rsplit('/')[-1]
print('\nSTART OF %s' % SCRNAME)
self = sys.modules[__name__] # __name__ = '__main__'
print('\nself.__doc__:', self.__doc__)
print('\n__name__:', __name__)
s = sys._getframe().f_code.co_name
print('\nsys._getframe().f_code.co_name:', s, type(s))
print('\ninspect.getsource(self):\n====\n%s====\n' % inspect.getsource(self))
sys.exit('END OF %s' % SCRNAME)
# EOF
