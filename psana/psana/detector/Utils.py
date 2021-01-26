"""
Utilities of common use for detector project
============================================

Usage::

  from psana.detector.Utils *

  is_selected = selected_record(nrec)
  s = info_command_line_parameters(parser) # for OptionParser
  s = info_command_line_arguments(parser) # for ArgumentParser

2020-11-06 created by Mikhail Dubrovin
"""

import sys
import numpy as np

from psana.pyalgos.generic.NDArrUtils import shape_nda_as_3d # shape_as_3d# info_ndarr, shape_as_3d

def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (nrec<500 and not nrec%100)\
       or (not nrec%1000)


def info_command_line_parameters(parser):
    """Prints input arguments and optional parameters
       from optparse import OptionParser
       parser = OptionParser(description=usage(1), usage = usage())
    """
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    s = 'Command: ' + ' '.join(sys.argv)+\
        '\n  Argument list: %s\n  Optional parameters:\n' % str(args)+\
        '    <key>      <value>              <default>\n'
    for k,v in opts.items():
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s


def info_command_line_arguments(parser):
    """Prints input arguments and optional parameters
       from argparse import ArgumentParser
       parser = ArgumentParser(description=usage(1))
    """
    args = parser.parse_args()
    opts = vars(args)
    defs = vars(parser.parse_args([])) # defaults only

    s = 'Command: ' + ' '.join(sys.argv)+\
        '\n  Argument list: %s\n  Optional parameters:\n' % str(args)+\
        '    <key>      <value>              <default>\n'
    for k,v in opts.items():
        s += '    %s %s %s\n' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))
    return s


# EOF
