"""
ControlTransition - Control Transition Class

Author: Chris Ford <caf@slac.stanford.edu>
"""

class ControlTransition(object):

    configure = b'CONFIGURE'
    unconfigure = b'UNCONFIGURE'
    beginrun = b'BEGINRUN'
    endrun = b'ENDRUN'
    enable = b'ENABLE'
    disable = b'DISABLE'
