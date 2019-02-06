
"""
Class :py:class:`CGParameters` store for common parameters
==========================================================

Usage ::
    from psdaq.control_gui.CGParameters import cp
    p = cp.thread_set_state

See:
    - :class:`CGDaqControl`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2019-02-04 by Mikhail Dubrovin
"""

#import logging
#logger = logging.getLogger(__name__)

#----------

class CGParameters :
    def __init__(self) :
        self.thread_set_state = None
        self.thread_get_state = None

#---------- SINGLETON

cp = CGParameters()

#----------
