#!/usr/bin/env python

"""
Class :py:class:`SegGeometryMatrixAnyV1` extends SegGeometryMatrixV2 object for postponed initialization of parameters
=======================================================================================================================

  MatrixV2 sensor coordinate are defined in Cartesian frame:

  @code
    (Xmin,Ymax)  ^ Y   (Xmax,Ymax)
    (0,0)        |     (0,ncols-1)
       +---------+---------+
       |         |         |
       |         |         |
       |         |         |
       |         |         |
     --+---------+---------+--> X
       |         |         |
       |         |         |
       |         |         |
       |         |         |
       +---------+---------+
    (nrows-1,0)  |    (nrows-1,ncols-1)
    (Xmin,Ymin)       (Xmax,Ymin)
  @endcode


Usage of interface methods::

import psana2.pscalib.geometry.SegGeometryMatrixAnyV1

    sg = SegGeometryMatrixAnyV1()
    sg.init_matrix_parameters(shape=(512,512), pix_size_rcsd_um=(75,75,75,400))

    # further, use sg.* with any of SegGeometryMatrixV2 methods.

See:
 * :py:class:`GeometryObject`,
 * :py:class:`SegGeometry`,
 * :py:class:`SegGeometryEpix100V1`,
 * :py:class:`SegGeometryEpix10kaV1`,
 * :py:class:`SegGeometryJungfrauV2`,
 * :py:class:`SegGeometryMatrixV2`,
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@date 2025-05-08
@author Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import psana2.pscalib.geometry.SegGeometryMatrixV2 as mv2

class SegGeometryMatrixAnyV1(mv2.SegGeometryMatrixV2):
    """Defines SegGeometryMatrixV2 with postponed initialization of parameters"""

    _name = 'SegGeometryMatrixAnyV1'

    def __init__(self):
        logger.info('in %s.__init__() WITH POSTPONED INITIALIZATION OF PARAMETERS' % self._name)

    def init_matrix_parameters(self, **kwa):
        """kwa keywords: shape, pix_size_rcsd_um, asic0indices, nasics_in_rows, nasics_in_cols"""
        sh = kwa.get('shape', (512,512))
        ps = kwa.get('pix_size_rcsd_um', (75,75,75,400))
        logger.info('in %s.init_matrix_parameters(**kwa) INITIALIZATION OF PARAMETERS with **kwa:\n  %s' % (self._name, str(kwa)))
        mv2.SegGeometryMatrixV2.__init__(self, rows=sh[0], cols=sh[1], pix_size_rows=ps[0], pix_size_cols=ps[1], pix_scale_size=ps[2], pix_size_depth=ps[3])

        # for converter
        self._asic0indices   = kwa.get('asic0indices', ((0, 0),))
        self._nasics_in_rows = kwa.get('nasics_in_rows', 1)
        self._nasics_in_cols = kwa.get('nasics_in_cols', 1)

# EOF
