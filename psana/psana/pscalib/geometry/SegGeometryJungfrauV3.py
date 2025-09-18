#!/usr/bin/env python
"""
Class :py:class:`SegGeometryJungfrauV3` describes the Jungfrau V3 sensor geometry
=================================================================================

V1 - regular matrix (512x1024) of 2x4 (256x256)-matrix asics
V2 - Y axis is flipped comparing to V1
V3 - accounts for wide pixels - all boarder pixels are double-size wide, 150um

\n ::

   (Xmin,Ymax)                          ^ Y                          (Xmax,Ymax)
   (511,0)                              |                             (1023,1023)
     ----------------- -----------------|----------------- -----------------
     |               | |               |||               | |               |
     |     ASIC      | |               |||               | |               |
     |    256x256    | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     ----------------- -----------------|----------------- -----------------
   -------------------------------------+-------------------------------------> X
     ----------------- -----------------|----------------- -----------------
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     |               | |               |||               | |               |
     ----------------- -----------------|----------------- -----------------
   (0,0)                                |                               (0,1023)
   (Xmin,Ymin)                                                        (Xmax,Ymin)


Usage::
    from psana.pscalib.geometry.SegGeometryJungfrauV3 import jungfrau_with_wide

See:
 * :py:class:`GeometryObject`
 * :py:class:`SegGeometry`
 * :py:class:`SegGeometryJungfrauV1`
 * :py:class:`SegGeometryJungfrauV2`
 * :py:class:`SegGeometryJungfrauV3`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2025-09-18 by Mikhail Dubrovin
"""

from psana.pscalib.geometry.SegGeometryJungfrauV2 import * # SegGeometryJungfrauV1, logging, np
logger = logging.getLogger(__name__)


class SegGeometryJungfrauV3(SegGeometryJungfrauV2):
    """Self-sufficient class for generation of Jungfrau 2x4 ASICs pixel coordinate array"""

    _name = 'SegGeometryJungfrauV3'
    _pixs  =   75    # regular pixel size in um (micrometer)
    _pixw  =  150    # wide pixel size in um (micrometer)
    _pixh  =  37.5   # half pixel size in um (micrometer)

    def __init__(sp, **kwa):
        logger.debug('SegGeometryJungfrauV2.__init__()')
        sp.use_wide_pix_center = kwa.get('use_wide_pix_center', True)
        SegGeometryJungfrauV2.__init__(sp)

    def correct_wide_pixel_center(sp):
        """see: https://confluence.slac.stanford.edu/spaces/PSDM/pages/223219435/Jungfrau#Jungfrau-Geometry"""
        for i in (255, 511, 767):
            sp.x_arr_um[i]   += sp._pixh
            sp.x_arr_um[i+1] -= sp._pixh
        sp.y_arr_um[255] += sp._pixh
        sp.y_arr_um[256] -= sp._pixh

    def correct_wide_pixel_size(sp):
        """see: https://confluence.slac.stanford.edu/spaces/PSDM/pages/223219435/Jungfrau#Jungfrau-Geometry"""
        for i in (255, 256, 511, 512, 767, 768): sp.x_psize_um[i] = sp._pixw
        for i in (255, 256):                     sp.y_psize_um[i] = sp._pixw

    def make_pixel_coord_arrs(sp):
        """Makes [512,1024] maps of x, y, and z pixel coordinates with origin in the center of 2x4"""
        x_asic = np.arange(sp._casic)*sp._pixs
        x0 = np.array((-512-2.5, -256.5, 1.5, 256+3.5))*sp._pixs
        sp.x_arr_um = np.hstack([x_asic+x0[0], x_asic+x0[1], x_asic+x0[2], x_asic+x0[3]])

        y_asic = np.arange(sp._rasic)*sp._pixs
        y0 = np.array((-256.5, 1.5))*sp._pixs
        sp.y_arr_um = np.hstack([y0[0]+y_asic, y0[1]+y_asic])

        if sp.use_wide_pix_center:
            sp.correct_wide_pixel_center()

        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows, sp._cols))

    def make_pixel_size_arrs(sp):
        """Makes [512,1024] maps of x, y, and z pixel size and pix_area_arr"""
        if sp.pix_area_arr is not None: return

        sp.x_psize_um = np.arange(sp._cols)*sp._pixs
        sp.y_psize_um = np.arange(sp._rows)*sp._pixs

        if sp.use_wide_pix_center:
            sp.correct_wide_pixel_size()

        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(sp.x_psize_um, sp.y_psize_um)
        sp.z_pix_size_um = np.ones((sp._rows, sp._cols))*sp._pixd

        area_normf = 1.0/(sp._pixs*sp._pixs)
        sp.pix_area_arr = np.multiply(sp.x_pix_size_um, sp.y_pix_size_um) * area_normf

jungfrau_with_wide = SegGeometryJungfrauV3()

# EOF

