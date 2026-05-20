#!/usr/bin/env python
"""
Class :py:class:`SegGeometryEpixUHR3x2V1` describes the EpixUHR3x2V1 sensor geometry
===================================================================================

In this class we use natural matrix notations like in raw data array
\n We assume that
\n * sensor consists of 2x3 (168, 192) ASICs has (2*168, 3*192) = (336, 576) rows, columns,
\n * upixuhr3x2 has a pixel size 100x100um, wide pixel size 100x262.5um
\n * X-Y coordinate system origin is in the sensor center,
\n * pixel (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax), as shown below
\n ::
\n FRONT VIEW FROM IP

      (Xmin,Ymax)                               ^ Y                                (Xmax,Ymax)
      (0,0)                                     |                                      (0,575)
      ----------------------------------------------------------------------------------------
      |                  A0(0,0) ||             |           A1 ||                         A2 |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
    --|--------------------------++-------------+--------------++----------------------------+--> X
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      |                          ||             |              ||                            |
      | A3(0,0)                  || A4          |              || A5                         |
      ----------------------------------------------------------------------------------------
      (335,0)                                   |                                    (335,575)
      (Xmin,Ymin)                                                                  (Xmax,Ymin)

From Gabriel: Asic data is physically arranged as:
  |   A1   |   A3   |   A5   |
  +--------+--------+--------+
  |   A0   |   A2   |   A4   |

Implemented Dawood's schema, "*" - shows location of ASIC(0,0),
det.raw._segments(evt)[<iseg>].raw has shape:(6, 32256)
assuming shaped as (2*3, 168, 192)

  |       *|       *|       *|
  |   A0   |   A1   |   A2   |
  +--------+--------+--------+
  |   A3   |   A4   |   A5   |
  |*       |*       |*       |

Usage::

    from SegGeometryEpixUHR3x2V1 import epixuhr3x2_one as sg

    sg.print_seg_info(0o377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    area     = sg.pixel_area_array()
    mask = sg.pixel_mask_array(width=5, wcenter=5)
    mask = sg.pixel_mask_array(width=0, wcenter=0, edge_rows=1, edge_cols=1, center_rows=1, center_cols=1)

    sizeX = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()

    X     = sg.pixel_coord_array('X')
    X,Y,Z = sg.pixel_coord_array()
    logger.info('X.shape =' + str(X.shape))

    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_max()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')

    # global method for rotation of numpy arrays:
    Xrot, Yrot = rotation(X, Y, C, S)
    ...

See:
 * :py:class:`GeometryObject`
 * :py:class:`SegGeometry`
 * :py:class:`SegGeometryCspad2x1V1`
 * :py:class:`SegGeometryEpixHR2x2V1`
 * :py:class:`SegGeometryEpix10kaV1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2026-05-12 by Mikhail Dubrovin
"""
import sys
import psana.detector.NDArrUtils as au

from psana.pscalib.geometry.SegGeometryEpix10kaV1 import *
logger = logging.getLogger(__name__)


class SegGeometryEpixUHR3x2V1(SegGeometryEpix10kaV1):
    """Self-sufficient class for generation of Epix10ka sensor (2x2 ASICs) pixel coordinate array"""

    def __init__(sp, **kwa):
        sp._name = 'SegGeometryEpixUHR3x2V1'
        logger.debug('%s.__init__()'%sp._name)

        sp._rows  = 336     # number of rows in 2x3
        sp._cols  = 576     # number of cols in 2x3
        sp._pixs  = 100     # pixel size in um (micrometer)
        sp._pixw  = 262.5   # wide pixel size in um (micrometer)
        sp._pixd  = 400.00  # pixel depth in um (micrometer)
        sp._accgap = 425    # gap betwen ASICs edge pixels, um

        sp._arows = sp._rows//2 # number of rows in ASIC = 168
        sp._acols = sp._cols//3 # number of cols in ASIC = 192 "//" - returns int part
        sp._pixsh = sp._pixs//2 # half pixel size = 50um
        sp._pixwh = sp._pixw/2  # half wide pixel size = 262.5/2
        sp._offsw = sp._pixwh - sp._pixsh  # (262.5/2-50=81.25) -> 81 wide pixel center offse relative to contact center

        sp._nasics_in_rows = 2 # Number of ASICs in row direction
        sp._nasics_in_cols = 3 # Number of ASICs in column direction

        sp._asic0indices = ((0, sp._acols-1),   (0, sp._acols*2-1),         (0, sp._acols*3-1),\
                            (2*sp._arows-1, 0), (2*sp._arows-1, sp._acols), (2*sp._arows-1, sp._acols*2))

        SegGeometryEpix10kaV1.__init__(sp, **kwa)

    def make_pixel_coord_arrs(sp):
        """Makes (336, 576) maps of x, y, and z 3x2 pixel coordinates
        with origin in the center of 3x2 panel
        """
        x_asic = np.arange(sp._acols)*sp._pixs
        size_asic_xcc = (sp._acols-1)*sp._pixs
        x_asic_1 = x_asic - size_asic_xcc//2
        x_asic_0 = x_asic_1 - size_asic_xcc - sp._accgap
        x_asic_2 = x_asic_1 + size_asic_xcc + sp._accgap
        if sp.use_wide_pix_center:
            logger.debug(f' use_wide_pix_center - apply offsets for x: {sp._offsw}')
            x_asic_0[-1] += sp._offsw
            x_asic_1[1]  -= sp._offsw
            x_asic_1[-1] += sp._offsw
            x_asic_2[1]  -= sp._offsw
        sp.x_arr_um = np.hstack((x_asic_0, x_asic_1, x_asic_2))

        logger.debug(f'  size_asic_xcc, um: {size_asic_xcc}'\
             +au.info_ndarr(x_asic_1,          '\n  x_asic_1      ', last=7)\
             +au.info_ndarr(x_asic_1[185:],    '\n  x_asic_1[185:]', last=7)\
             +au.info_ndarr(sp.x_arr_um,       '\n  x_arr_um      ', last=6)\
             +au.info_ndarr(sp.x_arr_um[570:], '\n  x_arr_um[570:]', last=6))

        y_asic_min = sp._pixw - sp._pixsh  # wide pixel size # + half pixel size 0.5 (to remove fractional um)
        y_asic_max = y_asic_min + (sp._arows-1)*sp._pixs
        y_asic_0 = np.arange(y_asic_max, y_asic_min-sp._pixs, -sp._pixs) # stop-=sp._pixs because the last value is not included

        if sp.use_wide_pix_center:
            logger.debug(f' use_wide_pix_center - apply offset for y: {sp._offsw}')
            y_asic_0[-1] -= sp._offsw # set y-coordinate of the wide pixel in its geometry center
        sp.y_arr_um = np.hstack((y_asic_0, -y_asic_0[::-1]))

        logger.debug(f'  y_asic_min / max, um: {y_asic_min} / {y_asic_max}'\
             +au.info_ndarr(y_asic_0,          '\n  y_asic_0      ', last=8)\
             +au.info_ndarr(y_asic_0[162:],    '\n  y_asic_0[162:]', last=8)\
             +au.info_ndarr(sp.y_arr_um,       '\n  y_arr_um      ', last=6)\
             +au.info_ndarr(sp.y_arr_um[330:], '\n  y_arr_um[330:]', last=6))

        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows, sp._cols))


    def make_pixel_size_arrs(sp):
        """Makes [352,384] maps of x, y, and z 2x2 pixel size
        """
        if sp.pix_area_arr is not None: return

        x_size_um_asic0 = np.ones(sp._acols)*sp._pixs
        x_size_um_asic1 = np.ones(sp._acols)*sp._pixs
        x_size_um_asic2 = np.ones(sp._acols)*sp._pixs
        x_size_um_asic0[-1] = sp._pixw
        x_size_um_asic1[-1] = sp._pixw
        x_size_um_asic1[0]  = sp._pixw
        x_size_um_asic2[0]  = sp._pixw
        x_arr_size_um = np.hstack((x_size_um_asic0, x_size_um_asic1, x_size_um_asic2))

        y_size_um_asic0 = np.ones(sp._arows)*sp._pixs
        y_size_um_asic3 = np.ones(sp._arows)*sp._pixs
        y_size_um_asic0[-1] = sp._pixw
        y_size_um_asic3[0]  = sp._pixw
        y_arr_size_um = np.hstack((y_size_um_asic0, y_size_um_asic3))

        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows, sp._cols)) * sp._pixd

        factor = 1./(sp._pixs*sp._pixs)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor
        logger.debug(au.info_ndarr(sp.pix_area_arr, '\n  sp.pix_area_arr', last=10))


    def print_member_data(sp):
        s = 'print_member_data()'\
          + '\n    _rows : %d' % sp._rows\
          + '\n    _cols : %d' % sp._cols\
          + '\n    _pixs : %d' % sp._pixs\
          + '\n    _pixw : %d' % sp._pixw\
          + '\n    _pixd : %d' % sp._pixd\
          + '\n    _accgap : %7.2f' % sp._pixd\
          + '\n    _arows: %d'    % sp._arows\
          + '\n    _acols: %d'    % sp._acols\
          + '\n    _pixsh: %7.2f' % sp._pixsh\
          + '\n    _pixwh: %7.2f' % sp._pixwh\
          + '\n    _offsw: %7.2f' % sp._offsw\
          + '\n    _asic0indices: %s' % str(sp._asic0indices)
        logger.info(s)


    def pixel_mask_array(sp, width=0, wcenter=0, edge_rows=1, edge_cols=1, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa):
        """ Returns numpy array of pixel mask: 1/0 = ok/masked.

        Parameters
        ----------

        - width (uint) - width in pixels of masked edge
        - wcenter (uint) - width in pixels of masked central rows and columns
        - edge_rows (uint) - width in pixels of masked edge rows
        - edge_cols (uint) - width in pixels of masked edge columns
        - center_rows (uint) - width in pixels of masked central rows
        - center_cols (uint) - width in pixels of masked central columns

        Return
        ------

        np.array (dtype=np.uint8) - mask array shaped as data
        """
        mask = sp.pixel_ones_array()

        if width>0: edge_rows = edge_cols = width
        if wcenter>0: center_rows = center_cols = wcenter

        if edge_rows>0: # mask edge rows
            w = edge_rows
            zero_row = np.zeros((w,sp._cols),dtype=dtype)
            mask[0:w,:] = zero_row # mask top    edge rows
            mask[-w:,:] = zero_row # mask bottom edge rows

        if edge_cols>0: # mask edge cols
            w = edge_cols
            zero_col = np.zeros((sp._rows,w),dtype=dtype)
            mask[:,0:w] = zero_col # mask left  edge columns
            mask[:,-w:] = zero_col # mask right edge columns

        if center_rows>0: # mask central rows
            w = center_rows
            g = sp._arows
            zero_row = np.zeros((w,sp._cols),dtype=dtype)
            mask[g-w:g,:] = zero_row # mask central-low  rows
            mask[g:g+w,:] = zero_row # mask central-high rows

        if center_cols>0: # mask central rows
            w = center_cols
            g = sp._acols
            g2 = g*2
            zero_col = np.zeros((sp._rows,w),dtype=dtype)
            mask[:,g-w:g]   = zero_col # mask central-left  columns between A0:A1
            mask[:,g:g+w]   = zero_col # mask central-right columns between A0:A1
            mask[:,g2-w:g2] = zero_col # mask central-left  columns between A1:A2
            mask[:,g2:g2+w] = zero_col # mask central-right columns between A1:A2

        return mask


epixuhr3x2_one = SegGeometryEpixUHR3x2V1(use_wide_pix_center=False)
epixuhr3x2_wpc = SegGeometryEpixUHR3x2V1(use_wide_pix_center=True)

if __name__ == "__main__":

    import logging
    print(80*'_', '\n')
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO)

    epixuhr3x2_one.make_pixel_size_arrs()
    epixuhr3x2_one.print_member_data()
    epixuhr3x2_one.print_pixel_size_arrs(rowslice=slice(165,171), colslice=slice(189,195))  #(2*3, 168, 192)
    epixuhr3x2_one.print_maps_seg_um()
    epixuhr3x2_one.print_xy_1darr_um()
    epixuhr3x2_one.print_xyz_min_max_um()

# EOF
