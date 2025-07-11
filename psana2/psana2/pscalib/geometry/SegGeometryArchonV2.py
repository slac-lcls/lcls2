#!/usr/bin/env python
"""
Class :py:class:`SegGeometryArchonV2` describes ArchonV2 sensor geometry
===================================================================================

The same as SegGeometryArchonV2, but fake pixels of each bank preceed real

See:
 * :py:class:`SegGeometryArchonV1`
 * :py:class:`SegGeometryStore`

This software was developed for the LCLS-II project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2023-12-03 by Mikhail Dubrovin
"""

from psana2.pscalib.geometry.SegGeometryArchonV1 import *
logger = logging.getLogger(__name__)

class SegGeometryArchonV2(SegGeometryArchonV1):
    """Generates ArchonV2 sensor pixel coordinate array"""

    def __init__(sp, **kwa):
        sp._name = 'SegGeometryArchonV2'
        #logger.debug('%s.__init__()'%sp._name)
        SegGeometryArchonV1.__init__(sp, **kwa)

    def make_pixel_coord_arrs(sp, dtype=np.float64):
        """Makes [<nrows>,4800] maps of x, y, and z pixel coordinates with origin in the center"""
        #print('in SegGeometryArchonV2.make_pixel_coord_arrs')
        x_bank = np.array(sp._colsf*(np.nan,) + tuple(np.arange(sp._colsr)*sp._pixsc), dtype=dtype)
        w = sp._colsr * sp._pixsc # =2640
        y_bank = -np.arange(sp._rows, dtype=dtype)*sp._pixsr
        sp.x_arr_um = np.hstack([x_bank+i*w for i in range(sp._nbanks)])
        dx = (sp.x_arr_um[sp._colsf]+sp.x_arr_um[-1])/2
        sp.x_arr_um -= dx # offset to origin in center
        dy = (y_bank[0]+y_bank[-1])/2
        sp.y_arr_um = y_bank - dy # offset to origin in center
        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros(sp.x_pix_arr_um.shape)
        #logger.debug('x_arr_um:\n%s...\ny_arr_um:\n%s' % (str(sp.x_arr_um[0:-sp._colsf+1]), str(sp.y_arr_um)))

    def make_pixel_size_arrs(sp, dtype=np.float64):
        """Makes maps of x, y, and z pixel size and normalized (all 1) pixel area"""
        if sp.pix_area_arr is not None: return # member data are already evaluated
        x_arr_size_um = np.array(sp._colsf*(np.nan,) + sp._nbanks*(sp._colsr*(sp._pixsc,)), dtype=dtype)
        y_arr_size_um = np.ones(sp._rows, dtype=dtype)*sp._pixsr
        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones(sp.x_pix_size_um.shape, dtype=dtype) * sp._pixd
        factor = 1./(sp._pixsr*sp._pixsc)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor

    def mask_fake(sp, dtype=np.uint8, **kwa):
        """returns mask of shape=(<nrows>,4800), with fake pixels of all banks set to 0"""
        fake1bank = np.zeros((sp._rows, sp._colsf), dtype=dtype)
        mask = np.ones(sp.x_pix_arr_um.shape, dtype=dtype)
        sf, st = sp._colsf, sp._colst # = 36, 300
        for i in range(sp._nbanks):
             mask[:,st*i:st*i+sf] = fake1bank
        return mask

    def get_xyz_min_um(sp):
        return sp.x_arr_um[sp._colsf],\
               sp.y_arr_um[-1],\
               sp.z_pix_arr_um[0,0]

    def get_xyz_max_um(sp, axis=None):
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS"""
        return sp.x_arr_um[-1],\
               sp.y_arr_um[0],\
               sp.z_pix_arr_um[0,0]

# for converter

    def asic0indices(self): self.print_warning('asic0indices')
    def asic_rows_cols(self): self.print_warning('asic_rows_cols')
    def number_of_asics_in_rows_cols(self): self.print_warning('number_of_asics_in_rows_cols')
    def name(self): self.print_warning('name')

#archon_one = SegGeometryArchonV1(use_wide_pix_center=False)

# EOF
