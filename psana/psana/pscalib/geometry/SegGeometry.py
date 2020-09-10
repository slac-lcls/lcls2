#!/usr/bin/env python
#------------------------------
"""
:py:class:`SegGeometry` - abstract class with interface description
===================================================================

Methods of this class should be re-implemented in derived classes with name pattern SegGeometry<SensorVers> 
for pixel geometry description of all sensors.
For example, CSPAD 2x1 sensor is implemented in class :py:class:`psana.pscalib.geometry.SegGeometryCspad2x1V1`.
Access to all implemented sensors is available through the factory method in class :py:class:`psana.pscalib.geometry.SegGeometryStore`.

Usage::

    from SegGeometryCspad2x1V1 import cspad2x1_one as sg

    sg.print_seg_info(pbits=0o377)
    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = sg.pixel_scale_size()
    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask(mbits=0o377)
    sizeX    = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()
    X        = sg.pixel_coord_array('X')
    X,Y,Z    = sg.pixel_coord_array()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')
    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_mas()
    lst_a0rc = sg.asic0indices()
    r,c = sg.asic_rows_cols()
    nasics_in_rows, nasics_in_cols = sg.number_of_asics_in_rows_cols()
    segment_name = sg.name()

    ...
    print 'X.shape =', X.shape

See:
 * :py:class:`GeometryObject`, 
 * :py:class:`SegGeometry`, 
 * :py:class:`SegGeometryCspad2x1V1`, 
 * :py:class:`SegGeometryEpix100V1`, 
 * :py:class:`SegGeometryMatrixV1`, 
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)
import numpy as np

#------------------------------

def rotation(X, Y, C, S):
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

class SegGeometry:
    AXIS = ['X', 'Y', 'Z']
    DIC_AXIS = {'X':0, 'Y':1, 'Z':2}
    wmsg = 'WARNING! %s - interface method from the base class \nneeds to be re-implemented in the derived class'

    def __init__(self):
        pass

#------------------------------

    def print_warning(self, s):
        logger.warning(self.wmsg % s)

    def print_seg_info(self, pbits=0):
        """ Prints segment info for selected bits
        """
        self.print_warning('print_seg_info(pbits=0)')

    def size(self):
        """ Returns segment size - total number of pixels in segment
        """
        self.print_warning('size()')

    def rows(self):
        """ Returns number of rows in segment
        """
        self.print_warning('rows()')

    def cols(self):
        """ Returns number of cols in segment
        """
        self.print_warning('cols()')

    def shape(self):
        """ Returns shape of the segment [rows, cols]
        """
        self.print_warning('shape()')

    def pixel_scale_size(self):
        """ Returns pixel size in um for indexing
        """
        self.print_warning('pixel_scale_size()')

    def pixel_area_array(self):
        """ Returns array of pixel relative areas of shape=[rows, cols]
        """
        self.print_warning('pixel_area_array()')

    def pixel_size_array(self, axis):
        """ Returns array of pixel size in um for AXIS
        """
        self.print_warning('pixel_size_array(axis)')

    def pixel_coord_array(self, axis):
        """ Returns array of segment pixel coordinates in um for AXIS
        """
        self.print_warning('pixel_coord_array(axis)')

    def pixel_coord_min(self, axis):
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS
        """
        self.print_warning('pixel_coord_min(axis)')

    def pixel_coord_max(self, axis):
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS
        """
        self.print_warning('pixel_coord_max(axis)')

    def pixel_mask_array(self, mbits=0o377, **kwa):
        """ Returns array of masked pixels which content depends on bontrol bitword mbits
        """
        self.print_warning('pixel_mask_array(mask_bits)')

    def return_switch(self, meth, axis=None):
        """ Returns three x,y,z arrays if axis=None, or single array for specified axis 
        """
        if axis is None: return meth()
        else           : return dict(zip(self.AXIS, meth()))[axis]

#----------
# 2020-07 added for converter

    def asic0indices(self): self.print_warning('asic0indices')
    def asic_rows_cols(self): self.print_warning('asic_rows_cols')
    def number_of_asics_in_rows_cols(self): self.print_warning('number_of_asics_in_rows_cols')
    def name(self): self.print_warning('name')
  
#------------------------------

if __name__ == "__main__":

    import sys

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)
    logger.debug('Module %s describes interface methods for segment pixel geometry' % sys.argv[0])

    sg = SegGeometry()
    sg.print_seg_info()
    sg.size()
    sys.exit('End of %s' % sys.argv[0])

#------------------------------

