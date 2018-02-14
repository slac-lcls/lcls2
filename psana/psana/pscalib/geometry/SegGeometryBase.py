#### !/usr/bin/env python
#------------------------------
"""
:py:class:`SegGeometryBase` - abstract class with interface description
=======================================================================

Methods of this class should be re-implemented in derived classes with name pattern SegGeometry<SensorVers> 
for pixel geometry description of all sensors.
For example, CSPAD 2x1 sensor is implemented in class :py:class:`pscalib.geometry.SegGeometryCspad2x1V1`.
Access to all implemented sensors is available through the factory method in 
class :py:class:`pscalib.geometry.SegGeometryStore`.

Usage::

    from psana.pscalib.geometry.SegGeometryCspad2x1V1 import cspad2x1_one as sg

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
    ...
    print('X.shape =', X.shape)

See:
 * :py:class:`SegGeometryBase`
 * :py:class:`SegGeometryCspad2x1V1`
 * :py:class:`SegGeometryEpix100V1`
 * :py:class:`SegGeometryMatrixV1`
 * :py:class:`SegGeometryJungfrauV1`
 * :py:class:`SegGeometryStore`
 * :py:class:`GeometryAccess`
 * :py:class:`GeometryObject`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-01
"""
#------------------------------

#import sys
#import os
#import math
#import numpy as np

#------------------------------

def rotation(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

class SegGeometryBase :
    AXIS = ['X', 'Y', 'Z']
    DIC_AXIS = {'X':0, 'Y':1, 'Z':2}
    wmsg = 'WARNING! interface method SegGeometryBase.%s needs to be re-implemented in the derived class'

    def __init__(self) : 
        pass

#------------------------------

    def _show_warning(self, methname) :
        print(self.wmsg % methname)

    def print_seg_info(self, pbits=0) :
        """ Prints segment info for selected bits
        """
        self._show_warning('print_seg_info(pbits=0)')

    def size(self) :
        """ Returns segment size - total number of pixels in segment
        """
        self._show_warning('size()')

    def shape(self) :
        """ Returns shape of the segment [rows, cols]
        """
        self._show_warning('shape()')

    def rows(self) :
        """ Returns number of rows in segment
        """
        self._show_warning('rows()')

    def cols(self) :
        """ Returns number of cols in segment
        """
        self._show_warning('cols()')

    def pixel_scale_size(self) :
        """ Returns pixel size in um for indexing
        """
        self._show_warning('pixel_scale_size()')

    def pixel_area_array(self) :
        """ Returns array of pixel relative areas of shape=[rows, cols]
        """
        self._show_warning('pixel_area_array()')

    def pixel_size_array(self, axis) :
        """ Returns array of pixel size in um for AXIS
        """
        self._show_warning('pixel_size_array(axis)')

    def pixel_coord_array(self, axis) :
        """ Returns array of segment pixel coordinates in um for AXIS
        """
        self._show_warning('pixel_coord_array(axis)')

    def pixel_coord_min(self, axis) :
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS
        """
        self._show_warning('pixel_coord_min(axis)')

    def pixel_coord_max(self, axis) :
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS
        """
        self._show_warning('pixel_coord_max(axis)')

    def pixel_mask_array(self, mbits=0) :
        """ Returns array of masked pixels which content depends on bontrol bitword mbits
        """
        self._show_warning('pixel_mask_array(mask_bits)')

    def return_switch(sp, meth, axis=None) :
        """ Returns three x,y,z arrays if axis=None, or single array for specified axis 
        """
        if axis is None : return meth()
        else            : return dict(zip(sp.AXIS, meth()))[axis]
  
#------------------------------

if __name__ == "__main__" :
    import sys
    print('%s\nModule %s describes interface methods for segment pixel geometry'%(54*'_',sys.argv[0]))
    sg = SegGeometryBase()
    sg.print_seg_info()
    sg.size()
    sg.shape()
    sg.cols()
    sg.pixel_mask_array()
    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
#------------------------------
#------------------------------


