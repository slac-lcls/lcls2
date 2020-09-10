#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`GeometryAccess` - holds and access hierarchical geometry for generic pixel detector
====================================================================================================

Usage::
 
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays

    fname_geometry = '/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    geometry = GeometryAccess(fname_geometry)

    # load constants from geometry file
    geometry.load_pars_from_file(path=None)

    # load constants from next-line-symbol separated str / text object
    geometry.load_pars_from_str(s)

    # check if geometry info is available, returns bool
    status = geometry.is_valid()

    # get pixel coordinate [um] arrays
    X, Y, Z = geometry.get_pixel_coords(oname=None, oindex=0, do_tilt=True, cframe=0)

    # get pixel x,y coordinate [um] arrays projected toward origin on specified zplane, if zplane=None then zplane=Z.mean()
    X, Y = geometry.get_pixel_xy_at_z(zplane=None, oname=None, oindex=0, do_tilt=True)

    # print a portion of pixel X, Y, and Z coordinate arrays
    geometry.print_pixel_coords(oname=None, oindex=0)

    # get pixel area array; A=1 for regular pixels, =2.5 for wide.
    area = geometry.get_pixel_areas(oname=None, oindex=0)

    # returns (smallest) pixel size [um]
    pixel_size = geometry.get_pixel_scale_size(oname=None, oindex=0)

    # returns dictionary of comments associated with geometry (file)
    dict_of_comments = geometry.get_dict_of_comments()

    # print comments associated with geometry (file) 
    geometry.print_comments_from_dict()

    # print list of geometry objects
    geometry.print_list_of_geos()

    # print list of geometry object children
    geometry.print_list_of_geos_children()

    # get pixel mask array;
    # mbits = +1-mask edges, +2-wide pixels, +4-non-bonded pixels, +8/+16 - four/eight neighbours of non-bonded
    mask = geometry.get_pixel_mask(oname=None, oindex=0, mbits=0o377)

    # get image martix index arrays for entire detector
    rows, cols = geometry.get_pixel_coord_indexes(do_tilt=True, cframe=0)

    # get image martix index arrays for specified quad with offset
    rows, cols = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=(1000,1000), do_tilt=True, cframe=0)

    # get image martix index arrays for pixel coordinates projected toward origin on specified zplane
    rows, cols = geometry.get_pixel_xy_inds_at_z(zplane=None, oname=None, oindex=0, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0)

    # get image martix irow and icol indexes for specified point in [um]. By default p_um=(0,0) - detector origin coordinates (center).
    irow, icol = geometry.point_coord_indexes(p_um=(0,0))
    # all other parameters should be the same as in get_pixel_coord_indexes method
    irow, icol = geometry.point_coord_indexes(p_um=(0,0), 'QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=(1000,1000), do_tilt=True, cframe=0)

    # get 2-d image from index arrays
    img = img_from_pixel_arrays(rows,cols,W=arr)

    # Get specified object of the class GeometryObject, all objects are kept in the list self.list_of_geos
    geo = geometry.get_geo('QUAD:V1', 1) 
    # Get top GeometryObject - the object which includes all other geometry objects
    geo = geometry.get_top_geo()

    # modify currect geometry objects' parameters
    geometry.set_geo_pars('QUAD:V1', 1, x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)
    geometry.move_geo('QUAD:V1', 1, 10, 20, 0)
    geometry.tilt_geo('QUAD:V1', 1, 0.01, 0, 0)

    # save current geometry parameters in file
    geometry.save_pars_in_file(fname_geometry_new)

    # DEPRECATED change verbosity bit-control word; to print everythisg use pbits = 0xffff
    geometry.set_print_bits(pbits=0o377)

    # return geometry parameters in "psf" format as a tuple psf[32][3][3]
    psf = geometry.get_psf()
    geometry.print_psf()

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

Author: Mikhail Dubrovin
"""
#------------------------------

import os
import numpy as np
from math import floor, fabs

from psana.pscalib.geometry.GeometryObject import GeometryObject

import logging
logger = logging.getLogger(__name__)

#------------------------------

def divide_protected(num, den, vsub_zero=0):
    """Returns result of devision of numpy arrays num/den with substitution of value vsub_zero for zero den elements.
    """
    pro_num = np.select((den!=0,), (num,), default=vsub_zero)
    pro_den = np.select((den!=0,), (den,), default=1)
    return pro_num / pro_den

#------------------------------

class GeometryAccess:
    """ :py:class:`GeometryAccess`
    """

    # DEPRECATED: def __init__(self, path=None, pbits=0, use_wide_pix_center=False):
    def __init__(self, *args, **kwargs):
        """Constructor of the class :py:class:`GeometryAccess`      

        Parameters

        - path : str - path to the geometry file
        - pbits : int - verbosity bitword
        """        
        self.path  = args[0] if len(args)>0 else kwargs.get('path', None)   # positional or optional argument
        self.pbits = args[1] if len(args)>1 else kwargs.get('pbits', 0)     # deprecated, but backward compatable
        self.use_wide_pix_center = kwargs.get('use_wide_pix_center', False) # optional only
        self.valid = False

        if self.path is None or not os.path.exists(self.path):
            logger.warning('%s: geometry file "%s" does not exist' % (self.__class__.__name__, path))
            return

        self.load_pars_from_file()

        if logger.getEffectiveLevel() == logging.DEBUG:  # or logger.root.level
            self.print_list_of_geos()
            self.print_list_of_geos_children()
            self.print_comments_from_dict()
    
    #------------------------------

    def reset_cash(self):
        # Parameters for caching
        self.geo_old    = None
        self.oname_old  = None
        self.oindex_old = None
        self.tilt_old   = None
        self.X_old      = None
        self.Y_old      = None
        self.Z_old      = None
        self.rows_old   = None
        self.cols_old   = None
        self.irow_old   = None
        self.icol_old   = None
        self.p_um_old   = None
        self.cframe_old = None

    #------------------------------

    def is_valid(self):
        """Returns True if geometry is loaded and presumably valid, otherwise False.
        """
        return self.valid

    #------------------------------

    def load_pars_from_file(self, path=None):
        """Reads input "geometry" file, discards empty lines and comments, fills the list of geometry objects for data lines.
        """        
        self.valid = False
        if path is not None: self.path = path
            
        self.reset_cash()
        self.dict_of_comments = {}
        self.list_of_geos = []

        logger.debug('Load file: %s' % self.path)

        f=open(self.path,'r')
        for linef in f:
            line = linef.strip('\n')
            logger.debug(line)
            if not line.strip(): continue # discard empty strings
            if line[0] == '#':            # process line of comments
                self._add_comment_to_dict(line)
                continue
            #geo=self._parse_line(line)
            self.list_of_geos.append(self._parse_line(line))
    
        f.close()
    
        self._set_relations()
        self.valid = True
    
    #------------------------------

    def load_pars_from_str(self, s):
        """Reads input geometry from str, discards empty lines and comments, fills the list of geometry objects for data lines.
        """        
        self.valid = False
        if not isinstance(s, str):
            logger.debug('%s.load_pars_from_str input parameter is not a str, s: %s' % (self.__class__.__name__, str(s)))
            return
            
        self.reset_cash()
        self.dict_of_comments = {}
        self.list_of_geos = []

        logger.debug('Load text: %s' % s)

        for linef in s.split('\n'):
            line = linef.strip('\n')
            logger.debug(line)
            if not line: continue   # discard empty strings
            if line[0] == '#':      # process line of comments
                self._add_comment_to_dict(line)
                continue
            #geo=self._parse_line(line)
            self.list_of_geos.append(self._parse_line(line))
    
        self._set_relations()
        self.valid = True
    
    #------------------------------

    def save_pars_in_file(self, path):
        """Save geometry file with current content.
        """        
        if not self.valid: return

        logger.info('Save file: %s' % path)

        txt = ''
        # save comments
        for k in sorted(self.dict_of_comments):
            #txt += '# %10s  %s\n' % (k.ljust(10), self.dict_of_comments[k])
            txt += '# %s\n' % (self.dict_of_comments[k])

        txt += '\n'        

        # save data
        for geo in self.list_of_geos:
            if geo.get_parent_name() is None: continue
            txt += '%s\n' % (geo.str_data())

        f=open(path,'w')
        f.write(txt)
        f.close()

        logger.debug(txt)
    
    #------------------------------
    
    def _add_comment_to_dict(self, line):
        """Splits the line of comments for keyward and value and store it in the dictionary.
        """
        #cmt = line.lstrip('# ').split(' ', 1)
        cmt = line.lstrip('#').lstrip(' ')
        if len(cmt)<1: return
        ind = len(self.dict_of_comments)
        if len(cmt)==1:
            #self.dict_of_comments[cmt[0]] = ''
            self.dict_of_comments[ind] = ''
            return

        #beginline, endline = cmt
        #print '  cmt     : "%s"' % cmt
        #print '  len(cmt): %d' % len(cmt)        
        #print '  line    : "%s"' % line

        self.dict_of_comments[ind] = cmt.strip()

    #------------------------------
    
    def _parse_line(self, line):
        """Gets the string line with data from input file,
           creates and returns the geometry object for this string.
        """
        keys = ['pname','pindex','oname','oindex','x0','y0','z0','rot_z','rot_y','rot_x','tilt_z','tilt_y','tilt_x']
        f = line.split()
        if len(f) != len(keys):
            logger.warning('The list length for fields from file: %d is not equal to expected: %d' % (len(f), len(keys)))
            return
    
        vals = [str  (f[0]),
                int  (f[1]),
                str  (f[2]),
                int  (f[3]),
                float(f[4]),
                float(f[5]),
                float(f[6]),
                float(f[7]),
                float(f[8]),
                float(f[9]),
                float(f[10]),
                float(f[11]),
                float(f[12])
               ]

        # catch positive Z for IP - in the image-matrix psana frame (z opposite to beam) 
        # detector Z relative to IP should always be negative. 
        if vals[0][:2]=='IP' and vals[6]>0: vals[6]=-vals[6]
    
        d = dict(zip(keys, vals))
        d['use_wide_pix_center'] = self.use_wide_pix_center
        return GeometryObject(**d)
    
    #------------------------------
    
    def _find_parent(self, geobj):
        """Finds and returns parent for geobj geometry object.
        """
        for geo in self.list_of_geos:
            if geo == geobj: continue
            if  geo.oindex == geobj.pindex \
            and geo.oname  == geobj.pname:
                return geo
    
        # The name of parent object is not found among geo names in the self.list_of_geos
        # add top parent object to the list
        if geobj.pname is not None:
            top_parent = GeometryObject(pname=None, pindex=0, oname=geobj.pname, oindex=geobj.pindex,\
                                        use_wide_pix_center=self.use_wide_pix_center)
            self.list_of_geos.append(top_parent)
            return top_parent
                   
        return None # for top parent itself
       
    #------------------------------

    def _set_relations(self):
        """Set relations between geometry objects in the list_of_geos.
        """
        for geo in self.list_of_geos:
            #geo.print_geo()
            parent = self._find_parent(geo)        

            if parent is None: continue

            geo.set_parent(parent)
            parent.add_child(geo)

            logger.debug('geo:%s:%d has parent:%s:%d' % (geo.oname, geo.oindex, parent.oname, parent.oindex))

    #------------------------------

    def get_geo(self, oname, oindex):
        """Returns specified geometry object.
        """
        if not self.valid: return None

        if oindex == self.oindex_old and oname == self.oname_old: return self.geo_old

        for geo in self.list_of_geos:
            if  geo.oindex == oindex \
            and geo.oname  == oname:
                self.oindex_old = oindex
                self.oname_old  = oname
                self.geo_old    = geo
                return geo
        return None
    
    #------------------------------
    
    def get_top_geo(self):
        """Returns top geometry object.
        """
        if not self.valid: return None
        return self.list_of_geos[-1]
    
    #------------------------------

    def coords_psana_to_lab_frame(self, x, y, z):
        """ Switches arrays of pixel coordinates between psana <-(symmetric transformation)-> lab frame 
            returns x,y,z pixel arrays in the lab coordinate frame.
            cframe [int] = 0 - default psana frame for image-matrix from open panel side X-rows, Y-columns, Z-opposite the beam
                         = 1 - LAB frame - Y-top (-g - opposite to gravity) Z-along the beam, X=[YxZ]
        """ 
        return np.array(-y), np.array(-x), np.array(-z)

    #------------------------------

    def coords_lab_to_psana_frame(self, x, y, z):
        """Forth-back-symmetric transformation."""
        return np.array(-y), np.array(-x), np.array(-z)
 
    #------------------------------

    def get_pixel_coords(self, oname=None, oindex=0, do_tilt=True, cframe=0):
        """Returns three pixel X,Y,Z coordinate arrays for top or specified geometry object.
        """
        if not self.valid: return None

        if  oindex  == self.oindex_old\
        and oname   == self.oname_old\
        and do_tilt == self.tilt_old\
        and cframe  == self.cframe_old:
            return self.X_old, self.Y_old, self.Z_old

        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug('get_pixel_coords(...) for geo:')
            geo.print_geo_children();
        
        x,y,z = geo.get_pixel_coords(do_tilt) 
        self.X_old, self.Y_old, self.Z_old = self.coords_psana_to_lab_frame(x,y,z) if cframe>0 else (x,y,z)
        self.tilt_old = do_tilt
        self.cframe_old = cframe
        return self.X_old, self.Y_old, self.Z_old

    #------------------------------

    def get_pixel_xy_at_z(self, zplane=None, oname=None, oindex=0, do_tilt=True, cframe=0):
        """Returns pixel coordinate arrays XatZ, YatZ, for specified zplane and geometry object.

           This method projects pixel X, Y coordinates in 3-D
           on the specified Z plane along direction to origin.
        """
        if not self.valid: return None, None

        X, Y, Z = self.get_pixel_coords(oname, oindex, do_tilt, cframe)
        if X is None: return None, None
        Z0 = Z.mean() if zplane is None else zplane
        if fabs(Z0) < 1000: return X, Y

        XatZ = Z0 * divide_protected(X,Z)
        YatZ = Z0 * divide_protected(Y,Z)
        return XatZ, YatZ

    #------------------------------

    def get_pixel_areas(self, oname=None, oindex=0):
        """Returns pixel areas array for top or specified geometry object.
        """
        if not self.valid: return None
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.get_pixel_areas()

    #------------------------------

    def get_pixel_mask(self, oname=None, oindex=0, mbits=0o377, **kwargs):
        """Returns pixel mask array for top or specified geometry object.

        mbits =+1 - mask edges
               +2 - two wide-pixel central columns
               +4 - non-bonded pixels
               +8 - four nearest neighbours of non-bonded pixels
               +16- eight neighbours of non-bonded pixels
        """
        if not self.valid: return None
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.get_pixel_mask(mbits=mbits, **kwargs)

    #------------------------------

    def get_pixel_scale_size(self, oname=None, oindex=0):
        """Returns pixel scale size for top or specified geometry object.
        """
        if not self.valid: return None
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)        
        return geo.get_pixel_scale_size()

    #------------------------------
    
    def get_dict_of_comments(self):
        """Returns dictionary of comments.
        """
        if not self.valid: return None
        return self.dict_of_comments

    #------------------------------

    def set_geo_pars(self, oname=None, oindex=0, x0=0, y0=0, z0=0, rot_z=0, rot_y=0, rot_x=0, tilt_z=0, tilt_y=0, tilt_x=0):
        """Sets geometry parameters for specified or top geometry object.
        """
        if not self.valid: return None
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.set_geo_pars(x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)

    #------------------------------

    def move_geo(self, oname=None, oindex=0, dx=0, dy=0, dz=0):
        """Moves specified or top geometry object by dx, dy, dz.
        """
        if not self.valid: return None
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.move_geo(dx, dy, dz)

    #------------------------------

    def tilt_geo(self, oname=None, oindex=0, dt_x=0, dt_y=0, dt_z=0):
        """Tilts specified or top geometry object by dt_x, dt_y, dt_z.
        """
        if not self.valid: return None
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.tilt_geo(dt_x, dt_y, dt_z)

    #------------------------------
    
    def print_list_of_geos(self):
        s = 'print_list_of_geos():'
        if len(self.list_of_geos) == 0:  s += ' List_of_geos is empty...'
        logger.info(s)
        if not self.valid: return
        for geo in self.list_of_geos: geo.print_geo()

    #------------------------------
    
    def print_list_of_geos_children(self):
        s = 'print_list_of_geos_children():'
        if len(self.list_of_geos) == 0: s += ' List_of_geos is empty...'
        logger.info(s)
        if not self.valid: return
        for geo in self.list_of_geos: geo.print_geo_children()

    #------------------------------
    
    def print_comments_from_dict(self):
        s = '\nprint_comments_from_dict():'
        if not self.valid: return
        #for k,v in self.dict_of_comments.iteritems():
        for k in sorted(self.dict_of_comments):
            s += '\n  key: %3d  val: %s' % (k, self.dict_of_comments[k])
        logger.info(s)

    #------------------------------

    def print_pixel_coords(self, oname=None, oindex=0, cframe=0):
        """Partial print of pixel coordinate X,Y,Z arrays for selected or top(by default) geo.
        """
        if not self.valid: return
        X, Y, Z = self.get_pixel_coords(oname, oindex, do_tilt=True, cframe=cframe)

        s = 'size=%d' % X.size
        s += '\n X: %s...'% ', '.join(['%10.1f'%v for v in X.flatten()[0:9]])
        s += '\n Y: %s...'% ', '.join(['%10.1f'%v for v in Y.flatten()[0:9]])
        s += '\n Z: %s...'% ', '.join(['%10.1f'%v for v in Z.flatten()[0:9]])
        logger.info(s)

    #------------------------------
        
    def xy_to_rc_point(self, X, Y, p_um=(0,0), pix_scale_size_um=None, xy0_off_pix=None, cframe=0):
        if X is None or Y is None: return None, None

        x_um, y_um = self.p_um_old = p_um
        pix_size = self.get_pixel_scale_size() if pix_scale_size_um is None else pix_scale_size_um

        if cframe==1: #LAB frame z-along the beam, y-nodir, x=[y,z]
            xmin, ymax = X.min(), Y.max()
            if xy0_off_pix is not None:
                # Offset in pix -> um
                if xy0_off_pix[0]>0: xmin -= xy0_off_pix[0] * pix_size
                if xy0_off_pix[1]>0: ymax += xy0_off_pix[1] * pix_size
            xmin, ymax = xmin-pix_size/2, ymax+pix_size/2
            return int(floor((ymax-y_um)/pix_size)), int(floor((x_um-xmin)/pix_size))

        else: # PSANA image-matrix frame - x-along gravity(rows), y-right(columns), z=[x,y]-opposite to the beam
            xmin, ymin = X.min(), Y.min()
            if xy0_off_pix is not None:
                # Offset in pix -> um
                if xy0_off_pix[0]>0: xmin -= xy0_off_pix[0] * pix_size
                if xy0_off_pix[1]>0: ymin -= xy0_off_pix[1] * pix_size
            xmin, ymin = xmin-pix_size/2, ymin-pix_size/2
            return int(floor((x_um-xmin)/pix_size)), int(floor((y_um-ymin)/pix_size))

    #------------------------------

    def xy_to_rc_arrays(self, X, Y, pix_scale_size_um=None, xy0_off_pix=None, cframe=0):
        """Returns image martix rows and columns arrays evaluated from X,Y coordinate arrays.
        """
        if X is None or Y is None: return None, None

        pix_size = self.get_pixel_scale_size() if pix_scale_size_um is None else pix_scale_size_um

        if cframe>0: #LAB frame z-along the beam, y-nodir, x=[y,z]
            xmin, ymax = X.min(), Y.max()
            if xy0_off_pix is not None:
                # Offset in pix -> um
                if xy0_off_pix[0]>0: xmin -= xy0_off_pix[0] * pix_size
                if xy0_off_pix[1]>0: ymax += xy0_off_pix[1] * pix_size
            xmin, ymax = xmin-pix_size/2, ymax+pix_size/2
            return np.array((ymax-Y)/pix_size, dtype=np.uint), np.array((X-xmin)/pix_size, dtype=np.uint)

        else: # PSANA image-matrix frame - x-along gravity(rows), y-right(columns), z=[x,y]-opposite to the beam
            xmin, ymin = X.min(), Y.min()
            if xy0_off_pix is not None:
                # Offset in pix -> um
                if xy0_off_pix[0]>0: xmin -= xy0_off_pix[0] * pix_size
                if xy0_off_pix[1]>0: ymin -= xy0_off_pix[1] * pix_size
            xmin, ymin = xmin-pix_size/2, ymin-pix_size/2
            return np.array((X-xmin)/pix_size, dtype=np.uint), np.array((Y-ymin)/pix_size, dtype=np.uint)

    #------------------------------

    def get_pixel_coord_indexes(self, oname=None, oindex=0, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0):
        """Returns image martix rows and columns arrays evaluated from X,Y coordinate arrays for top or specified geometry object.
        """
        if not self.valid: return None, None

        if  oindex  == self.oindex_old\
        and oname   == self.oname_old\
        and do_tilt == self.tilt_old\
        and cframe  == self.cframe_old\
        and pix_scale_size_um is None\
        and xy0_off_pix is None\
        and self.rows_old is not None:
            return self.rows_old, self.cols_old

        X, Y, Z = self.get_pixel_coords(oname, oindex, do_tilt, cframe)
        self.rows_old, self.cols_old = self.xy_to_rc_arrays(X, Y, pix_scale_size_um, xy0_off_pix, cframe)
        return self.rows_old, self.cols_old

    #------------------------------

    def get_pixel_xy_inds_at_z(self, zplane=None, oname=None, oindex=0, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0):
        """Returns pixel coordinate index arrays rows, cols of size for specified zplane and geometry object.
        """
        if not self.valid: return None, None
        X, Y = self.get_pixel_xy_at_z(zplane, oname, oindex, do_tilt, cframe)
        self.rows_old, self.cols_old = self.xy_to_rc_arrays(X, Y, pix_scale_size_um, xy0_off_pix, cframe)
        return self.rows_old, self.cols_old

    #------------------------------

    def point_coord_indexes(self, p_um=(0,0), oname=None, oindex=0, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0):
        """Converts point (x_um, y_um) corrdinates [um] to pixel (row, col) indexes.
           All other parameters are the same as in get_pixel_coord_indexes.
           WARNING: indexes are not required to be inside the image. They are integer, may be negative or exceed pixel maximal index.
        """
        if not self.valid: return None, None

        if  oindex  == self.oindex_old\
        and oname   == self.oname_old\
        and do_tilt == self.tilt_old\
        and p_um    == self.p_um_old\
        and pix_scale_size_um is None\
        and xy0_off_pix is None\
        and self.irow_old is not None:
            return self.irow_old, self.icol_old

        X, Y, Z = self.get_pixel_coords(oname, oindex, do_tilt, cframe)
        self.irow_old, self.icol_old = self.xy_to_rc_point(X, Y, p_um, pix_scale_size_um, xy0_off_pix, cframe)
        return self.irow_old, self.icol_old

    #------------------------------

    def set_print_bits(self, pbits=0):
        """ Sets printout control bitword.
        """
        self.pbits = pbits

    #------------------------------

    def get_psf(self):
        """Returns array of vectors in CrystFEL format (psf stands for position-slow-fast vectors).
        """
        if not self.valid: return None
        X, Y, Z = self.get_pixel_coords() # pixel positions for top level object
        if X.size != 32*185*388: return None
        # For now it works for CSPAD only
        shape_cspad = (32,185,388)
        X.shape, Y.shape, Z.shape,  = shape_cspad, shape_cspad, shape_cspad

        psf = []

        for s in range(32):
            vp = (X[s,0,0], Y[s,0,0], Z[s,0,0])

            vs = (X[s,1,0]-X[s,0,0], \
                  Y[s,1,0]-Y[s,0,0], \
                  Z[s,1,0]-Z[s,0,0])

            vf = (X[s,0,1]-X[s,0,0], \
                  Y[s,0,1]-Y[s,0,0], \
                  Z[s,0,1]-Z[s,0,0])

            psf.append((vp,vs,vf))

        return psf

    #------------------------------

    def print_psf(self):
        """ Gets and prints psf array for test purpose.
        """
        if not self.valid: return None
        psf = np.array(self.get_psf())
        s = 'print_psf(): psf.shape: %s \npsf vectors:' % (str(psf.shape)) 
        for (px,py,pz), (sx,xy,xz), (fx,fy,fz) in psf:
            s += '\n    p=(%12.2f, %12.2f, %12.2f),    s=(%8.2f, %8.2f, %8.2f)   f=(%8.2f, %8.2f, %8.2f)' \
                  % (px,py,pz,  sx,xy,xz,  fx,fy,fz)
        logger.info(s)

#------------------------------
#------ Global Method(s) ------
#------------------------------

def img_default(shape=(10,10), dtype = np.float32):
    """Returns default image.
    """
    arr = np.arange(shape[0]*shape[1], dtype=dtype)
    arr.shape = shape
    return arr

#------------------------------

def img_from_pixel_arrays(rows, cols, W=None, dtype=np.float32, vbase=0):
    """Returns image from rows, cols index arrays and associated weights W.
       Methods like matplotlib imshow(img) plot 2-d image array oriented as matrix(rows,cols).
    """
    if rows.size != cols.size \
    or (W is not None and rows.size !=  W.size):
        msg = 'img_from_pixel_arrays(): input array sizes are different;' \
            + ' rows.size=%d, cols.size=%d, W.size=%d' % (rows.size, cols.size, W.size)
        logger.warning(msg)
        return img_default()

    rowsfl = rows.flatten()
    colsfl = cols.flatten()

    rsize = int(rowsfl.max())+1 
    csize = int(colsfl.max())+1

    weight = W.flatten() if W is not None else np.ones_like(rowsfl)
    img = vbase*np.ones((rsize,csize), dtype=dtype)
    img[rowsfl,colsfl] = weight # Fill image array with data 
    return img

#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------

if __name__ == "__main__":
  from time import time # for test purpose only
  import psana.pyalgos.generic.Graphics as gg # for test purpose
  from psana.pyalgos.generic.NDArrGenerators import cspad_ndarr # for test purpose only

  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)
  #logger.setLevel(logging.DEBUG)
  #logger.getEffectiveLevel()
  #logger.root.level

#------------------------------

  def test_access(geometry):
    """ Tests geometry acess methods of the class GeometryAccess.
    """
    geometry.print_list_of_geos()
    geometry.print_list_of_geos_children()

    logger.info('TOP GEO:')
    top_geo = geometry.get_top_geo()
    top_geo.print_geo_children()

    logger.info('INTERMEDIATE GEO (QUAD):')
    geo = geometry.get_geo('QUAD:V1', 0) 
    #geo = geometry.get_top_geo() 
    geo.print_geo_children()

    t0_sec = time()
    X,Y,Z = geo.get_pixel_coords(do_tilt=True)
    #X,Y = geo.get_2d_pixel_coords()
    s = 'X: %s' % str(X)
    s+= '\n  Consumed time to get 3d pixel coordinates = %7.3f sec' % (time()-t0_sec)
    s+= '\n  Geometry object: %s:%d X.shape:%s' % (geo.oname, geo.oindex, str(X.shape))
    logger.info(s)

    logger.info('Test of print_pixel_coords() for quad:')
    geometry.print_pixel_coords('QUAD:V1', 1)
    logger.info('Test of print_pixel_coords() for CSPAD:')
    geometry.print_pixel_coords()

    s = 'Test of get_pixel_areas() for QUAD:'
    A = geo.get_pixel_areas()
    s+= '\n  Geometry object: %s:%d A.shape:%s' % (geo.oname, geo.oindex, str(A.shape))
    s+= '\n  A[0,0:5,190:198]:\n' + str(A[0,0:5,190:198])
    logger.info(s)
 
    s = 'Test of get_pixel_areas() for CSPAD:'
    A = top_geo.get_pixel_areas()
    s+= '\n  Geometry object: %s:%d A.shape:%s' % (geo.oname, geo.oindex, str(A.shape))
    s+= '\n  A[0,0,0:5,190:198]:\n' + str(A[0,0,0:5,190:198])
    logger.info(s)

    s = 'Test of get_size_geo_array()'
    s+= '\n  for QUAD: %d' % geo.get_size_geo_array()
    s+= '\n  for CSPAD: %d' % top_geo.get_size_geo_array()
    logger.info(s)

    s = 'Test of get_pixel_scale_size()'
    s+= '\n  for QUAD    : %8.2f' % geo.get_pixel_scale_size()
    s+= '\n  for CSPAD   : %8.2f' % top_geo.get_pixel_scale_size()
    s+= '\n  for geometry: %8.2f' % geometry.get_pixel_scale_size()
    logger.info(s)

    s = 'Test of get_dict_of_comments():'
    d = geometry.get_dict_of_comments()
    s+= '\n  d[0] = %s' % str(d[0])
    logger.info(s)

#------------------------------

  def test_plot_quad(geometry):
    """ Tests geometry acess methods of the class GeometryAccess object for CSPAD quad.
    """
    ## get index arrays
    rows, cols = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True)

    # get intensity array
    arr = cspad_ndarr(n2x1=rows.shape[0])
    arr.shape = (8,185,388)
    amp_range = (0,185+388)
 
    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

  def test_mask_quad(geometry, mbits):
    """ Tests geometry acess methods of the class GeometryAccess object for CSPAD quad.
    """
    ## get index arrays
    rows, cols = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True)

    # get intensity array
    arr = geometry.get_pixel_mask('QUAD:V1', 1, mbits)
    arr.shape = (8,185,388)
    amp_range = (-1,2)
 
    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows, cols, W=arr, vbase=0.5)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

  def test_plot_cspad(geometry, fname_data, amp_range=(0,0.5)):
    """ The same test as previous, but use get_pixel_coord_indexes(...) method.
    """
    #rad1 =  93
    #rad2 = 146
    rad1 = 655
    rad2 = 670

    # get pixel coordinate index arrays:
    xyc = xc, yc = 500, 500# None 

    #rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=None)
    rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc, do_tilt=True)

    ixo, iyo = geometry.point_coord_indexes(xy0_off_pix=xyc, do_tilt=True)
    logger.info('Detector origin indexes ixo:%d iyo:%d' % (ixo, iyo))

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 
    arr.shape= (4,8,185,388)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))

    arr.shape = rows.shape
    img = img_from_pixel_arrays(rows, cols, W=arr)

    rcc_ring = (iyo, ixo)
    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.drawCircle(axim, rcc_ring, rad1, linewidth=1, color='w', fill=False) 
    gg.drawCircle(axim, rcc_ring, rad2, linewidth=1, color='w', fill=False) 
    gg.drawCenter(axim, rcc_ring, rad1, linewidth=1, color='w') 
    gg.move(500,10)
    gg.show()

#------------------------------

  def test_img_default():
    """ Test default image.
    """
    axim = gg.plotImageLarge(img_default())
    gg.move(500,10)
    gg.show()

#------------------------------

  def test_init_is_silent():
    logger.info('Init GeometryAccess is silentin INFO level? (see below)')
    logger.setLevel(logging.INFO)
    ga0 = GeometryAccess(fname_geometry)

#------------------------------

  def test_save_pars_in_file(geometry):
    """ Test default image.
    """
    # change one line of parameters
    x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x = -3500, 5800, 0, 0.123, 0.123, 0.123, 1, 2, 3
    geometry.set_geo_pars('QUAD:V1', 1, x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)

    geometry.set_print_bits(32)
    fname = './test.txt'
    geometry.save_pars_in_file(fname)
    logger.info('saved file %s' % fname)

#------------------------------

  def test_load_pars_from_file(geometry):
    """ Test default image.
    """
    geometry.set_print_bits(32+64)
    geometry.load_pars_from_file('./test.txt')
    geometry.print_list_of_geos()

#------------------------------

  def test_cspad2x2():
    """ Test cspad2x2 geometry table.
    """
    basedir = '/reg/g/psdm/detector/alignment/cspad2x2/calib-cspad2x2-01-2013-02-13/'   
    fname_geometry = basedir + 'calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/0-end.data'
    fname_data     = basedir + 'cspad2x2.1-ndarr-ave-meca6113-r0028.dat'    

    geometry = GeometryAccess(fname_geometry, pbits=0o377, use_wide_pix_center=False)
    amp_range = (0,15000)

    # get pixel coordinate index arrays:
    #xyc = xc, yc = 1000, 1000
    #rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc)

    rows, cols = geometry.get_pixel_coord_indexes(do_tilt=True)

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 
    arr.shape= (185,388,2)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

  def test_epix100a():
    """ Test test_epix100a geometry table.
    """
    basedir = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/'    
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data'
    fname_data     = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.txt'    

    #basedir = '/reg/neh/home1/dubrovin/LCLS/GeometryCalib/calib-xpp-Epix100a-2014-11-05/'    
    #fname_geometry = basedir + 'calib/Epix100a::CalibV1/NoDetector.0:Epix100a.0/geometry/0-end.data'
    #fname_data     = basedir + 'epix100a-ndarr-ave-clb-xppi0614-r0073.dat'    

    geometry = GeometryAccess(fname_geometry, pbits=0o377)
    amp_range = (-4,10)

    rows, cols = geometry.get_pixel_coord_indexes()

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

  def test_cspad_xy_at_z():
    """ Test cspad geometry table.
    """
    ## 'CxiDs1.0:Cspad.0)' or 'DscCsPad' 
    basedir = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-camera1-2014-09-24/'    
    fname_geometry = basedir + '2016-06-03-geometry-cxi06216-r25-camera1-z175mm.txt'
    fname_data     = basedir + '2016-06-03-chun-cxi06216-0025-DscCsPad-max.txt'    

    geometry = GeometryAccess(fname_geometry, pbits=0o377)

    # get pixel coordinate index arrays:
    xyc = xc, yc = 1000, 1000
    #rows, cols = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc)
    #rows, cols = geometry.get_pixel_coord_indexes(do_tilt=True)
    #rows, cols = geometry.get_pixel_xy_inds_at_z(zplane=None, xy0_off_pix=xyc)
    rows, cols = geometry.get_pixel_xy_inds_at_z(zplane=150000)

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 

    #logger.info('arr.shape=', arr.shape
    arr.shape= (32,185,388)

    #ave, rms = arr.mean(), arr.std()
    #amp_range = (ave-rms, ave+3*rms)
    amp_range = (0, 1000)
    logger.info('amp_range:' + str(amp_range))

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------
#------------------------------
#------------------------------

  def usage(tname='0'):
    s = ''
    if tname in ('0',): s+='\n==== Usage: python %s <test-number>' % sys.argv[0]
    if tname in ('0', '1'): s+='\n 1 - test_access(geometry)'
    if tname in ('0', '2'): s+='\n 2 - test_plot_quad(geometry)'
    if tname in ('0', '3'): s+='\n 3 - test_plot_cspad(geometry, fname_data, amp_range)'
    if tname in ('0', '4'): s+='\n 4 - test_img_default()'
    if tname in ('0', '5'): s+='\n 5 - test_init_is_silent()'
    if tname in ('0', '6'): s+='\n 6 - ga0377 = GeometryAccess(fname_geometry, pbits=0o377)'
    if tname in ('0', '7'): s+='\n 7 - test_save_pars_in_file(geometry)'
    if tname in ('0', '8'): s+='\n 8 - test_load_pars_from_file(geometry)'
    if tname in ('0', '9'): s+='\n 9 - test_mask_quad(geometry, 1+2+8)'
    if tname in ('0','10'): s+='\n10 - geometry.print_psf()'
    if tname in ('0','11'): s+='\n11 - test_cspad2x2()'
    if tname in ('0','12'): s+='\n12 - test_epix100a()'
    if tname in ('0','13'): s+='\n13 - geometry.print_comments_from_dict()'
    if tname in ('0','14'): s+='\n14 - test_cspad_xy_at_z()'
    return s

#------------------------------

if __name__ == "__main__":

    import sys

    ##fname = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/1-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxi83714-r0136.dat'
    #amp_range = (0,0.5)

    # CXI
    basedir = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-03-19/'
    fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #fname_geometry = '/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    amp_range = (0,500)

    ## XPP
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-xpp-2013-01-29/'
    #fname_data     = basedir + 'cspad-xpptut13-r1437-nda.dat'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/XppGon.0:Cspad.0/geometry/0-end.data'
    #amp_range = (1500,2500)


    #basedir = '/home/pcds/LCLS/calib/geometry/'
    #fname_geometry = basedir + '0-end.data'
    #fname_geometry = basedir + '2-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    #fname_geometry = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #amp_range = (0,500)

    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data'
    #fname_data     = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.npy'
    #amp_range = (0,500)

    logger.info('%s\nfname_geometry: %s\nfname_data: %s' %(120*'_', fname_geometry, fname_geometry))

    geometry = GeometryAccess(fname_geometry)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv)==1: logger.info(usage())
    elif tname=='1': test_access(geometry)
    elif tname=='2': test_plot_quad(geometry)
    elif tname=='3': test_plot_cspad(geometry, fname_data, amp_range)
    elif tname=='4': test_img_default()
    elif tname=='5': test_init_is_silent()
    elif tname=='6': ga0377 = GeometryAccess(fname_geometry, pbits=0o377)
    elif tname=='7': test_save_pars_in_file(geometry)
    elif tname=='8': test_load_pars_from_file(geometry)
    elif tname=='9': test_mask_quad(geometry, 1+2+8)
    elif tname=='10': geometry.print_psf()
    elif tname=='11': test_cspad2x2()
    elif tname=='12': test_epix100a()
    elif tname=='13': geometry.print_comments_from_dict()
    elif tname=='14': test_cspad_xy_at_z()
    else: logger.warning('NON-EXPECTED TEST NAME: %s\n\n%s' % (tname, usage()))
    if len(sys.argv)>1: logger.info(usage(tname))
    sys.exit('END OF TEST')

#------------------------------
