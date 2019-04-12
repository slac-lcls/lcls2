
"""
TBD

Class :py:class:`geometry` supports tentative format of geometry constants for LCLS2
====================================================================================

Usage::
    from psana.pscalib.geonew.utils import object_from_python_code
    geometry_cspad = object_from_python_code('lcls2/psana/psana/pscalib/geonew/geometry_cspad.py', 'geometry_cspad')
    g = geometry_cspad()
    print('dir(g):\n', dir(g))
    print('str_geometry_code():\n', g.str_geometry_code(cmtext='Additional comment comes here.\n  '))

    coords = g.get_pixel_coords(TBD)

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-30 by Mikhail Dubrovin
"""

#----------

import numpy as np
from math import sqrt, radians, sin, cos

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(name)s %(lineno)d %(message)s', level=logging.DEBUG)

#----------

GEOFMT  = '%-11s %2d, %11s %2d, %8d, %8d, %8d, %8d, %8d, %8d, %8.3f, %8.3f, %8.3f'
GEOKFMT = '%8s  %s %8s %s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s'
GEOKEYS = ('pname','pindex','oname','oindex','x0','y0','z0','rot_z','rot_y','rot_x','tilt_z','tilt_y','tilt_x')

#----------

def rotation_cs(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

def rotation(X, Y, angle_deg) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot rotated by angle_deg
    """
    angle_rad = radians(angle_deg)
    S, C = sin(angle_rad), cos(angle_rad)
    return rotation_cs(X, Y, C, S)

#----------

class geometry_object :
    logger.debug('in geometry_objects')
    def __init__(self, geopars) :
        #logger.debug('in geometry_object.__init__')
        for k,v in zip(GEOKEYS, geopars) : setattr(self, k, v)

        # ---- for 2-nd stage of initialization
        self.parent = None
        self.list_of_children = []
        self.algo = None

    def set_parent(self, parent) :
        """ Set parent geometry object for self.
        """
        self.parent = parent

    def add_child(self, child) :
        """ Add children geometry object to the list of children.
        """
        self.list_of_children.append(child)

    def get_parent(self) :
        """ Returns parent geometry object.
        """
        return self.parent

    def get_list_of_children(self) :
        """ Returns list of children geometry objects.
        """
        return self.list_of_children

    def get_geo_name(self) :
        """ Returns self geometry object name.
        """
        return self.oname

    def get_geo_index(self) :
        """ Returns self geometry object index.
        """
        return self.oindex

    def get_parent_name(self) :
        """ Returns parent geometry object name.
        """
        return self.pname

    def get_parent_index(self) :
        """ Returns parent geometry object index.
        """
        return self.pindex

    def get_origin(self) :
        """ Returns object origin x, y, z coordinates [um] relative to parent frame.
        """
        return self.x0, self.y0, self.z0

    def get_rot(self) :
        """ Returns object tilt angles [degree] around z, y, and x axes, respectively.
        """
        return self.rot_z, self.rot_y, self.rot_x

    def get_tilt(self) :
        """ Returns object rotation angles [degree] around z, y, and x axes, respectively.
        """
        return self.tilt_z, self.tilt_y, self.tilt_x

    def transform_geo_coord_arrays(self, X, Y, Z, do_tilt=True) :
        """ Transforms geometry object coordinates to the parent frame.
        """
        angle_z = self.rot_z + self.tilt_z if do_tilt else self.rot_z
        angle_y = self.rot_y + self.tilt_y if do_tilt else self.rot_y
        angle_x = self.rot_x + self.tilt_x if do_tilt else self.rot_x

        X1, Y1 = rotation(X,  Y,  angle_z)
        Z2, X2 = rotation(Z,  X1, angle_y)
        Y3, Z3 = rotation(Y1, Z2, angle_x)

        Zt = Z3 + self.z0
        Yt = Y3 + self.y0
        Xt = X2 + self.x0

        return Xt, Yt, Zt 


    def get_pixel_coords(self, do_tilt=True) :
        """ Returns three numpy arrays with pixel X, Y, Z coordinates for self geometry object.
        """
        if self.algo is not None :
            xac, yac, zac = self.algo.pixel_coord_array()
            return self.transform_geo_coord_arrays(xac, yac, zac, do_tilt)

        xac, yac, zac = None, None, None
        for ind, child in enumerate(self.list_of_children) :
            if child.oindex != ind :
                logger.warning('Geometry object %s:%d has non-consequtive index in calibration file, reconst index:%d' % \
                      (child.oname, child.oindex, ind))

            xch, ych, zch = child.get_pixel_coords(do_tilt)

            if ind == 0 :
                xac = xch
                yac = ych
                zac = zch
            else :
                xac = np.vstack((xac, xch))
                yac = np.vstack((yac, ych))
                zac = np.vstack((zac, zch))

        # define shape for output x,y,z arrays
        shape_child = xch.shape
        len_child = len(self.list_of_children)
        geo_shape = np.hstack(([len_child], xch.shape))
        #print 'geo_shape = ', geo_shape        
        xac.shape = geo_shape
        yac.shape = geo_shape
        zac.shape = geo_shape
        X, Y, Z = self.transform_geo_coord_arrays(xac, yac, zac, do_tilt)
        return self.det_shape(X), self.det_shape(Y), self.det_shape(Z) 


    def str_header(self) :
        """Returns (str) formatted header for constants.
        """
        return GEOKFMT % GEOKEYS


    def str_geo_attribute(self, k) :
        v = getattr(self, k, None)
        if 'name' in k :
            return '"%s",'%v if isinstance(v,str) else '%s,' % v._objname
        return v
        

    def str_geometry_object(self) :
        """Returns a single (str) record with constants for calibration file.
        """
        return GEOFMT % tuple([self.str_geo_attribute(k) for k in GEOKEYS])


    def set_geo_pars(self,\
                     x0=0, y0=0, z0=0,\
                     rot_z=0, rot_y=0, rot_x=0,\
                     tilt_z=0, tilt_y=0, tilt_x=0) :
        """ Sets self object geometry parameters.
        """
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.rot_z  = rot_z  
        self.rot_y  = rot_y 
        self.rot_x  = rot_x 
                            
        self.tilt_z = tilt_z
        self.tilt_y = tilt_y
        self.tilt_x = tilt_x


    def move_geo(self, dx=0, dy=0, dz=0) :
        """ Adds offset for origin of the self object relative current position.
        """
        self.x0 += dx
        self.y0 += dy
        self.z0 += dz


    def tilt_geo(self, dt_x=0, dt_y=0, dt_z=0) :
        """ Tilts the self object relative current orientation.
        """
        self.tilt_x += dt_x
        self.tilt_y += dt_y
        self.tilt_z += dt_z

    def rotate_geo(self, dr_x=0, dr_y=0, dr_z=0) :
        """ Rotates the self object relative current orientation.
        """
        self.rot_x += dr_x
        self.rot_y += dr_y
        self.rot_z += dr_z

#----------

class geometry_access :
    logger.debug('in geometry_access')

    def __init__(self) :
        logger.debug('in geometry_access.__init__')
        self.list_of_geos = [geometry_object(rec) for rec in self.detparts]


    def str_geometry_code_body(self) :
        """Returns main body of the geometry code.
        """
        return '    (%s)'%('),\n    ('.join([o.str_geometry_object() for o in self.list_of_geos]))


    def str_geometry_code(self, cmtext='') :
        """Returns (str) code of the object to be save in geometry file.
        """
        hat = '\n#----------\n' +\
              '##### from utils import object_from_python_code\n' +\
              '##### SENS2X1 = object_from_python_code("geoseg_cspad2x1.py", "SENS2X1")\n' +\
              '#####\n' +\
              '#####\n' +\
              '#####\n' +\
              '##### global SENS2X1\n' +\
              '##### SENS2X1 = geoseg_cspad2x1()\n' +\
              '#----------\n\n' +\
              'from geometry_access import geometry_access\n' +\
              'class %s(geometry_access) :' % self.__class__.__name__
        doc  = self.__doc__ + cmtext
        hdr  = self.list_of_geos[0].str_header()
        body = self.str_geometry_code_body()
        return '%s\n  """%s"""\n  #%s\n  detparts = (\n%s\n  )' %\
               (hat, doc, hdr, body)


    def print_geometry_code_body(self) :
        print('%s' % self.list_of_geos[0].str_header())
        print(self.str_geometry_code_body())

#----------
