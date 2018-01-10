#!@PYTHON@
####!/usr/bin/env python
"""
Class :py:class:`Quaternion` works with quaternion rotations
============================================================

This software was developed in co-operation with Meng for analysis of data cxif5315.

See:
  - :py:class:`Graphics`
  - :py:class:`GlobalGraphics`
  - :py:class:`GlobalUtils`
  - :py:class:`NDArrGenerators`
  - :py:class:`Quaternion`
  - :py:class:`FiberAngles`
  - :py:class:`FiberIndexing`
  - :py:class:`PeakData`
  - :py:class:`PeakStore`
  - :py:class:`TDCheetahPeakRecord`
  - :py:class:`TDFileContainer`
  - :py:class:`TDGroup`
  - :py:class:`TDMatchRecord`
  - :py:class:`TDNodeRecord`
  - :py:class:`TDPeakRecord`
  - :py:class:`HBins`
  - :py:class:`HPolar`
  - :py:class:`HSpectrum`
  - :py:class:`RadialBkgd`
  - `Analysis of data for cxif5315 <https://confluence.slac.stanford.edu/display/PSDMInternal/Analysis+of+data+for+cxif5315>`_.

Created in April 2016 by Mikhail Dubrovin
"""
#------------------------------

import numpy as np
from math import sqrt, sin, cos, radians, degrees, fabs, atan2

ZERO_TOLERANCE = 1e-6

#------------------------------

def sin_cos(angle_deg) :
    """Returns sin and cos of angle_deg
    """
    angle_rad = radians(angle_deg)
    s, c = sin(angle_rad), cos(angle_rad)
    if fabs(s) < ZERO_TOLERANCE : s = 0
    if fabs(c) < ZERO_TOLERANCE : c = 0
    return s, c

#------------------------------
#------------------------------

class Quaternion :
    def __init__(self, w=1, x=0, y=0, z=0) :
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def str_obj(self, cmt='Quaternion w,x,y,z: ', fmt='%7.3f') :
        pfmt = '%s%s  %s  %s  %s' % (cmt, fmt, fmt, fmt, fmt)
        return pfmt % (self.w, self.x, self.y, self.z)

    def print_obj(self, cmt='Quaternion w,x,y,z: ', fmt='%7.3f') :
        print self.str_obj(cmt, fmt)

#------------------------------

class Vector :
    def __init__(self, x, y, z) :
        self.u = self.x = x
        self.v = self.y = y
        self.w = self.z = z

    def str_obj(self, cmt='Vector: ', fmt='%6.3f') :
        pfmt = '%s%s  %s  %s' % (cmt, fmt, fmt, fmt)
        return pfmt % (self.u, self.v, self.w)

    def print_obj(self, cmt='Vector: ', fmt='%6.3f') :
        print self.str_obj(cmt, fmt)

#------------------------------

class Matrix :
    def __init__(self, m00=1, m01=0, m02=0,\
                       m10=0, m11=1, m12=0,\
                       m20=0, m21=0, m22=1) :
        self.m00, self.m01, self.m02 = m00, m01, m02
        self.m10, self.m11, self.m12 = m10, m11, m12
        self.m20, self.m21, self.m22 = m20, m21, m22

    def str_obj(self, cmt='Matrix:\n', fmt='%6.3f') :
        pfmt = '%s%s  %s  %s\n%s  %s  %s\n%s  %s  %s'%\
               (cmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt)
        return pfmt %\
                 (self.m00, self.m01, self.m02,\
                  self.m10, self.m11, self.m12,\
                  self.m20, self.m21, self.m22)

    def print_obj(self, cmt='Matrix:\n', fmt='%6.3f') :
        print self.str_obj(cmt, fmt)

    def product(self, A, B) :
        self.m00 = A.m00*B.m00 + A.m01*B.m10 + A.m02*B.m20
        self.m01 = A.m00*B.m01 + A.m01*B.m11 + A.m02*B.m21
        self.m02 = A.m00*B.m02 + A.m01*B.m12 + A.m02*B.m22

        self.m10 = A.m10*B.m00 + A.m11*B.m10 + A.m12*B.m20
        self.m11 = A.m10*B.m01 + A.m11*B.m11 + A.m12*B.m21
        self.m12 = A.m10*B.m02 + A.m11*B.m12 + A.m12*B.m22

        self.m20 = A.m20*B.m00 + A.m21*B.m10 + A.m22*B.m20
        self.m21 = A.m20*B.m01 + A.m21*B.m11 + A.m22*B.m21
        self.m22 = A.m20*B.m02 + A.m21*B.m12 + A.m22*B.m22

    def as_list(self) :
        return [[self.m00, self.m01, self.m02],\
                [self.m10, self.m11, self.m12],\
                [self.m20, self.m21, self.m22]]

    def np_matrix(self, dtype=np.double) :
        return np.matrix(as_list, dtype, copy=True)

    def set_from_np_matrix(self, npm) :
        self.m00, self.m01, self.m02 = npm[0,0], npm[0,1], npm[0,2]
        self.m10, self.m11, self.m12 = npm[1,0], npm[1,1], npm[1,2]
        self.m20, self.m21, self.m22 = npm[2,0], npm[2,1], npm[2,2]

    def rotation_matrix_x(self, angle_deg) :
        s,c = sin_cos(angle_deg)
        self.m00, self.m01, self.m02 = 1, 0, 0
        self.m10, self.m11, self.m12 = 0, c,-s
        self.m20, self.m21, self.m22 = 0, s, c

    def rotation_matrix_y(self, angle_deg) :
        s,c = sin_cos(angle_deg)
        self.m00, self.m01, self.m02 = c, 0, s
        self.m10, self.m11, self.m12 = 0, 1, 0
        self.m20, self.m21, self.m22 =-s, 0, c

    def rotation_matrix_z(self, angle_deg) :
        s,c = sin_cos(angle_deg)
        self.m00, self.m01, self.m02 = c,-s, 0
        self.m10, self.m11, self.m12 = s, c, 0
        self.m20, self.m21, self.m22 = 0, 0, 1

    def rotation_matrix(self, alpha, beta, gamma) :
        rotx = Matrix()
        roty = Matrix()
        rotz = Matrix()
        rotzy= Matrix()
        rotx.rotation_matrix_x(gamma)
        roty.rotation_matrix_y(beta)
        rotz.rotation_matrix_z(alpha)        
        rotzy.product(rotz, roty)
        #rotx.print_obj('X-rotation matrix:')
        #roty.print_obj('Y-rotation matrix:')
        #rotz.print_obj('Z-rotation matrix:')
        #rotzy.print_obj('ZY-rotation matrix:')
        self.product(rotzy, rotx)


    def get_angles(self) :
        ang_x = atan2(self.m21, self.m22)
        ang_y = atan2(-self.m20, sqrt(self.m21*self.m21 + self.m22*self.m22))
        ang_z = atan2(self.m10, self.m00)
        return degrees(ang_x), degrees(ang_y), degrees(ang_z)

#------------------------------

def quaternion_modulus(q) :
    """q - quaternion

       If a quaternion represents a pure rotation, its modulus should be unity.
       Returns: the modulus of the given quaternion.
    """
    return sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)

#------------------------------

def normalise_quaternion(q) :
    """q - quaternion

       Rescales the quaternion such that its modulus is unity.

       Returns: the normalised version of q
    """
    mod = quaternion_modulus(q)

    w = q.w / mod
    x = q.x / mod
    y = q.y / mod
    z = q.z / mod

    return Quaternion(w, x, y, z)

#------------------------------

def random_quaternion() :
    """
       Returns: a randomly generated, normalised, quaternion 
    """
    w, x, y, z = 2.0*np.random.random((4,)) - 1.0
    q = Quaternion(w, x, y, z)
    return normalise_quaternion(q)

#------------------------------

def quaternion_valid(q, tol=0.001) :
    """q - quaternion

       Checks if the given quaternion is normalised.

       Returns: 1 if the quaternion is normalised, 0 if not.
    """
    qmod = quaternion_modulus(q)
    if (qmod > 1+tol) or (qmod < 1-tol) : return 0
    return 1

#------------------------------

def quat_rot(v, q) :
    """v - vector (in the form of a "struct rvec") 
       q - quaternion                             
                                                     
       Rotates a vector according to a quaternion.   
                                                     
       Returns: rotated vector vrot            
    """
    t01 = q.w*q.x
    t02 = q.w*q.y
    t03 = q.w*q.z
    t11 = q.x*q.x
    t12 = q.x*q.y
    t13 = q.x*q.z
    t22 = q.y*q.y
    t23 = q.y*q.z
    t33 = q.z*q.z

    u = (1.0 - 2.0 * (t22 + t33)) * v.u\
            + (2.0 * (t12 + t03)) * v.v\
            + (2.0 * (t13 - t02)) * v.w

    v =       (2.0 * (t12 - t03)) * v.u\
      + (1.0 - 2.0 * (t11 + t33)) * v.v\
            + (2.0 * (t01 + t23)) * v.w

    w =       (2.0 * (t02 + t13)) * v.u\
            + (2.0 * (t23 - t01)) * v.v\
      + (1.0 - 2.0 * (t11 + t22)) * v.w

    return Vector(u, v, w)

#------------------------------

def rotmatrix_from_quaternion(q) :
    """q - quaternion                             
                                                     
       Evaluates rotation matrix from quaternion.   
                                                     
       Returns: rotation matrix as an object Matrix            
    """
    t01 = q.w*q.x
    t02 = q.w*q.y
    t03 = q.w*q.z
    t11 = q.x*q.x
    t12 = q.x*q.y
    t13 = q.x*q.z
    t22 = q.y*q.y
    t23 = q.y*q.z
    t33 = q.z*q.z
    return Matrix(1 - 2*(t22 + t33),     2*(t12 - t03),     2*(t13 + t02),\
                      2*(t12 + t03), 1 - 2*(t11 + t33),     2*(t23 - t01),\
                      2*(t13 - t02),     2*(t23 + t01), 1 - 2*(t11 + t22))

    # Matrix from Thomas White
    #return Matrix(1 - 2*(t22 + t33),     2*(t12 + t03),     2*(t13 - t02),\
    #                  2*(t12 - t03), 1 - 2*(t11 + t33),     2*(t01 + t23),\
    #                  2*(t02 + t13),     2*(t23 - t01), 1 - 2*(t11 + t22))

#------------------------------

def quaternion_from_rotmatrix(m) :
    """m - 3-d rotation matrix, class Matrix 

       Evaluates quaternion from rotation matrix.
       Implemented as
       https://en.wikipedia.org/wiki/Rotation_matrix

       Returns: normalised quaternion.
    """
    Qxx, Qxy, Qxz = m.m00, m.m01, m.m02
    Qyx, Qyy, Qyz = m.m10, m.m11, m.m12
    Qzx, Qzy, Qzz = m.m20, m.m21, m.m22

    t = Qxx+Qyy+Qzz
    r = sqrt(1+t)
    if fabs(r)<ZERO_TOLERANCE : r = ZERO_TOLERANCE
    s = 0.5/r
    w = 0.5*r
    x = (Qzy-Qyz)*s
    y = (Qxz-Qzx)*s
    z = (Qyx-Qxy)*s

    return Quaternion(w,x,y,z)

#------------------------------

def quaternion_for_angles(ax, ay, az) :
    """Returns: quaternion for three input rotation angles [deg].
    """
    #print 'Angles around x,y,z:  %6.1f  %6.1f  %6.1f' % (ax, ay, az)
    m = Matrix()
    m.rotation_matrix(az, ay, ax)
    return quaternion_from_rotmatrix(m)

#------------------------------

def record_for_angles(ax, ay, az) :
    """Prints string like:
       Angles around x,y,z:    72.0    -3.5   176.4   quaternion: w,x,y,z:   0.007459   0.043148   0.586445   0.808804
    """
    print 'Angles around x,y,z:  %6.1f  %6.1f  %6.1f' % (ax, ay, az),
    m = Matrix()
    m.rotation_matrix(az, ay, ax)
    q = quaternion_from_rotmatrix(m)
    q.print_obj('  quaternion:', fmt='%9.6f')

#------------------------------
#------------------------------
#-----------  TEST  -----------
#------------------------------
#------------------------------

def test_quaternion(tname) :

    v1 = Vector(1,0,0)
    v2 = Vector(0,1,0)
    v3 = Vector(0,0,1)
    v1.print_obj()
    v2.print_obj()
    v3.print_obj()

    q1 = Quaternion()
    q1.print_obj()

    m1 = Matrix()
    m1.print_obj()

#------------------------------

def test_rotation_matrix(tname) :
    alpha, beta, gamma = 5, 5, 5 # angles degree
    m = Matrix()
    m.rotation_matrix(alpha, beta, gamma)
    m.print_obj('R3-rotation matrix:\n')

#------------------------------

def test_quaternion_from_rotation_matrix(tname) :
    #alpha, beta, gamma = 5, 5, 5 # angles degree
    ax, ay, az =0, 90, 90 # angles degree

    vfmt = '%9.6f'

    print 'Inpurt angles around x,y,z:  %.2f  %.2f  %.2f' % (ax, ay, az)
    m = Matrix()
    m.rotation_matrix(az, ay, ax)
    m.print_obj('R3-rotation matrix:\n', fmt=vfmt)
    q = quaternion_from_rotmatrix(m)
    q.print_obj('Associated with matrix quaternion:\n', fmt=vfmt)
    mq = rotmatrix_from_quaternion(q)
    mq.print_obj('R3-rotation matrix back from quaternion:\n', fmt=vfmt)
    axo, ayo, azo = mq.get_angles()
    print 'Output angles around x,y,z:  %.2f  %.2f  %.2f' % (axo, ayo, azo)

#------------------------------

def test_quaternion_table(tname) :
    ax, ay, az = 0, 0, 0
    for ax in range(0, -180, -30) :
        record_for_angles(ax, ay, az)

    ax, ay, az = 0, 0, 0
    for ay in range(0, 180, 30) :
        record_for_angles(ax, ay, az)

    ax, ay, az = 0, 0, 0
    for az in range(0, 180, 30) :
        record_for_angles(ax, ay, az)

#------------------------------

def test_quaternion_table_crystal(tname) :
    #ax, ay, az = 90, 0, 0
    ax, ay, az = 90, -10, 0
    #ax, ay, az = 90, -3.5, 0
    #ax, ay, az = 72, -3.5, 0
    #ax, ay, az = 108, 3.5, 0
    #ax, ay, az = 102, 0, 0
    #ax, ay, az = 90, 10, 0
    #ax, ay, az = 108, 3.5, 0
    #for az in range(0, 180, 30) :
    for az in range(0, 180, 1) :
        record_for_angles(ax, ay, az)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname

    if   tname == '0' : test_quaternion(tname)
    elif tname == '1' : test_rotation_matrix(tname)
    elif tname == '2' : test_quaternion_from_rotation_matrix(tname)
    elif tname == '3' : test_quaternion_table(tname)
    elif tname == '4' : test_quaternion_table_crystal(tname)
    else : sys.exit('Test %s is undefined' % tname)

    sys.exit('End of test %s' % tname)

#------------------------------
