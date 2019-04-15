#!/usr/bin/env python

"""
Class :py:class:`FiberAngles` - contains methods to evaluate angles in fiber diffraction experiments
====================================================================================================

Usage::
    # Import
    from pyimgalgos.FiberAngles import fraser, calc_phi, calc_beta, funcy

    # Fraser transformation:
    s12rot, s3rot, fraser_img = fraser(arr, beta_deg, dist_pix, center=None, oshape=(1500,1500))

    # HBins objects for Fraser transformation:
    qh_hbins, qv_hbins = fraser_bins(fraser_img, dist_pix)

    # Evaluation of fiber tilt angles beta phi (in the image plane) (in transverse to image plane).
    phi  = calc_phi (x1, y1, x2, y2, dist)
    beta = calc_beta(x1, y1, phi, dist)

    #Fit functions
    yarr = funcy(xarr, phi_deg, bet_deg)

    yarr = funcy_l1_v0(xarr, phi_deg, bet_deg, DoR=0.4765, sgnrt=-1.)
    (depric.) yarr = funcy_l1_v1(xarr, phi_deg, bet_deg, DoR=0.4765, sgnrt=-1.)

    yarr2 = funcy2(xarr, a, b, c)

    # Conversion methods
    qx, qy, qz = recipnorm(x, y, z) # returns q/fabs(k) components for 3-d point along k^prime.

    xe, ye = rqh_to_xy(re, qh) # Returns reciprocal (xe,ye) coordinates of the qh projection on Ewald sphere of radius re.
    xe, ye, ze = rqhqv_to_xyz(re, qh, qv) # Returns reciprocal (xe,ye,ze) coordinates of the qh,qv projection on Ewald sphere of radius re.

    # Commands to test in the release directory: 
    # python ./pyimgalgos/src/FiberAngles.py <test-id>
    # where
    # <test-id> = 1 - test of the Fraser transformation
    # <test-id> = 2 - test of the phi angle
    # <test-id> = 3 - test of the beta angle 
    
This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

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
  - `Analysis of data for cxif5315  <https://confluence.slac.stanford.edu/display/PSDMInternal/Analysis+of+data+for+cxif5315>`_.

Created in 2015 by Mikhail Dubrovin
"""

##-----------------------------
import numpy as np
import math # pi, sin, sqrt, ceil, floor, fabs
##-----------------------------

class Storage :
    """Storage class for local data exchange between methods.
    """
    def __init__(self) :
        pass

#------------------------------
sp = Storage() # singleton
##-----------------------------

def _fillimg(r,c,a) :
    """This method is used in fraser(...), is called in map(_fillimg, irows, icols, arr) and serves to fill image.
    """
    sp.image[r,c] += a
    sp.count[r,c] += 1
    return r
    
##-----------------------------

def fraser_xyz(x, y, z, beta_deg, k=1.0) :
    """Do Fraser transformation for of 3-d points given by x,y,z arrays for angle beta around x and
       returns horizontal and vertical components of the scattering vector in units of k (1(def)-relative or eV or 1/A).
       x,y-arrays representing image point coordinates, z-can be scalar - distance from sample to detector.
       x, y, and z should be in the same units; ex.: number of pixels (109.92um) or [um], angle is in degrees.
       Example: fraser_xy(x,y,909,10); (10 degrees at 909 pixel (100mm) distance)

       ASSUMPTION:
       - x,y,z    - [in] point coordinate arrays with originn in IP
       - beta_deg - [in] angle beta in degrees
       - k        - [in] scale factor, ex.: wave number in units of (eV or 1/A).
    """

    d = np.sqrt(x*x+y*y+z*z)
    s1 = x/d
    s2 = z/d - 1
    s3 = y/d

    cb = math.cos(math.radians(beta_deg))
    sb = math.sin(math.radians(beta_deg))

    s1rot = s1
    s2rot = s2 * cb - s3 * sb
    s3rot = s2 * sb + s3 * cb

    s12rot = np.sign(s1rot)*np.sqrt(np.square(s1rot) + np.square(s2rot))

    return s12rot*k, s3rot*k

##-----------------------------

def fraser(arr, beta_deg, z, center=None, oshape=(1500,1500)) :
    """Do Fraser correction for an array at angle beta and sample-to-detector distance z, given in
       number of pixels (109.92um), angle is in degrees.
       Example: fraser(array,10,909); (10 degrees at 100mm distance)

       ASSUMPTION:
       1. by default 2-d arr image center corresponds to (x,y) origin 
       - arr      - [in] 2-d image array
       - beta_deg - [in] angle beta in degrees
       - z        - [in] distance from sample to detector given in units of pixel size (110um)
       - center   - [in] center (row,column) location on image, which will be used as (x,y) origin 
       - oshape   - [in] ouitput image shape
    """

    sizex = arr.shape[0]
    sizey = arr.shape[1]

    xc, yc = center if center is not None else (sizex/2, sizey/2) 

    xarr = np.arange(math.floor(-xc), math.floor(sizex-xc))
    yarr = np.arange(math.floor(-yc), math.floor(sizey-yc))

    x,y = np.meshgrid(yarr, xarr) ### SWAPPED yarr, xarr to keep correct shape for grids

    d = np.sqrt(x*x+y*y+z*z)
    s1 = x/d
    s2 = z/d - 1
    s3 = y/d

    cb = math.cos(math.radians(beta_deg))
    sb = math.sin(math.radians(beta_deg))

    s1rot = s1
    s2rot = s2 * cb - s3 * sb
    s3rot = s2 * sb + s3 * cb

    s12rot = np.sqrt(np.square(s1rot) + np.square(s2rot))
    s12rot[:,1:int(math.floor(sizex-xc))] *= -1

    scale = float(z)

    s12rot = np.ceil(s12rot * scale)
    s3rot  = np.ceil(s3rot  * scale)

    orows, orows1 = oshape[0], oshape[0] - 1
    ocols, ocols1 = oshape[1], oshape[1] - 1
    
    icols = np.array(s12rot + math.ceil(ocols/2), dtype=np.int)
    irows = np.array(s3rot  + math.ceil(orows/2), dtype=np.int)

    irows = np.select([irows < 0, irows > orows1], [0,orows1], default=irows)
    icols = np.select([icols < 0, icols > ocols1], [0,ocols1], default=icols)

    sp.image = np.zeros(oshape, dtype=arr.dtype)
    sp.count = np.zeros(oshape, dtype=np.int)

    unused_lst = map(_fillimg, irows, icols, arr)

    #print 'arr.shape: ', arr.shape
    #print 's3rot.shape: ', s3rot.shape
    #print 's12rot.shape: ', s12rot.shape
    #print 'reciparr.shape: ', reciparr.shape
    #print 'count min=%d, max=%d' % (sp.count.min(), sp.count.max())    

    countpro = np.select([sp.count < 1], [-1], default=sp.count)
    reciparr = np.select([countpro > 0], [sp.image/countpro], default=0)

    return s12rot, s3rot, reciparr

##-----------------------------

def fraser_bins(fraser_img, dist_pix, dqv=0) :
    """Returns horizontal and vertical HBins objects for pixels in units of k=1
    Fraser imaging array, returned by method fraser(...).
    Units: sample to detector distance dist_pix given in pixels, 
    dqv - normalized offset for qv (for l=1 etc.)
    """
    from pyimgalgos.HBins import HBins

    rows,cols = fraser_img.shape
    # this is how Fraser's image pixel defined relative to the scale factor dist_pix
    qhmax = 0.5*cols/dist_pix
    qvmax = 0.5*rows/dist_pix
    qh_bins = HBins((-qhmax, qhmax), cols, vtype=np.float32)
    qv_bins = HBins((-qvmax+dqv, qvmax+dqv), rows, vtype=np.float32)
    return qh_bins, qv_bins

#------------------------------

def rotation_cs(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

def rotation(X, Y, angle_deg) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot rotated by angle_deg
    """
    angle_rad = math.radians(angle_deg)
    S, C = math.sin(angle_rad), math.cos(angle_rad)
    return rotation_cs(X, Y, C, S)

#------------------------------

def rotation_phi_beta(x, y, z, phi_deg, beta_deg, scale) :
    """Returns horizontal and vertical components of the scattering vector in units of scale (k)
       x, y can be arrays, z-scalar in the same units, ex. scale = k[1/A] or in number of pixels etc.
    """
    xrot, yrot = rotation(np.array(x), np.array(y), phi_deg)
    return fraser_xyz(xrot, yrot, z, beta_deg, scale)

##-----------------------------

def calc_phi(x1pix, y1pix, x2pix, y2pix, dist) :
    """Evaluates fiber phi angle [rad] for two peaks in equatorial region
       - x1pix - [in] x coordinate of the 1st point
       - y1pix - [in] y coordinate of the 1st point 
       - x1pix - [in] x coordinate of the 2nd point 
       - y1pix - [in] y coordinate of the 2nd point 
       - dist  - [in] distance from sample to detector
    """	
    x1 = x1pix / dist
    y1 = y1pix / dist
    x2 = x2pix / dist
    y2 = y2pix / dist	
    d1 = math.sqrt(x1*x1 + y1*y1 + 1.) - 1.
    d2 = math.sqrt(x2*x2 + y2*y2 + 1.) - 1.
    return math.atan((y2*d1 - y1*d2) / (x1*d2 - x2*d1)) 

##-----------------------------

def calc_beta(xpix, ypix, phi, dist) :
    """Evaluates fiber beta angle [rad] for two peaks in equatorial region
       - xpix - [in] x coordinate of the point
       - ypix - [in] y coordinate of the point 
       - phi  - [in] fiber tilt angle [rad] if the detector plane
       - dist - [in] distance from sample to detector
    """	
    x1 = xpix / dist
    y1 = ypix / dist
    d = math.sqrt(1. + x1*x1 + y1*y1) - 1.
    return math.atan((y1*math.cos(phi) + x1*math.sin(phi)) / d)

##-----------------------------

def funcy_l0(x, phi_deg, bet_deg) :
    """Function for parameterization of y(x, phi, beta)
       of peaks in mediane plane for fiber diffraction
       ATTENTION!: curve_fit assumes that x and returned y are numpy arrays.
    """
    INFI, ZERO = 1e20, 1e-20
    phi, bet = math.radians(phi_deg), math.radians(bet_deg)
    s, c = math.sin(phi), math.cos(phi)
    sb, cb = math.sin(bet), math.cos(bet)
    if not sb :
        return -x*s/c if c else INFI 

    t = sb/cb if cb else INFI
    s, c = (s/t, c/t) if t else (s*INFI, c*INFI)

    denum = c*c - 1 if math.fabs(c) != 1 else ZERO

    B = c*(x*s+1)/denum
    C = (x*x*(s*s-1) + 2*x*s)/denum
    sqarg = B*B-C

    if isinstance(sqarg, np.float64) :
        if sqarg < 0 : print 'WARNING in funcy_l0: solution does not exist because of sqarg<0 :', sqarg
    else :
        for v in sqarg :
            if v < 0 : print 'WARNING in funcy_l0: solution does not exist because of sqarg<0 :', sqarg

    sqapro = np.select([sqarg > 0,], [sqarg,], default=0)
    return -B + np.sign(B)*np.sqrt(sqapro)

##-----------------------------

# DEPRICATED methods alias for funcy_l0(x, phi, beta)
funcy_v0 = funcy_l0
funcy = funcy_l0

##-----------------------------

def funcy_l1_v0(x, phi_deg, bet_deg, DoR=0.474, sgnrt=-1.) :
    """DEPRICATED: D/L - is not a constant as it should be
       Function for parameterization of y(x, phi, beta). v0: EQUATION FOR D/L=...
       of peaks in l=1 plane for fiber diffraction
       DoR = d/R ratio, where d is a distance between l=0 and l=1 on image,
       DoR = 433/913.27 - default constant
       R is sample to detector distance
       ATTENTION!: curve_fit assumes that x and returned y are numpy arrays.
    """
    INFI, ZERO = 1e20, 1e-20
    phi, bet = math.radians(phi_deg), math.radians(bet_deg)
    sb, cb = math.sin(bet), math.cos(bet)
    s, c = math.sin(phi), math.cos(phi)
    if not sb :
        return (DoR - x*s)/c if c else INFI

    t = sb/cb if cb else INFI
    s, c = s/t, c/t
    G = 1-DoR/sb if sb else INFI
    denum = c*c - 1 if math.fabs(c) != 1 else ZERO

    # parameters of of y^2 + 2By + C = 0 
    B = c*(x*s+G)/denum
    C = (x*x*(s*s-1) + 2*x*s*G + G*G - 1)/denum
    sqarg = B*B-C

    if isinstance(sqarg, np.float64) :
        if sqarg < 0 : print 'WARNING in funcy_l1_v0: solution does not exist because of sqarg<0 :', sqarg
    else :
        for v in sqarg :
            if v < 0 : print 'WARNING in funcy_l1_v0: solution does not exist because of sqarg<0 :', sqarg

    sqapro = np.select([sqarg > 0,], [sqarg,], default=0)

    #sign = 1 if bet<-13 else -1
    #return -B + sign * np.sqrt(sqapro)
    return -B + sgnrt * np.sqrt(sqapro)

##-----------------------------

def funcy_l1_v1(x, phi_deg, bet_deg, DoR=0.4292, sgnrt=1.) :
    """v0: EQUATION FOR D/R. Function for parameterization of y(x, phi, beta)
       of peaks in l=1 plane for fiber diffraction
       DoR = d/R ratio, where d is a distance between l=0 and l=1 on image,
       DoR = 392/913.27 - default
       R is sample to detector distance
       ATTENTION!: curve_fit assumes that x and returned y are numpy arrays.
     """
    INFI, ZERO = 1e20, 1e-20
    phi, bet = math.radians(phi_deg), math.radians(bet_deg)
    sb, cb = math.sin(bet), math.cos(bet)
    t = sb/cb if cb else INFI
    s, c = math.sin(phi), math.cos(phi)
    g = 0
    if sb :
        G = 1+DoR/sb if sb else INFI
        g = 1/G if G else INFI
        s, c = (g*s/t, g*c/t) if t else (g*s*INFI, g*c*INFI)
    else :
        s, c = s*cb/DoR, c*cb/DoR
    denum = c*c - 1 if math.fabs(c) != 1 else ZERO

    # parameters of of y^2 + 2By + C = 0 
    B = c*(x*s+g)/denum
    C = (x*x*(s*s-1) + 2*g*x*s + g*g - 1)/denum
    sqarg = B*B-C

    #print 's:%8.3f  c:%8.3f  sb:%8.3f  cb:%8.3f  t:%8.3f  g:%8.3f  denum:%8.3f' % (s, c, sb, cb, t, g, denum),\
    #      '  B:', B, '  C:', C, '  sqarg:', sqarg

    if isinstance(sqarg, np.float64) :
        if sqarg < 0 : print 'WARNING in funcy_l1_v1: solution does not exist because of sqarg<0 :', sqarg
    else :
        for v in sqarg :
            if v < 0 : print 'WARNING in funcy_l1_v1: solution does not exist because of sqarg<0 :', sqarg

    sqapro = np.select([sqarg > 0,], [sqarg,], default=0)
    #sgn = 1. if bet>-13.3 else -1.
    #return -B + sign*np.sqrt(sqapro)
    return -B + sgnrt * np.sqrt(sqapro)
    #return -B + sgn * np.sqrt(sqapro)
    #return -B - np.sign(B)*np.sqrt(sqapro)
##-----------------------------

def funcy2(x, a, b, c) :
    """Quadratic polynomial function to test curve_fit.
    """    
    return a*x*x + b*x + c

##-----------------------------

def rqh_to_xz(re, qh) :
    """Returns reciprocal (qxe,qze) coordinates of the qh projection on Ewald sphere of radius re.

    Parameters

      - re - (float scalar) Ewald sphere radius (1/A)
      - qh - (numpy array) horizontal component of q values (1/A)

    Assumption: 
       reciprocal frame origin (0,0) is on Ewald sphere,  
       center of the Ewald sphere is in (-re,0), 
       qh is a length of the Ewald sphere chorde from origin to the point with peak.
       NOTE: qh, sina, cosa, qxe, qze - can be numpy arrays shaped as qh.
       Returns: qxe, qze - coordinates of the point on Ewald sphere equivalent to q(re,qh);
       Ewald sphere frame has an experiment/detector-like definition of axes;
       - qze - along the beam,
       - qxe - transverse to the beam in l=0 horizontal plane.
    """
    sina = qh/(2*re)
    sqa = 1.-sina*sina
    sqapro = np.select([sqa > 0,], [sqa,], default=0)
    cosa = np.sqrt(sqapro)
    #qxe, qze =  qh*cosa, -qh*sina
    return  qh*cosa, -qh*sina

##-----------------------------

def rqhqv_to_xyz(re, qh, qv) :
    """Returns reciprocal (xe,ye,ze) coordinates of the q(re,qh,qv) projections on Ewald sphere frame.
       Reciprocal frame has origin on Ewald sphere and experiment/detector-like definition of axes;
       ze - along the beam,
       xe, ye - transverse to the beam in horizontal and vertical plane, respectively.
       Need in this method for l=1 or other 3-d cases.
    """
    qh2 = qh*qh
    qv2 = qv*qv
    qze = -(qh2+qv2)/(2*re)
    sqa = qh2 - qze*qze
    sqapro = np.select([sqa > 0,], [sqa,], default=0)
    qxe = np.sqrt(sqapro)*np.sign(qh)
    qye = qv
    return  qxe, qye, qze

##-----------------------------

def qh_to_xy(qh, R) :
    """Alias to DEPRICATED method with swaped parameters.
    """
    qxe, qze = rqh_to_xz(R, qh)
    return qze, qxe

##-----------------------------

def recipnorm(x, y, z) :
    """Returns normalizd reciprocal space coordinates (qx,qy,qz)
       of the scattering vector q = (k^prime - k)/abs(k),
       and assuming that
       - scattering point is a 3-d space origin, also center of the Ewalds sphere
       - k points from 3-d space origin to the point with coordinates x, y, z
       (pixel coordinates relative to IP)
       - scattering is elastic, no energy loss or gained, abs(k^prime)=abs(k)
       - reciprocal space origin is in the intersection point of axes z and Ewald's sphere. 
    """
    L = np.sqrt(z*z + x*x + y*y) 
    return x/L, y/L, z/L-1.

##-----------------------------
##---------- TESTS ------------
##-----------------------------

def test_plot_phi() :
    print """Test plot for phi angle"""

    import pyimgalgos.GlobalGraphics as gg

    xarr = np.linspace(-2,2,50)
    tet = -12
    y0 = [funcy(x,   0, tet) for x in xarr]
    y1 = [funcy(x,  -5, tet) for x in xarr]
    y2 = [funcy(x,  -6, tet) for x in xarr]
    y3 = [funcy(x,  -7, tet) for x in xarr]
    y4 = [funcy(x, -10, tet) for x in xarr]
    y5 = [funcy(x,  10, tet) for x in xarr]
    
    fig1, ax1 = gg.plotGraph(xarr, y0, figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80))
    ax1.plot(xarr, y1,'r-')
    ax1.plot(xarr, y2,'y-')
    ax1.plot(xarr, y3,'k-')
    ax1.plot(xarr, y4,'m-')
    ax1.plot(xarr, y5,'g.')
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('y', fontsize=14)
    ax1.set_title('tet=-12, phi=10,0,-5,-6,-7,-10', color='k', fontsize=20)

    #gg.savefig('variation-phi.png')
    gg.show()

##-----------------------------

def test_plot_beta() :
    print """Test plot for beta angle"""

    import pyimgalgos.GlobalGraphics as gg

    xarr = np.linspace(-2,2,50)
    phi = 0
    y0 = [funcy(x, phi,   0) for x in xarr]
    y1 = [funcy(x, phi,  -2) for x in xarr]
    y2 = [funcy(x, phi,  -5) for x in xarr]
    y3 = [funcy(x, phi,  -6) for x in xarr]
    y4 = [funcy(x, phi,  -7) for x in xarr]
    y5 = [funcy(x, phi, -10) for x in xarr]
    y6 = [funcy(x, phi,   2) for x in xarr]
    y7 = [funcy(x, phi,   5) for x in xarr]
    y8 = [funcy(x, phi,  10) for x in xarr]
    
    fig2, ax2 = gg.plotGraph(xarr, y0, figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80)) 
    ax2.plot(xarr, y1,'r-')
    ax2.plot(xarr, y2,'y-')
    ax2.plot(xarr, y3,'b-')
    ax2.plot(xarr, y4,'m-')
    ax2.plot(xarr, y5,'g-')
    ax2.plot(xarr, y6,'r.')
    ax2.plot(xarr, y7,'g.')
    ax2.plot(xarr, y8,'b.')
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('y', fontsize=14)
    ax2.set_title('phi=0, theta=10, 5, 2, 0,-2,-5,-6,-7,-10', color='k', fontsize=20)

    #gg.savefig('variation-theta.png')
    gg.show()

##-----------------------------

def test_plot_beta_l0() :
    print """Test plot for beta angle"""

    import pyimgalgos.GlobalGraphics as gg

    xarr = np.linspace(-2,2,50)
    phi = 0
    cmt = 'l0'


    
    y_000 = [funcy_l0(x, phi,   0) for x in xarr]
    y_p10 = [funcy_l0(x, phi,  10) for x in xarr]
    y_p20 = [funcy_l0(x, phi,  20) for x in xarr]
    y_p30 = [funcy_l0(x, phi,  30) for x in xarr]
    y_p40 = [funcy_l0(x, phi,  40) for x in xarr]
    y_p50 = [funcy_l0(x, phi,  50) for x in xarr] # 48
    y_m10 = [funcy_l0(x, phi, -10) for x in xarr]
    y_m20 = [funcy_l0(x, phi, -20) for x in xarr]
    y_m30 = [funcy_l0(x, phi, -30) for x in xarr]    
    y_m40 = [funcy_l0(x, phi, -40) for x in xarr]    
    y_m50 = [funcy_l0(x, phi, -50) for x in xarr] # -48

    #fig2, ax2 = gg.plotGraph(xarr, y_m01, pfmt='k.', figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80)) 
    fig2, ax2 = gg.plotGraph(xarr, y_000, pfmt='k-', figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80), lw=2) 

    #b: blue
    #g: green
    #r: red
    #c: cyan
    #m: magenta
    #y: yellow
    #k: black
    #w: white


    ax2.plot(xarr, y_p50,'g-x', label=' 50')
    ax2.plot(xarr, y_p40,'m-',  label=' 40')
    ax2.plot(xarr, y_p30,'b-',  label=' 30')
    ax2.plot(xarr, y_p20,'y-',  label=' 20')
    ax2.plot(xarr, y_p10,'r-',  label=' 10')
    ax2.plot(xarr, y_000,'k-',  label='  0')
    ax2.plot(xarr, y_m10,'r.',  label='-10')
    ax2.plot(xarr, y_m20,'y.',  label='-20')
    ax2.plot(xarr, y_m30,'b.',  label='-30')
    ax2.plot(xarr, y_m40,'m.',  label='-40')
    ax2.plot(xarr, y_m50,'g+',  label='-50')
                                        
    ax2.legend(loc='upper right')

    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('y', fontsize=14)
    ax2.set_title('%s: phi=%.1f, beta=[-50,50]' % (cmt,phi), color='k', fontsize=20)

    gg.savefig('test-plot-beta-l0.png')
    gg.show()

##-----------------------------
    #b: blue
    #g: green
    #r: red
    #c: cyan
    #m: magenta
    #y: yellow
    #k: black
    #w: white
##-----------------------------

def test_plot_beta_l1(DoR=0.4292, sgnrt=1.) :
    print """Test plot for beta angle"""

    import pyimgalgos.GlobalGraphics as gg

    xarr = np.linspace(-2,2,50)
    phi = 0

    fancy_plt = funcy_l1_v1
    #fancy_plt = funcy_l1_v0

    cmt = 'POS' if sgnrt > 0 else 'NEG' #'-B -/+ sqrt(B*B-C)'
    cmt = '%s-DoR-%.3f' % (cmt, DoR)
    
    y_p10 = [fancy_plt(x, phi,  10,   DoR, sgnrt) for x in xarr]
    y_000 = [fancy_plt(x, phi,   0,   DoR, sgnrt) for x in xarr]
    y_m10 = [fancy_plt(x, phi, -10,   DoR, sgnrt) for x in xarr]
    y_m13 = [fancy_plt(x, phi, -13,   DoR, sgnrt) for x in xarr]    
    y_m15 = [fancy_plt(x, phi, -15,   DoR, sgnrt) for x in xarr]    
    y_m20 = [fancy_plt(x, phi, -20,   DoR, sgnrt) for x in xarr]
    y_m30 = [fancy_plt(x, phi, -30,   DoR, sgnrt) for x in xarr]
    y_m35 = [fancy_plt(x, phi, -35,   DoR, sgnrt) for x in xarr]    
    y_m40 = [fancy_plt(x, phi, -40,   DoR, sgnrt) for x in xarr]
    
    fig2, ax2 = gg.plotGraph(xarr, y_000, pfmt='k-', figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80), lw=2) 
    ax2.plot(xarr, y_p10,'g-',  label=' 10')
    ax2.plot(xarr, y_000,'k-',  label='  0')
    ax2.plot(xarr, y_m10,'g.',  label='-10')
    ax2.plot(xarr, y_m13,'r-.', label='-13')
    ax2.plot(xarr, y_m15,'y.',  label='-14')
    ax2.plot(xarr, y_m20,'r.',  label='-20')
    ax2.plot(xarr, y_m30,'c.',  label='-30')
    ax2.plot(xarr, y_m35,'m.',  label='-35')
    ax2.plot(xarr, y_m40,'b+',  label='-40')

    ax2.legend(loc='upper right')
    
    ax2.set_title('%s: phi=%.1f, beta=[-40,10]' % (cmt,phi), color='k', fontsize=20)

    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('y', fontsize=14)

    gg.savefig('test-plot-beta-l1-%s.png' % cmt)
    gg.show()

##-----------------------------
##-----------------------------
    #b: blue
    #g: green
    #r: red
    #c: cyan
    #m: magenta
    #y: yellow
    #k: black
    #w: white
##-----------------------------

def test_plot_beta_l1_zoom(DoR=0.4292, sgnrt=1.) :
    print """Test plot for beta angle"""

    import pyimgalgos.GlobalGraphics as gg

    phi = 0

    fancy_plt = funcy_l1_v1
    #fancy_plt = funcy_l1_v0

    if sgnrt > 0 : 

        cmt = 'POS' #'-B -/+ sqrt(B*B-C)'
        cmt = '%s-DoR-%.3f' % (cmt, DoR)
        
        xarr = np.linspace(-0.29,0.29,60)

        y_000 = [fancy_plt(x, phi,   0,   DoR, sgnrt) for x in xarr]
        y_m05 = [fancy_plt(x, phi,  -5,   DoR, sgnrt) for x in xarr]
        y_m09 = [fancy_plt(x, phi,  -9,   DoR, sgnrt) for x in xarr]
        y_m13 = [fancy_plt(x, phi, -13.3, DoR, sgnrt) for x in xarr]    
        y_m18 = [fancy_plt(x, phi, -18,   DoR, sgnrt) for x in xarr]
        y_m20 = [fancy_plt(x, phi, -20,   DoR, sgnrt) for x in xarr]
        
        fig2, ax2 = gg.plotGraph(xarr, y_000, pfmt='k-', figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80), lw=2) 
        ax2.plot(xarr, y_000,'k-',  label='  0')
        ax2.plot(xarr, y_m05,'g.',  label=' -5')
        ax2.plot(xarr, y_m09,'y.',  label=' -9')
        ax2.plot(xarr, y_m13,'r-.', label='-13')
        ax2.plot(xarr, y_m18,'c.',  label='-18')
        ax2.plot(xarr, y_m20,'b.',  label='-20')
        
        ax2.set_title('%s: phi=%.1f, beta=[-20,0]' % (cmt,phi), color='k', fontsize=20)
        ax2.legend(loc='upper center')

    if sgnrt < 0 : 

        cmt = 'NEG' #'-B -/+ sqrt(B*B-C)'
        cmt = '%s-DoR-%.3f' % (cmt, DoR)
        
        xarr = np.linspace(-1,1,50)

        y_m20 = [fancy_plt(x, phi, -20,   DoR, sgnrt) for x in xarr]
        y_m23 = [fancy_plt(x, phi, -23,   DoR, sgnrt) for x in xarr]
        y_m25 = [fancy_plt(x, phi, -25,   DoR, sgnrt) for x in xarr]
        y_m27 = [fancy_plt(x, phi, -27,   DoR, sgnrt) for x in xarr]
        y_m30 = [fancy_plt(x, phi, -30,   DoR, sgnrt) for x in xarr]
        y_m35 = [fancy_plt(x, phi, -35,   DoR, sgnrt) for x in xarr]
        y_m40 = [fancy_plt(x, phi, -40,   DoR, sgnrt) for x in xarr]
        y_m60 = [fancy_plt(x, phi, -60,   DoR, sgnrt) for x in xarr]
        
        fig2, ax2 = gg.plotGraph(xarr, y_m25, pfmt='k-', figsize=(10,5), window=(0.15, 0.10, 0.78, 0.80), lw=2) 
        ax2.plot(xarr, y_m20,'g+-', label='-20')
        ax2.plot(xarr, y_m23,'m-',  label='-23')
        ax2.plot(xarr, y_m25,'k-',  label='-25')
        ax2.plot(xarr, y_m27,'b.',  label='-27')
        ax2.plot(xarr, y_m30,'y.',  label='-30')
        ax2.plot(xarr, y_m35,'r.',  label='-35')
        ax2.plot(xarr, y_m40,'c.',  label='-40')
        ax2.plot(xarr, y_m60,'+',   label='-60')
        
        ax2.set_title('%s: phi=%.1f, beta=[-60,-20]' % (cmt,phi), color='k', fontsize=20)
        ax2.legend(loc='lower right')

    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('y', fontsize=14)

    gg.savefig('test-plot-beta-l1-%s-zoomed.png' % cmt)
    gg.show()

##-----------------------------

def test_fraser() :
    print """Test fraser transformation"""

    from pyimgalgos.GlobalGraphics import fig_axes, plot_img, store
    import matplotlib.pyplot as plt

    fname = '/reg/neh/home1/dubrovin/LCLS/rel-mengning/plots/cspad-cxif5315-0169-000079.npy'

    img2d = np.load(fname)

    s12, s3, recimg = fraser(img2d, 10, 1000, center=None)

    store.fig, store.axim, store.axcb = fig_axes() # if not do_plot else (None, None, None)
    plot_img(recimg, mode='do not hold', amin=0, amax=100)
    #plot_img(img2d, mode='do not hold', amin=0, amax=200)
    plt.ioff() # hold control on show() after the last image
    plt.show()

#------------------------------

if __name__ == "__main__" :

    import sys

    if len(sys.argv) < 2 :
        print 'For specific test use command:\n> %s <test-id-string>' % sys.argv[0]
        print 'Default test: test_fraser()'
        test_fraser()
        sys.exit('Default test is completed')

    DoR = 390/913.27

    print 'Test: %s' % sys.argv[1]
    if   sys.argv[1] == '1' : test_fraser()
    elif sys.argv[1] == '2' : test_plot_phi()
    elif sys.argv[1] == '3' : test_plot_beta()
    elif sys.argv[1] == '4' : test_plot_beta_l1(DoR=DoR, sgnrt=+1.)
    elif sys.argv[1] == '5' : test_plot_beta_l1(DoR=DoR, sgnrt=-1.)
    elif sys.argv[1] == '6' : test_plot_beta_l0()
    elif sys.argv[1] == '7' : test_plot_beta_l1_zoom(DoR=DoR, sgnrt=+1.)
    elif sys.argv[1] == '8' : test_plot_beta_l1_zoom(DoR=DoR, sgnrt=-1.)
    else :
        print 'Default test: test_fraser()'
        test_fraser()
    sys.exit('Test %s is completed' % sys.argv[1])
    
##-----------------------------
##-----------------------------
##-----------------------------

