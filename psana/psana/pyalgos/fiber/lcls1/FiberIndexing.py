#!@PYTHON@
####!/usr/bin/env python
#------------------------------
"""
:py:class:`FiberIndexing` collection of methods for fiber indexing
==================================================================

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

Created on Oct 7, 2015 by Mikhail Dubrovin
"""
#------------------------------

import sys
import math
import numpy as np

#------------------------------
#------------------------------
#------------------------------
#------------------------------

class BinPars() :
    """ Bin parameters storage
    """
    def __init__(self, edges, nbins, vtype=np.float32, endpoint=False):
        self.vmin       = vmin = min(edges)
        self.vmax       = vmax = max(edges)
        self.nbins      = nbins
        self.binwidth   = (vmax-vmin)/nbins
        self.halfbinw   = 0.5 * self.binwidth
        self.vtype      = vtype
        self.endpoint   = endpoint
        self.binedges   = np.linspace(vmin, vmax, nbins, endpoint=endpoint, dtype=vtype)
        self.bincenters = self.binedges + self.halfbinw
        self.inds       = range(self.binedges.size)
        self.indedges   = zip(self.inds, self.binedges)          
        self.indcenters = zip(self.inds, self.bincenters)          
        self.strrange   = '%.0f-%.0f-%d' % (vmin, vmax, nbins)

#------------------------------
#------------------------------

def wavelength_nm_from_energy_ev(E_eV=6000) :
    """Returns wavelength in nm, evaluated from photon energy in eV (1Angstroms = 10**-10m)
       E=h*v = h*c/lambda
       6keV approx. = 2A
    """
    #h/2pi = 6.58211928e-16     # eV*s (PDG 2014) reduced Planck constant
    #h = 2 * math.pi * h/2pi    # Planck constant
    #c = 299792458              # m/s - speed of light (exact)
    #hc = h*c                   # eV*m
    #wavelen = hc/E_eV * 1e9    # Angstrom, where 1m = 10^9nm = 10^10A
    return 1239.8493/E_eV       # nm 

#------------------------------

def wave_vector_value(E_eV=6000) :
    """Returns wave vector/number value
       k = 1/lambda    [1/A] - crystalographer's definition
       k = 2*pi/lambda [1/A] - physics definition
    """
    return 1. / (wavelength_nm_from_energy_ev(E_eV) * 10) # 1nm = 10A
    #return 2 * math.pi / (wavelength_nm_from_energy_ev(E_eV) * 10) # 1nm = 10A

#------------------------------

def round_vzeros(v,d=10) :
    """Returns input vector with rounded to zero components
       which precision less than requested number of digits.
    """
    prec = pow(10,-d)
    vx = v[0] if math.fabs(v[0]) > prec else 0.0
    vy = v[1] if math.fabs(v[1]) > prec else 0.0
    vz = v[2] if math.fabs(v[2]) > prec else 0.0
    return vx,vy,vz

#------------------------------

def triclinic_primitive_vectors(a=18.36,  b=26.65, c=4.81,\
                                alpha=90, beta=90, gamma=102.83) :
    """Returns 3-d (Bravias) primitive vectors directed along crystal axes (edges)
       from lattice cell edge lengths [Angstrom or other prefered units] and angles [degree]
       for triclinic crystal cell parametes::

                  *----------* 
                 / \        / \ 
                /   \      /   \ 
               /     \ gamma    \ 
              /       *----------* 
             /       /  /       / 
            /alpha  /  /       / 
           *-------/--*       c 
            \     /    \ beta/ 
             a   /      \   / 
              \ /        \ / 
               *-----b----* 
             
       where a, b, c - crystal cell edge lengths,
       alpha, beta, gamma - interaxial angles around a, b, c edges, respectively'
       By design, a1 vector for edge a is directed along x,
       a2 vector for edge b is in x-y plane, has (x,y,0) components only,
       a3 vector for edge c has (x,y,z) components.

       See geometry details in my logbook p.7-8.
    """
    alp, bet, gam  = math.radians(alpha), math.radians(beta), math.radians(gamma)
    ca, cb, cg = math.cos(alp), math.cos(bet), math.cos(gam)
    sa, sb, sg = math.sin(alp), math.sin(bet), math.sin(gam)

    cx, cy, cz = 0, 0, c
    if cb != 0 : 
        tanphi = (ca/cb-cg)/sg
        cx = c*cb
        cy = cx*tanphi
        cz = math.sqrt(c*c - cx*cx - cy*cy)
    elif ca != 0 : # cb==0
        cx = 0.
        cy = c*ca/sg
        cz = math.sqrt(c*c - cx*cx - cy*cy)

    a1 = (a, 0, 0)
    a2 = (b*cg, b*sg, 0)
    a3 = (cx, cy, cz)

    return round_vzeros(a1), round_vzeros(a2), round_vzeros(a3)

#------------------------------

def parameters_of_primitive_vectors(a1, a2, a3) :
    """Returns a, b, c, alpha, beta, gamma for three primitive vectors a1, a2, a3.
    """
    a = math.sqrt(np.dot(a1,a1))
    b = math.sqrt(np.dot(a2,a2))
    c = math.sqrt(np.dot(a3,a3))

    ca = np.dot(a2,a3)/b/c
    cb = np.dot(a1,a3)/a/c
    cg = np.dot(a1,a2)/a/b

    alp = math.degrees(math.acos(ca))
    bet = math.degrees(math.acos(cb))
    gam = math.degrees(math.acos(cg))

    return a, b, c, alp, bet, gam

#------------------------------

def print_primitive_vectors(a1, a2, a3, fmt='%10.6f') :
    """Prints three primitive vectors and their derived parameters.
    """
    print '\nIn %s' % sys._getframe().f_code.co_name
    print 'primitive vectors:\n  a1 = (%s)\n  a2 = (%s)\n  a3 = (%s)' %\
           (', '.join([fmt % v for v in a1]),\
            ', '.join([fmt % v for v in a2]),\
            ', '.join([fmt % v for v in a3]))
    a, b, c, alp, bet, gam = parameters_of_primitive_vectors(a1, a2, a3)
    print 'Derived parameters of primitive vectors:\n',\
           'a=%.3f  b=%.3f  c=%.3f  alp=%.2f  bet=%.2f  gam=%.2f' % (a, b, c, alp, bet, gam)

#------------------------------

def reciprocal_from_bravias(a1, a2, a3) :
    """Returns reciprocal primitive vectors from 3-d Bravias primitive vectors
       using crystallographer's definition for conversion as 1/d
       (2*pi/d - comes from natural physics definition).
    """
    s1 = np.cross(a2,a3)
    s2 = np.cross(a3,a1)
    s3 = np.cross(a1,a2)

    b1 = s1/np.dot(a1,s1)
    b2 = s2/np.dot(a2,s2)
    b3 = s3/np.dot(a3,s3)

    return b1, b2, b3

#------------------------------

def lattice(b1 = (1.,0.,0.), b2 = (0.,1.,0.), b3 = (0.,0.,1.),\
            hmax=3, kmax=2, lmax=1, cdtype=np.float32,\
            hmin=None, kmin=None, lmin=None) :
    """ returns n-d arrays of 3d coordinates or 2d(if lmax=0) for 3d lattice and Miller hkl indices
    """
    #print '\nIn %s' % sys._getframe().f_code.co_name

    lst_h = range(-hmax, hmax+1) if hmin is None else range(hmin, hmax+1)
    lst_k = range(-kmax, kmax+1) if kmin is None else range(kmin, kmax+1)
    lst_l = range(-lmax, lmax+1) if lmin is None else range(lmin, lmax+1)
    
    x1d = np.array([b1[0]*h for h in lst_h], dtype=cdtype)
    y1d = np.array([b1[1]*h for h in lst_h], dtype=cdtype)
    z1d = np.array([b1[2]*h for h in lst_h], dtype=cdtype)

    #print 'x1d: ', x1d
    #print 'y1d: ', y1d
    #print 'z1d: ', z1d

    x2d = np.array([x1d+b2[0]*k for k in lst_k], dtype=cdtype)
    y2d = np.array([y1d+b2[1]*k for k in lst_k], dtype=cdtype)
    z2d = np.array([z1d+b2[2]*k for k in lst_k], dtype=cdtype)
    r2d = np.sqrt(x2d*x2d + y2d*y2d + z2d*z2d)

    h2d, k2d = np.meshgrid(lst_h, lst_k)
    l2d = np.zeros_like(h2d)

    if lmax == 0 : return x2d, y2d, z2d, r2d, h2d, k2d, l2d
    
    onehk = np.ones_like(h2d)
    h3d = np.array([h2d         for l in lst_l], dtype=np.int32)    
    k3d = np.array([k2d         for l in lst_l], dtype=np.int32)
    l3d = np.array([onehk*l     for l in lst_l], dtype=np.int32)    

    x3d = np.array([x2d+b3[0]*l for l in lst_l], dtype=cdtype)
    y3d = np.array([y2d+b3[1]*l for l in lst_l], dtype=cdtype)
    z3d = np.array([z2d+b3[2]*l for l in lst_l], dtype=cdtype)
    r3d = np.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)

    return x3d, y3d, z3d, r3d, h3d, k3d, l3d

#------------------------------

def rotation_cs(X, Y, c, s) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
       for specified rotation angle cosine and sine values.
    """
    Xrot = X*c - Y*s 
    Yrot = Y*c + X*s 
    return Xrot, Yrot

#------------------------------

def rotation(X, Y, angle_deg) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot rotated by angle_deg
    """
    angle_rad = math.radians(angle_deg)
    s, c = math.sin(angle_rad), math.cos(angle_rad)
    return rotation_cs(X, Y, c, s)

#------------------------------

def q_components(X, Y, Z, evald_rad=0.5) :
    """For all defined nodes of the lattice returns
       dr - distance from evald sphere to the reciprocal lattice node,
       qv, qh - vertical, horizontal components of the momentum transfer vector.
       NOTE: X, Y, Z, DX, L, dr, qv, qh, ql, qy, ql are the numpy arrays with shape=(2*lmax+1, 2*kmax+1, 2*hmax+1), evald_rad is a scalar
    """
    DX = X + evald_rad
    L  = np.sqrt(DX*DX + Y*Y + Z*Z)
    dr = L - evald_rad
    qv = evald_rad * Z/L
    ql = evald_rad * (DX/L-1)
    qt = evald_rad * Y/L
    qh = np.sqrt(ql*ql + qt*qt) * np.select([Y < 0], [-1], default=1) 
    return dr, qv, qh, qt, ql

#------------------------------

def print_nda(nda, cmt, fmt=' %8.4f') :
    """Prints ndarray and its shape with preceded comment.
    """
    print '\n%s.shape: %s' % (cmt, str(nda.shape)),

    if len(nda.shape) == 1 :
        for c in nda : print fmt % c,

    elif len(nda.shape) == 2 : 
        for row in nda :
            print '\nrow: ',
            for c in row : print fmt % c,
            
    elif len(nda.shape) == 3 : 
        for layer in nda :
            print '\n(3d) layer: ',
            for row in layer :
                print '\nrow: ',
                for c in row : print fmt % c,
    print '\n'
    
#------------------------------

def print_omega_dr(omega_deg, dr, drmax=1) :
    """ Depricated, see str_omega_drhkl.
    """
    lst_dr = [d for d in dr.flatten() if math.fabs(d) < drmax]       
    if len(lst_dr) > 1:
        print 'omega=%.2f degree, lst_dr: ' % (omega_deg),
        for dr in lst_dr : print ' %.2f' % dr,
        print ''
    else : print 'omega=%.2f degree, lst_dr is empty' % (omega_deg)
    
#------------------------------

def str_omega_drhkl(ind, beta_deg, omega_deg, dr, r, qv, qh, qt, ql, h, k, l, sigma_ql) :
    """ Returns the record to save in look-up table or print.
    """
    drmax = 3 * sigma_ql
    factor = -1./(2*sigma_ql*sigma_ql)
    
    lst_drhkl = [e for e in zip(dr.flatten(), h.flatten(), k.flatten(), l.flatten(),\
                                r.flatten(), qv.flatten(), qh.flatten(), qt.flatten(), ql.flatten()) if math.fabs(e[0]) < drmax]       
    s = ''

    if len(lst_drhkl) > 1:  #because lst_drhkl always has a record (0.0, 0, 0, 0, 0.0, 0.0, 0.0)
        s = '# beta %.2f  omega %.2f degree' % (beta_deg, omega_deg)\
          + '\n# index    beta   omega   h   k   l   dr[1/A]  R(h,k,l)   qv[1/A]   qh[1/A]   qt[1/A]   ql[1/A]   P(omega)'
        for e in lst_drhkl :
            if e[1] == 0 and e[2] == 0 and e[3] == 0 : continue
            d = math.fabs(e[0])
            if sigma_ql and d > drmax : continue
            prob = math.exp(factor*d*d)
            s += '\n%6d  %7.2f %7.2f %3d %3d %3d %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f  %9.6f' %\
                  (ind, beta_deg, omega_deg, e[1], e[2], e[3], e[0], e[4], e[5], e[6], e[7], e[8], prob)
        return '%s\n\n' % s
    else : return '# beta %.2f  omega %.2f degree EMPTY\n' % (beta_deg, omega_deg)
    
#------------------------------

def fill_row(dr, qv, qh, h, k, l, sigma_ql, sigma_qt, bpq) :
    """Returns histogram array (row) for horizontal q component
       filled by probability to see the peak, modulated by the Gaussian function of dr,
       where dr is a radial distance between the lattice node and Evald's sphere.
    """
    row = np.zeros((bpq.nbins,), dtype=np.float32)

    range_ql = 3 * sigma_ql
    range_qt = 3 * sigma_qt
    factor_ql = -1./(2.*sigma_ql*sigma_ql)
    factor_qt = -1./(2.*sigma_qt*sigma_qt)
    
    # loop over lattice nodes
    for dr1, qv1, qh1, h1, k1, l1 in zip(dr.flatten(), qv.flatten(), qh.flatten(), h.flatten(), k.flatten(), l.flatten()) :

        #if dr1==0 and qv1==0 : continue # and qh1==0 
        if h1 == 0 and k1 == 0 : continue

        if math.fabs(dr1) > range_ql : continue

        f_angle = math.exp(factor_ql*dr1*dr1)

        # loop over q bins
        for indq, binq in bpq.indcenters :

            dq = qh1 - binq
            if math.fabs(dq) > range_qt : continue

            row[indq] = f_angle * math.exp(factor_qt*dq*dq)

    return row

#------------------------------

def make_lookup_table(b1 = (1.,0.,0.), b2 = (0.,1.,0.), b3 = (0.,0.,1.),\
                      hmax=3, kmax=2, lmax=1, cdtype=np.float32,\
                      evald_rad=3, sigma_q=0.001, fout=None, bpq=None, bpomega=None, bpbeta=None,\
                      hmin=None, kmin=None, lmin=None) :
    """Depricated, see  make_lookup_table_v2 with sigma_ql, sigma_qt in stead of sigma_q
    """
    return make_lookup_table_v2(b1, b2, b3, hmax, kmax, lmax, cdtype, evald_rad, sigma_q, sigma_q,\
                                fout, bpq, bpomega, bpbeta, hmin, kmin, lmin)

#------------------------------

def make_lookup_table_v2(b1 = (1.,0.,0.), b2 = (0.,1.,0.), b3 = (0.,0.,1.),\
                      hmax=3, kmax=2, lmax=1, cdtype=np.float32,\
                      evald_rad=3, sigma_ql=0.001, sigma_qt=0.001, fout=None, bpq=None, bpomega=None, bpbeta=None,\
                      hmin=None, kmin=None, lmin=None) :
    """Makes lookup table - crystal lattice nodes information as a function of angle beta and omega, where
       beta  [deg] - fiber axis tilt,  
       omega [deg] - fiber rotation around axis,  
       For each crysal orientation (beta, gamma) lookup table contains info about lattice nodes
       closest to the Evald's sphere:

       * # beta 0.00  omega 52.50 degree
       * # index    beta   omega   h   k   l   dr[1/A]  R(h,k,l)   qv[1/A]   qh[1/A]   qt[1/A]   ql[1/A]   P(omega)
       *   106     0.00   52.50  -2  -1   0 -0.002756  0.123157  0.000000 -0.123478 -0.122470 -0.015745   0.165321
       *   106     0.00   52.50   1   1   0 -0.000249  0.072533  0.000000  0.072551  0.072347 -0.005436   0.985422
       *   106     0.00   52.50   3   5   0  0.000687  0.273564  0.000000  0.273369  0.262250 -0.077171   0.894200

       where:

       - index - orientation index (just an unique integer number)
       - beta, omega [deg] - crystal orientation angles,
       - h, k, l - Miller indeces
       - dr [1/A] - distance between lattice node and Evald's sphere
       - R(h,k,l) [1/A] - distance between nodes (h,k,l) and (0,0,0)
       - qv, qh [1/A] - vertical and horizontal components of scattering vector q
       - qt, ql [1/A] - transverse (in horizontal plane) and longitudinal components of vector q
       - P(omega) - un-normalized probability (<1) evaluated for dr(omega) using sigma_ql.

       File name is generated automatically with current time stamp like
       lut-cxif5315-r0169-2015-10-23T14:58:36.txt

       Input parameters:
       b1, b2, b3 - reciprocal lattice primitive vectors,
       hmax, kmax, lmax - lattice node indeces
       cdtype - data type for lattice node coordinates,
       evald_rad - Evald's sphere radius,
       sigma_ql - expected q resolution along k (due to crystal rotation),
       sigma_qt - expected qt resolution (in detector plane),
       fout - open output file object,
       bpq, bpomega, bpbeta - binning parameters for q, omega, and beta
       NOTE: Units of b1, b2, b3, evald_rad, and sigma_q should be the same, for example [1/A].

       Returns 2-d numpy array for image; summed for all beta probobility(omega vs. q_horizontal).
    """
    x, y, z, r, h, k, l = lattice(b1, b2, b3, hmax, kmax, lmax, cdtype, hmin, kmin, lmin)

    lut = np.zeros((bpomega.nbins, bpq.nbins), dtype=np.float32)
    
    #beta_deg = 0
    #beta_deg = 15

    ind = 0
    for beta_deg in bpbeta.binedges :
        for iomega, omega_deg in bpomega.indedges :
        
            ind += 1
        
            xrot1, yrot1 = rotation(x, y, omega_deg)
            xrot2, zrot2 = rotation(xrot1, z, beta_deg)
        
            dr, qv, qh, qt, ql = q_components(xrot2, yrot1, zrot2, evald_rad)
        
            txt = str_omega_drhkl(ind, beta_deg, omega_deg, dr, r, qv, qh, qt, ql, h, k, l, sigma_ql)
            print txt,
            if fout is not None : fout.write(txt)
        
            lut[iomega,:] += fill_row(dr, qv, qh, h, k, l, sigma_ql, sigma_qt, bpq)
        
    return lut
        
#------------------------------

def lattice_node_radius(b1 = (1.,0.,0.), b2 = (0.,1.,0.), b3 = (0.,0.,1.),\
                 hmax=3, kmax=2, lmax=1, cdtype=np.float32, fmt = '%10.6f',\
                 hmin=None, kmin=None, lmin=None) :

    print '\nIn %s' % sys._getframe().f_code.co_name
    print 'reciprocal space primitive vectors:\n  b1 = (%s)\n  b2 = (%s)\n  b3 = (%s)' %\
           (', '.join([fmt % v for v in b1]),\
            ', '.join([fmt % v for v in b2]),\
            ', '.join([fmt % v for v in b3]))

    x, y, z, rarr, harr, karr, larr = lattice(b1, b2, b3, hmax, kmax, lmax, cdtype, hmin, kmin, lmin)

    hklarr = zip(harr.flatten(), karr.flatten(), larr.flatten())
    dic_r_hkl = dict(zip(rarr.flatten(),hklarr))   

    r_nodes = sorted(dic_r_hkl.keys())

    print '\n%s\nTable of lattice node parameters sorted by radius' % (80*'_')

    if lmax == 0 : print '( h, k) R(h,k)[1/A]'
    else       : print '( h, k, l) R(h,k,l)[1/A]'
    for rnode in sorted(dic_r_hkl.keys()) :
        hkl = dic_r_hkl[rnode]
        if lmax == 0 : print '(%2i,%2i) %6.4f' % (hkl[0], hkl[1], rnode)
        else       : print '(%2i,%2i,%2i) %6.4f' % (hkl[0], hkl[1], hkl[2], rnode) 

#------------------------------

def test_lattice(b1 = (1.,0.,0.), b2 = (0.,1.,0.), b3 = (0.,0.,1.),\
                 hmax=3, kmax=2, lmax=1, cdtype=np.float32,\
                 hmin=None, kmin=None, lmin=None) :

    from Detector.GlobalUtils import print_ndarr

    print '\nIn %s' % sys._getframe().f_code.co_name
    print '%s\nTest lattice with default parameters' % (80*'_')

    x, y, z, r, h, k, l = lattice(b1, b2, b3, hmax, kmax, lmax, cdtype, hmin, kmin, lmin)

    print_nda(h, 'h', fmt=' %3d')
    print_nda(k, 'k', fmt=' %3d')
    print_nda(l, 'l', fmt=' %3d')
    print_nda(x, 'x coords')
    print_nda(y, 'y coords')
    print_nda(z, 'z coords')
    print_nda(r, 'r of lattice nodes')
        
    omega_deg = 9
    beta_deg = 0
    
    xrot1, yrot1 = rotation(x, y, omega_deg)
    xrot2, zrot2 = rotation(xrot1, z, beta_deg)

    #print_nda(xrot2, 'xrot2')
    #print_nda(yrot1, 'yrot1')
    #print_nda(zrot2, 'zrot2')
    #print_nda(zrot, 'zrot')
    print_ndarr(xrot2, 'xrot2')
    print_ndarr(yrot1, 'yrot1')
    print_ndarr(zrot2, 'zrot2')

#------------------------------

def plot_lattice(b1 = (1.,0.,0.), b2 = (0.,1.,0.), b3 = (0.,0.,1.),\
                 hmax=3, kmax=2, lmax=1, cdtype=np.float32,\
                 evald_rad=0.5, qtol=0.01, prefix='', do_movie=False, delay=400,\
                 hmin=None, kmin=None, lmin=None, title_add='') :
    """Plots 2-d reciprocal space lattice, evald sphere,
       generates series of plots for rotated lattice and movie from these plots.

       - do_movie = True/False - on/off production of movie
       - delay - is a time in msec between movie frames.
    """

    import matplotlib.pyplot as plt
    import pyimgalgos.GlobalGraphics as gg
    
    print '\nIn %s' % sys._getframe().f_code.co_name
    print '%s\nTest lattice with default parameters' % (80*'_')

    x, y, z, r, h, k, l = lattice(b1, b2, b3, hmax, kmax, lmax, cdtype, hmin, kmin, lmin)

    x.shape = (x.size,)
    y.shape = (y.size,)
    z.shape = (z.size,)

    xlimits = ylimits = (-0.3, 0.3) # plot limits in (1/A)
    #ylimits = (-0.4, 0.4) # plot limits in (1/A)
    #xlimits = (-0.5, 0.3) # plot limits in (1/A)

    fig, ax = gg.plotGraph(x,y, figsize=(8,7.5), window=(0.17, 0.10, 0.78, 0.84), pfmt='bo')

    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_xlabel('Reciprocal x ($1/\AA$)', fontsize=18)
    ax.set_ylabel('Reciprocal y ($1/\AA$)', fontsize=18)
    gg.save_fig(fig, '%sreciprocal-space-lattice.png' % prefix, pbits=1)

    lst_omega = range(0,180,2) if do_movie else range(0,11,10)
    #lst_omega = range(0,180,5) if do_movie else range(0,13,11)
    #lst_omega = range(0,180,45) if do_movie else range(0,13,11)

    beta_deg = 0
    for omega_deg in lst_omega :

        xrot1, yrot1 = rotation(x, y, omega_deg)
        xrot2, zrot2 = rotation(xrot1, z, beta_deg)        
        dr, qv, qh, qt, ql = q_components(xrot2, yrot1, zrot2, evald_rad)

        xhit = [xr for dq,xr in zip(dr.flatten(), xrot2.flatten()) if math.fabs(dq) < qtol]
        yhit = [yr for dq,yr in zip(dr.flatten(), yrot1.flatten()) if math.fabs(dq) < qtol]

        #fig, ax = gg.plotGraph(xrot2, yrot1, figsize=(8,7.5), window=(0.15, 0.10, 0.78, 0.84), pfmt='bo')
        ax.cla()
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.plot(xrot1, yrot1, 'yo')
        if len(xhit) > 0 and len(yhit) > 0 : ax.plot(xhit, yhit, 'bo')

        tit = 'beta=%.0f omega=%.0f' % (beta_deg, omega_deg)
        if title_add : tit += ' %s' % (title_add)
        ax.set_title(tit, color='k', fontsize=12)
        ax.set_xlabel('Reciprocal x ($1/\AA$)', fontsize=18)
        ax.set_ylabel('Reciprocal y ($1/\AA$)', fontsize=18)
        gg.drawCenter(ax, (-evald_rad,0), s=0.04, linewidth=2, color='k')
        gg.drawCircle(ax, (-evald_rad,0), evald_rad, linewidth=1, color='k', fill=False)
        fig.canvas.draw()
        gg.show('Do not hold!')
        gg.save_fig(fig, '%slattice-rotated-beta%03d-omega%03d.png' %\
                    (prefix, int(beta_deg), int(omega_deg)), pbits=1)

    if do_movie :
        import os
        #dir_movie = 'movie'
        #os.system('mkdir %s'% dir_movie)
        cmd = 'convert -delay %f %slattice-rotated-beta*.png movie.gif' % (delay, prefix)
        print 'Wait for completion of the command: %s' % cmd
        os.system(cmd)
        print 'DONE!'
    
    gg.show()

#------------------------------
#------------------------------
#------------------------------
#------------------------------
def make_index_table(prefix='./v01-') :

    from pyimgalgos.GlobalUtils import str_tstamp
    fname = '%slut-cxif5315-r0169-%s.txt' % (prefix, str_tstamp())
    fout = open(fname,'w')
    fout.write('# file name: %s\n' % fname)

    #------------------------------
    # Photon energy
    Egamma_eV  = 6003.1936                               # eV SIOC:SYS0:ML00:AO541
    wavelen_nm = wavelength_nm_from_energy_ev(Egamma_eV) # nm
    evald_rad  = wave_vector_value(Egamma_eV)            # 1/A
    #-------
    sigma_ql   = 0.002 * evald_rad
    sigma_qt   = 0.002 * evald_rad
    #-------
    rec  = '\n# photon energy = %.4f eV' % (Egamma_eV)\
         + '\n# wavelength = %.4f A' % (wavelen_nm*10)\
         + '\n# wave number/Evald radius k = 1/lambda = %.6f 1/A' % (evald_rad)\
         + '\n# sigma_ql = %.6f 1/A (approximately = k * <pixel size>/' % (sigma_ql)\
         + '\n# sigma_qt = %.6f 1/A (approximately = k * <pixel size>/' % (sigma_qt)\
         + '<sample-to-detector distance> = k*100um/100mm)'\
         + '\n# 3*sigma_ql = %.6f 1/A\n' % (3*sigma_ql)\
         + '\n# 3*sigma_qt = %.6f 1/A\n' % (3*sigma_qt)
    print rec
    fout.write(rec)

    #------------------------------
    # Lattice parameters
    # from previous analysis note:
    #a, b, c = 18.36, 26.65, 4.81        # Angstrom
    #alpha, beta, gamma = 90, 90, 77.17  # 180 - 102.83 degree
    a = 18.55 # Angstrom
    b, c = 1.466*a, 0.262*a              # Angstrom
    alpha, beta, gamma = 90, 90, 78.47   # 180 - 101.53 degree
    hmax, kmax, lmax = 4, 6, 0           # size of lattice to consider
    #hmin, kmin, lmin =-4,-6, 0          # size of lattice to consider
    hmin, kmin, lmin = None, None, None  # default [-hmax,hmax], [-kmax,kmax],

    a1, a2, a3 = triclinic_primitive_vectors(a, b, c, alpha, beta, gamma)
    b1, b2, b3 = reciprocal_from_bravias(a1, a2, a3)

    msg1 = '\n# Triclinic crystal cell parameters:'\
         + '\n#   a = %.2f A\n#   b = %.2f A\n#   c = %.2f A' % (a, b, c)\
         + '\n#   alpha = %.2f deg\n#   beta  = %.2f deg\n#   gamma = %.2f deg' % (alpha, beta, gamma)

    msg2 = '\n# 3-d space primitive vectors:\n#   a1 = %s\n#   a2 = %s\n#   a3 = %s' %\
           (str(a1), str(a2), str(a3))

    msg3 = '\n# reciprocal space primitive vectors:\n#   b1 = %s\n#   b2 = %s\n#   b3 = %s' %\
           (str(b1), str(b2), str(b3))

    rec = '%s\n%s\n%s\n' % (msg1, msg2, msg3)
    print rec
    fout.write(rec)

    fout.write('\n# %s\n\n' % (89*'_'))

    #for line in triclinic_primitive_vectors.__doc__.split('\n') : fout.write('\n# %s' % line)

    test_lattice       (b1, b2, b3, hmax, kmax, lmax, np.float32, hmin, kmin, lmin)
    lattice_node_radius(b1, b2, b3, hmax, kmax, lmax, np.float32, '%10.6f', hmin, kmin, lmin)
    lattice_node_radius(b1, b2, b3, hmax, kmax, 1,    np.float32, '%10.6f', hmin, kmin, lmin)

    #------------------------------
    #return

    #------------------------------
    # binning for look-up table and plots

    # bin parameters for q in units of k = Evald's sphere radius [1/A]
    bpq = BinPars((-0.25, 0.25), 1000, vtype=np.float32, endpoint=False)

    # bin parameters for omega [degree] - fiber rotation angle around axis
    bpomega = BinPars((0.,  180.), 360, vtype=np.float32, endpoint=False)
    
    # bin parameters for beta [degree] - fiber axis tilt angle
    #bpbeta = BinPars((15.,  195.),  2, vtype=np.float32, endpoint=True)
    #bpbeta = BinPars((15.,   15.),  1, vtype=np.float32, endpoint=False)
    #bpbeta = BinPars((5.,    25.),  2, vtype=np.float32, endpoint=True)
    bpbeta  = BinPars((0.,    50.), 11, vtype=np.float32, endpoint=True)
    bpbeta2 = BinPars((180., 230.), 11, vtype=np.float32, endpoint=True)
    str_beta = 'for-beta:%s' % (bpbeta.strrange)
     
    print '\n%s\nIndexing lookup table\n' % (91*'_')
    lut  = make_lookup_table_v2(b1, b2, b3, hmax, kmax, lmax, np.float32, evald_rad, sigma_ql, sigma_qt, fout, bpq, bpomega, bpbeta, hmin, kmin, lmin)
    lut2 = make_lookup_table_v2(b1, b2, b3, hmax, kmax, lmax, np.float32, evald_rad, sigma_ql, sigma_qt, fout, bpq, bpomega, bpbeta2, hmin, kmin, lmin)

    fout.close()
    print '\nIndexing lookup table is saved in the file: %s' % fname

    #------------------------------
    # produce and save plots
    import pyimgalgos.GlobalGraphics as gg

    img = lut # or lut2
    img = lut + lut2

    img_range = (bpq.vmin, bpq.vmax, bpomega.vmax, bpomega.vmin) 
    axim = gg.plotImageLarge(lut, img_range=img_range, amp_range=None, figsize=(15,13),\
                      title='Non-symmetrized for beta', origin='upper', window=(0.05,  0.06, 0.94, 0.94))
    axim.set_xlabel('$q_{H}$ ($1/\AA$)', fontsize=18)
    axim.set_ylabel('$\omega$ (degree)', fontsize=18)
    gg.save('%splot-img-prob-omega-vs-qh-%s.png' % (prefix, str_beta), pbits=1)

    axim = gg.plotImageLarge(img, img_range=img_range, amp_range=None, figsize=(15,13),\
                      title='Symmetrized for beta (beta, beta+pi)', origin='upper', window=(0.05,  0.06, 0.94, 0.94))
    axim.set_xlabel('$q_{H}$ ($1/\AA$)', fontsize=18)
    axim.set_ylabel('$\omega$ (degree)', fontsize=18)
    gg.save('%splot-img-prob-omega-vs-qh-sym-%s.png' % (prefix, str_beta), pbits=1)

    arrhi = np.sum(img,0)    
    fighi, axhi, hi = gg.hist1d(bpq.binedges, bins=bpq.nbins-1, amp_range=(bpq.vmin, bpq.vmax), weights=arrhi,\
                                color='b', show_stat=True, log=False,\
                                figsize=(15,5), axwin=(0.05, 0.12, 0.85, 0.80),\
                                title=None, xlabel='$q_{H}$ ($1/\AA$)', ylabel='Intensity', titwin=None)
    gg.show()

    gg.save_fig(fighi, '%splot-his-prob-vs-qh-%s.png' % (prefix, str_beta), pbits=1)

    qh_weight = zip(bpq.bincenters, arrhi)
    fname = '%sarr-qh-weight-%s.npy' % (prefix, str_beta)
    print 'Save qh:weigt array in file %s' % fname
    np.save(fname, qh_weight)

#------------------------------

def check_triclinic_primitive_vectors(a,b,c,alp,bet,gam,vrb=True) :
    """1) prints input parameters of primitive vectors,
       2) prints three primitive vectors and their reconstructed parameters,
       3) reitrive parameters of primitive vectors,
       4) compare with input and print results of comparison.
    """
    print 50*'_'
    if vrb : print '\nIn %s' % (sys._getframe().f_code.co_name)
    print 'Input parameters of primitive vectors:\n',\
           'a=%.3f  b=%.3f  c=%.3f  alp=%.2f  bet=%.2f  gam=%.2f' % (a, b, c, alp, bet, gam)
    a1, a2, a3 = triclinic_primitive_vectors(a,b,c,alp,bet,gam)
    if vrb : print_primitive_vectors(a1, a2, a3, fmt='%10.6f')
    ra, rb, rc, ralp, rbet, rgam = parameters_of_primitive_vectors(a1, a2, a3)
    DIF = 1e-10
    fabs = math.fabs
    if  fabs(a-ra) < DIF\
    and fabs(b-rb) < DIF\
    and fabs(c-rc) < DIF\
    and fabs(alp-ralp) < DIF\
    and fabs(bet-rbet) < DIF\
    and fabs(gam-rgam) < DIF : print 'Test os OK'
    else : print 'WARNING test is failed'

#------------------------------

def test_triclinic_primitive_vectors(vrb=True) :
    check_triclinic_primitive_vectors(18, 26, 5, 90, 90,100, vrb)
    check_triclinic_primitive_vectors(18, 26, 5, 90, 90, 80, vrb)
    check_triclinic_primitive_vectors(18, 26, 5, 80, 90, 60, vrb)
    check_triclinic_primitive_vectors(18, 26, 5, 80, 70, 90, vrb)
    check_triclinic_primitive_vectors(18, 26, 5,100,100, 70, vrb)
    check_triclinic_primitive_vectors(18, 26, 5,100, 90, 70, vrb)
    check_triclinic_primitive_vectors(18, 26, 5,100,110, 90, vrb)
    check_triclinic_primitive_vectors(18, 26, 5,100,110,120, vrb)

#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname

    if   tname == '0' : make_index_table()
    elif tname == '1' : test_triclinic_primitive_vectors()
    elif tname == '2' : test_triclinic_primitive_vectors(vrb=False)
    else : sys.exit('Test %s is undefined' % tname)

#------------------------------
