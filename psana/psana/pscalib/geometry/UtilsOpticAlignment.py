#!/usr/bin/env python

""" A set of generic methods for optical metrology
    Created on 2018-12-18 by Mikhail Dubrovin
    2025-08-06 adopted to lcls2

    Usage:
    from psana.pscalib.geometry.UtilsOpticAlignment import OpticalMetrologyEpix10ka2M
"""

import os
import sys
import math
from math import atan2, degrees, sqrt, fabs, pi, radians, cos, sin
import numpy as np


import logging
logger = logging.getLogger(__name__)

TXY = 60  # tolerance in x-y for warning
TOZ = 100 # tolerance in z for warning

FMT = '%12s %2i %12s %2i  %8d %8d %8d  %6d %6d %6d  %9.5f %9.5f %9.5f'


def rotation_cs(X, Y, C, S):
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S
    Yrot = Y*C + X*S
    return Xrot, Yrot

def rotation(X, Y, angle_deg):
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot rotated by angle_deg
    """
    angle_rad = radians(angle_deg)
    S, C = sin(angle_rad), cos(angle_rad)
    return rotation_cs(X, Y, C, S)

def rotate_vector_xy(v, angle_deg):
    """Returns 3-component (np.array) vector rotated by angle in degree.
    """
    xrot, yrot = rotation(v[0], v[1], angle_deg)
    return np.array((xrot, yrot, v[2]))


def create_directory(dname, mode=0o2775):
    if not dname or dname is None: return
    if os.path.exists(dname):
        pass
    else:
        os.makedirs(dname, mode)


def save_textfile(text, path, accmode=0o664):
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append
    """
    f=open(path,'w')
    f.write(text)
    f.close()
    os.chmod(path, accmode)


def read_optical_metrology_file(fname='metrology.txt', apply_offset_and_tilt_correction=True):
    """Reads the metrology.txt file with original optical measurements
       and returns array of records [(n, x, y, z, quad),]
    """
    #logger.debug('In %s' % sys._getframe().f_code.co_name)
    logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    arr_opt = []

    if not os.path.lexists(fname):
        raise IOError('File "%s" is not available' % fname)

    logger.info('open file %s' % fname)
    file = open(fname, 'r')

    quad = 0

    for linef in file:

        line = linef.strip('\n').strip()

        if not line:
            logger.debug('EMPTY LINE IS IGNORED')
            continue   # discard empty strings

        if line[0] == '#': # comment
            logger.debug('COMMENT IS IGNORED: "%s"' % line)
            continue

        list_of_fields = line.split()
        field0 = list_of_fields[0]

        if field0.lower() == 'quad': # Treat quad header lines
            quad = int(list_of_fields[1])
            logger.debug('IS RECOGNIZED QUAD: %d' % quad)
            continue

        if field0.lower() in ('point', 'sensor'): # Treat the title lines
            logger.debug('TITLE LINE:     %s' % line)
            continue

        if not field0.lstrip("-+").isdigit(): # is 1-st field digital?
            logger.debug('RECORD IS IGNORED due to unexpected format of the line: %s' % line)
            continue

        if len(list_of_fields) != 4: # Ignore lines with non-expected number of fields
            logger.warning('len(list_of_fields) =', len(list_of_fields))
            logger.warning('RECORD IS IGNORED due to unexpected format of the line: %s' % line)
            continue

        n, x, y, z = [int(v) for v in list_of_fields]

        logger.debug('ACCEPT RECORD: %3d %7d %7d %7d ' % (n, x, y, z))

        #arr_opt.append((quad, n, x, y, z))
        arr_opt.append((n, x, y, z, quad))

    file.close()

    arr = np.array(arr_opt, dtype= np.int32)

    if apply_offset_and_tilt_correction:
        correct_detector_center_offset(arr)
        correct_detector_tilt(arr)

    return arr # _opt


def correct_detector_center_offset(arr):
    """Subtract x,y,z mean offset from point coordinates
    """
    logger.debug('correct_detector_center_offset')
    logger.debug('raw array of points:\n%s' % str(arr))
    coords = arr[:,1:4]
    logger.debug('point coords:\n%s' % str(coords))

    det_center = np.mean(coords, axis=0).astype(np.int32)
    logger.debug('det_center: %s' % str(det_center))

    arr[:,1:4] -= det_center


def correct_detector_tilt(arr):
    """Correct for common detector tilt around x,y axes which may happen in optical metrology
    """
    logger.debug('in correct_detector_tilt')
    n = arr[:,0]
    x = arr[:,1].astype(np.float64)
    y = arr[:,2].astype(np.float64)
    z = arr[:,3].astype(np.float64)
    q = arr[:,4]
    logger.debug('  point n: %s' % str(n))
    logger.debug('  point x: %s' % str(x))
    logger.debug('  point y: %s' % str(y))
    logger.debug('  point z: %s' % str(z))
    logger.debug('  point q: %s' % str(q))

    wx_tilt = z * np.sign(x) # weight*angle = |x| * z/x
    wx      = np.absolute(x)
    wy_tilt = z * np.sign(y) # weight*angle = |y| * z/y
    wy      = np.absolute(y)

    tilt_x = np.sum(wx_tilt)/np.sum(wx)
    tilt_y = np.sum(wy_tilt)/np.sum(wy)

    logger.debug('  mean tilts[rad] around x: %.6f y: %.6f' % (tilt_x, tilt_y))
    logger.debug('  mean tilts[deg] around x: %.6f y: %.6f' % (degrees(tilt_x), degrees(tilt_y)))

    arr[:,3] -= (tilt_x*x + tilt_y*y).astype(np.int32) # point z-coordinate correction


def correct_twisters(arr_of_twisters, usez):
    """Apply x,y,z mean offset to all panel twisters
       push rotation angle in range [0,360]
       correct z and tilts for usez parameter
    """
    seg_centers = arr_of_twisters[:,:3]
    logger.debug('seg_centers:\n%s' % str(seg_centers))

    det_center = np.mean(seg_centers, axis=0)
    logger.debug('det_center:\n%s' % str(det_center))

    arr_of_twisters[:,:3] -= det_center
    arr_of_twisters[:,3] %= 360 # for example 540 -> 180
    #logger.debug('arr_of_twisters after center offset:\n%s' % str(arr_of_twisters))
    if not usez:
        arr_of_twisters[:,2] = 0 # Z
        arr_of_twisters[:,7] = 0 # tilt Y
        arr_of_twisters[:,8] = 0 # tilt X


def make_table_of_segments(arr,qoff=0):
    """Reshape optical metrology table to arr_segs;
       arr_segs.shape=(nsegs, 4(points-per-segment), 5(n, x, y, z, q))
       NOTE: npoints may not nesserely be dividable by 4...

       Input: arr (n, x, y, z, q)
       Output: arr_segs
    """
    logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    npoints = arr.shape[0]
    nsegs = npoints//4 # int(npoints/4)

    logger.debug('number of optical metrology points: %d number of segments: %d' % (npoints, nsegs))
    arr_segs = np.empty(shape=(nsegs, 4, 5), dtype=np.int64)

    npoints = nsegs*4
    for i in range(npoints):
        nseg = i//4 # [0, npoints/4]
        npoi = i%4 # [0,3]
        #print 'XXX nseg: %d  npoi: %d' % (nseg, npoi)

        #apply qoff to sdegment index
        seginq = nseg%4
        q = (nseg//4 + qoff)%4
        s = q*4 + seginq
        print('XXXXXX seg: %2d quad:%d quad+off:%d seg_off:%2d' % (nseg, nseg//4, q, s))

        arr_segs[s, npoi,:] = arr[i,:]
        #arr_segs[nseg, npoi,:] = arr[i,:]

    return arr_segs


def is_correct_numeration(mylst):
    for i, v in enumerate(mylst):
        if i==0:
            if (v-1)%4 != 0: return False
            continue
        if v != (mylst[i-1]+1): return False
    return True


def check_points_numeration(arr_segs):
    logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    msg = ''

    if is_correct_numeration(arr_segs[:,:,0].flatten()): msg += '\nOK - points in table are sequential'
    else: msg += '\nWARNING - numeration of points in table is NOT sequential or started from non-x4 number'

    nsegs = arr_segs.shape[0]
    for nseg in range(nsegs):
        pnums = arr_segs[nseg,:,0]
        msg += '\nMeasured segment %2d  point numbers: (%3d %3d %3d %3d)'%\
               (nseg, pnums[0], pnums[1], pnums[2], pnums[3])
        if is_correct_numeration(pnums): msg += ' OK - points in segment are sequential'
        else: msg += '\nWARNING - numeration of points in segment is NOT sequential or started from non-x4 number'

    logger.debug(msg)


def segment_center_coordinates(arr1seg):
    """Returns segment center coordinates x, y, z in micrometer
       Input : arr1seg - numpy array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
       Output: x_um, y_um, z_um - segment center coordinates
    """
    #n, x_um, y_um, z_um = 0.25 * arr1seg.sum(axis=0)
    return 0.25 * arr1seg.sum(axis=0)[1:4]


def evaluate_short_long_average(S1, S2, L1, L2, dS1, dS2, dL1, dL2, dZS1, dZS2, dZL1, dZL2):
    dZSA = 0.5 * (dZS1 + dZS2)
    dZLA = 0.5 * (dZL1 + dZL2)
    SA   = 0.5 * (S1   + S2)
    LA   = 0.5 * (L1   + L2)
    dSA  = 0.5 * (dS1  + dS2)
    dLA  = 0.5 * (dL1  + dL2)
    return LA, SA, dSA, dLA, dZLA, dZSA


def get_segment_vectors(arr1seg, iorgn=0):
    """Returns segment vectors relative to its (x,y) origin point.
       (x,y) origin is a one of [0,3] corner, not necessarily pixel(0,0).
       For quality check origin corner does not matter.
       For real geometry it is important to get correct tilt angles.

       1) makes dictionary of 3 vectors from segment origin corner to 3 other corners,
       2) orders dictionary by vector length and assign them to vS1, vL1, and vD1,
       3) find complementary vectors vS2, vL2, vD2 and return results.

       Input : arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
       Output: vS1, vS2, vL1, vL2, vD1, vD2
    """
    dic_v = {}
    for i in range(4):
        if i == iorgn: continue
        v = arr1seg[i,1:4] - arr1seg[iorgn,1:4]
        vlen = sqrt(np.sum(np.square(v)))
        #print 'v.shape: %s, v: %s, vlen:%f ' % (v.shape, v, vlen)
        dic_v[vlen] = v

    list_v_keys = sorted(dic_v.keys())
    #print '   sorted(list_v_keys) = ', list_v_keys
    vS1, vL1, vD1 = [dic_v[k] for k in list_v_keys]
    vS2 = vD1 - vL1
    vL2 = vD1 - vS1
    vD2 = vL1 - vS1
    #print 'XXX: vS1, vS2, vL1, vL2, vD1, vD2 = ', vS1, vS2, vL1, vL2, vD1, vD2

    return vS1, vS2, vL1, vL2, vD1, vD2


def cyclic_index(i, csize=4):
    """Returns cyclic index in the range [0,3].
    """
    return i % csize


def segment_side_vectors_in_metrology_frame(arr1seg, iorgn):
    """Returns segment side vectors in optical metrology frame
       naming vectors vx, vy, vd relative to the segment origin point,
       assuming that (x,y) origin is in one of the [0,3] corners numerated clockwise in RHS.

       Input :
             - arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
             - iorgn - (int) segment origin point index [0,3]
       Output:
             - vx1, vx2, vy1, vy2, vd1, vd2
    """
    ix = cyclic_index(iorgn-1)
    iy = cyclic_index(iorgn+1)
    im = cyclic_index(iorgn+2)

    vx1, vy1, vd1 = [(arr1seg[i,1:4] - arr1seg[iorgn,1:4]) for i in (ix, iy, im)]

    vx2 = vd1 - vy1
    vy2 = vd1 - vx1
    vd2 = vx1 - vy1

    return vx1, vx2, vy1, vy2, vd1, vd2


def n90_orientation(x, y, gate_deg=45):
    """Returns (int) n90 dominant orientation of the (x,y) point in the range [0,3]
    """
    andle_deg = degrees(atan2(y, x))
    n90 = 0 if fabs(andle_deg)     < gate_deg else\
          1 if fabs(andle_deg-90)  < gate_deg else\
          2 if fabs(andle_deg-180) < gate_deg else\
          2 if fabs(andle_deg+180) < gate_deg else\
          3
    return n90


def evaluate_length_width_angle(arr1seg, iorgn):
    """
       Input:
             - arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
             - iorgn - (int) segment origin point index [0,3]
       Output:
             - x_um, y_um, z_um - segment center coordinates
    """
    #if self.vrb & DEBUG: print 'In %s' % (sys._getframe().f_code.co_name)

    vS1, vS2, vL1, vL2, vD1, vD2 = get_segment_vectors(arr1seg, iorgn)

    ix, iy = 0, 1
    segm_n90 = n90_orientation(vL1[ix], vL1[iy])

    # horizontal or vertical segment
    horiz = segm_n90 in (0,2)
    vert  = segm_n90 in (1,3)
    ix, iy, iz = (0, 1, 2) if vert else (1, 0, 2)

    S1   = vS1[ix]
    S2   = vS2[ix]

    dS1  = vL1[ix]
    dS2  = vL2[ix]

    L1   = vL1[iy]
    L2   = vL2[iy]

    dL1  = vS1[iy]
    dL2  = vS2[iy]

    dZS1 = vS1[iz]
    dZS2 = vS2[iz]
    dZL1 = vL1[iz]
    dZL2 = vL2[iz]

    LA, SA, dSA, dLA, dZLA, dZSA =\
        evaluate_short_long_average(S1, S2, L1, L2, dS1, dS2, dL1, dL2, dZS1, dZS2, dZL1, dZL2)

    XSize = fabs(LA if horiz else SA)
    YSize = fabs(SA if horiz else LA)
    dZX   = dZLA    if horiz else dZSA
    dZY   = dZSA    if horiz else dZLA

    D1   = sqrt(np.sum(np.square(vD1)))
    D2   = sqrt(np.sum(np.square(vD2)))
    dD   = D1 - D2

    ddS  = dS1 - dS2
    ddL  = dL1 - dL2

    ddZS = dZS1 - dZS2
    ddZL = dZL1 - dZL2

    rotXYDegree = segm_n90 * 90
    rotXZDegree = 0
    rotYZDegree = 0

    vLA = 0.5*(vL1+vL2)
    vSA = 0.5*(vS1+vS2)

    ix, iy = 0, 1
    tiltXY = atan2(vLA[iy], vLA[ix])
    tiltXZ = atan2(dZX, XSize)
    tiltYZ = atan2(dZY, YSize)

    #vLlen = sqrt(np.sum(np.square(vLA)))
    #vSlen = sqrt(np.sum(np.square(vSA)))
    #tiltXZ = atan2(vLA[iz], vLlen)
    #tiltYZ = atan2(vSA[iz], vSlen)

    if abs(tiltXY)>0.1 and tiltXY<0: tiltXY += 2*pi # move angle in range [0,2pi]

    tiltXYDegree = degrees(tiltXY) - rotXYDegree
    tiltXZDegree = degrees(tiltXZ) - rotXZDegree
    tiltYZDegree = degrees(tiltYZ) - rotYZDegree

    #print 'rotXY : %f' % self.rotXYDegree
    #print 'tiltXY: %f' % self.tiltXYDegree

    return S1, S2, dS1, dS2, ddS, L1, L2, dL1, dL2, ddL, D1, D2, dD, tiltXYDegree, tiltXZDegree, tiltYZDegree,\
           dZS1, dZS2, dZL1, dZL2, ddZS, ddZL, LA, SA, dSA, dLA, dZLA, dZSA, XSize, YSize, dZX, dZY


def txt_qc_table_xy(arr_segs, arr_iorgn):
    """Returns (str)  text of the quality check table
       Input : arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
       Output: text of the quality check table
    """
    logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    sepline = '%s\n' % (124*'-')
    txt = sepline
    txt += 'segm:        S1      S2     dS1     dS2        L1      L2     dL1     dL2    angle(deg)   D1      D2      dD   d(dS)   d(dL)\n'
    txt += sepline
    wrg = '\n'

    nsegs = arr_segs.shape[0]
    for nseg in range(nsegs):
        iorgn = arr_iorgn[nseg]
        arr1seg = arr_segs[nseg,:,:4]
        S1, S2, dS1, dS2, ddS, L1, L2, dL1, dL2, ddL, D1, D2, dD, tiltXYDegree, tiltXZDegree, tiltYZDegree,\
        dZS1, dZS2, dZL1, dZL2, ddZS, ddZL, LA, SA, dSA, dLA, dZLA, dZSA, XSize, YSize, dZX, dZY =\
            evaluate_length_width_angle(arr1seg, iorgn)

        txt += 'segm:%2d  %6d  %6d  %6d  %6d    %6d  %6d  %6d  %6d   %8.5f  %6d  %6d  %6d  %6d  %6d\n' % \
            (nseg, S1, S2, dS1, dS2, \
                   L1, L2, dL1, dL2, \
                   tiltXYDegree, \
                   D1, D2, dD, ddS, ddL)
        if fabs(dD)  > TXY: wrg += '  WARNING segm %2d:  |%d| > %.1f\n' % (nseg, dD,  TXY)
        if fabs(ddS) > TXY: wrg += '  WARNING segm %2d:  |%d| > %.1f\n' % (nseg, ddS, TXY)
        if fabs(ddL) > TXY: wrg += '  WARNING segm %2d:  |%d| > %.1f\n' % (nseg, ddL, TXY)
    txt += sepline + wrg
    return txt


def evaluate_deviation_from_flatness(arr1seg, iorgn):
        """Evaluates deviation from segment flatness in micron self.arr_dev_um.
           Input : arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: self.arr_dev_um - segment corner 3 deviation from flatness
        """
        #if self.vrb & DEBUG: print 'In %s' % (sys._getframe().f_code.co_name)

        #vS1, vS2, vL1, vL2, vD1, vD2 = get_segment_vectors(arr1seg, iorgn)
        #vx, vy, vd = vL1, vS1, vD1
        vx1, vx2, vy1, vy2, vd1, vd2 = segment_side_vectors_in_metrology_frame(arr1seg, iorgn)
        vx, vy, vd = vx1, vy1, vd1
        #print vx, vy, vd,

        vort = np.array(np.cross(vx, vy), dtype=np.double) # vort = [vx x vy]        - vector product
        norm = sqrt(np.sum(vort*vort))                     # norm = |vort|             - length of the vector vort
        vort_norm = vort / norm                            # vort_norm = vort / |vort| - normalized vector orthogonal to the plane
        dev = np.sum(vd*vort_norm)                         # dev = (vd * vort_norm)   - scalar product

        #self.arr_dev_um = dev

        #print '  vort_norm=', vort_norm, '  norm =', norm, '  dev =', dev
        #print '  vort_norm=', vort_norm, '  norm =', norm, '  dev =', dev
        #if self.vrb & DEBUG: print 'quad:%1d, segm:%2d,  dz3[um]: %8.3f' % (quad, segm, self.arr_dev_um[quad,segm])

        return dev


def txt_qc_table_z(arr_segs, arr_iorgn):
    """Returns (str) text of the quality check table
       Input : arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
       Output: text of the quality check table
    """
    logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    sepline = '%s\n' % (137*'-')
    txt = sepline
    txt += 'segm:        SA      LA   XSize   YSize    dZS1  dZS2  dZL1  dZL2    dZSA  dZLA  ddZS  ddZL     dZX   dZY   angXZ(deg) angYZ(deg) dz3(um)\n'
    txt += sepline
    wrg = '\n'

    nsegs = arr_segs.shape[0]
    for nseg in range(nsegs):
        iorgn = arr_iorgn[nseg]
        arr1seg = arr_segs[nseg,:,:4]
        S1, S2, dS1, dS2, ddS, L1, L2, dL1, dL2, ddL, D1, D2, dD, tiltXYDegree, tiltXZDegree, tiltYZDegree,\
        dZS1, dZS2, dZL1, dZL2, ddZS, ddZL, LA, SA, dSA, dLA, dZLA, dZSA, XSize, YSize, dZX, dZY =\
            evaluate_length_width_angle(arr1seg, iorgn)

        arr_dev_um = evaluate_deviation_from_flatness(arr1seg, iorgn)

        txt += 'segm:%2d  %6d  %6d  %6d  %6d   %5d %5d %5d %5d   %5d %5d %5d %5d   %5d %5d  %8.5f   %8.5f   %8.3f\n' % \
            (nseg, SA,   LA,   XSize, YSize, \
                   dZS1, dZS2, dZL1,  dZL2, \
                   dZSA, dZLA, ddZS,  ddZL, \
                   dZX,  dZY,  tiltXZDegree, tiltYZDegree, \
                   arr_dev_um)
        if fabs(arr_dev_um) > TOZ: wrg += '  WARNING segm %2d:  |%.1f| > %.1f\n' % (nseg, arr_dev_um, TOZ)
    txt += sepline + wrg
    return txt #+'\n'


def print_quality_check_tables(arr_segs, arr_iorgn):

    logger.info('X-Y quality check for optical metrology measurements \n%s'%\
                txt_qc_table_xy(arr_segs, arr_iorgn))

    logger.info('Z quality check for optical metrology measurements \n%s'%\
                txt_qc_table_z(arr_segs, arr_iorgn))


def segment_metrology_constants(arr1seg, iorgn):
    """Returns tuple of segment raw metrology constants
       as they are defined in metrology file, before transforming to qudrants, detector, etc.
    """
    #print 'segment metrology data:\n%s' % str(arr1seg[:,1:4])

    # evaluate segment center (x,y,z) in the optical metrology frame
    mean_xyz = tuple(np.mean(arr1seg[:,1:4], axis=0))

    vx1, vx2, vy1, vy2, vd1, vd2 = segment_side_vectors_in_metrology_frame(arr1seg, iorgn)
    #print 'vx1, vy1, vd1 = ', vx1, vy1, vd1

    segm_n90 = n90_orientation(vx1[0], vx1[1])
    #print 'iorgn:%2d  segment n90%2d' % (iorgn, segm_n90)

    # expected for flat geometry
    rot_xy_deg = segm_n90 * 90
    rot_xz_deg = 0
    rot_yz_deg = 0

    # evaluate mean segment side vectors
    vx = 0.5*(vx1+vx2)
    vy = 0.5*(vy1+vy2)

    # rotate by n90 in xy mean side vectors from optical metrology frame to segment local frame
    vx_in_loc = rotate_vector_xy(vx, -rot_xy_deg)
    vy_in_loc = rotate_vector_xy(vy, -rot_xy_deg)

    tilt_xy_deg = degrees(atan2(vx_in_loc[1], vx_in_loc[0])) # y vs x
    tilt_xz_deg = degrees(atan2(vx_in_loc[2], vx_in_loc[0])) # z vs x
    tilt_yz_deg = degrees(atan2(vy_in_loc[2], vy_in_loc[1])) # z vs y

    return mean_xyz + (rot_xy_deg, rot_xz_deg, rot_yz_deg, tilt_xy_deg, tilt_xz_deg, tilt_yz_deg)


def geometry_constants_v0(arr_segs, arr_iorgn, nsegs_in_quad, quad_orientation_deg, segnums_in_daq,\
                          def_constants, center_offset_optmet, center_ip_twister, usez):
    """Returns list/tuple of per-segment geometry constants
    """
    #logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    name_quad = def_constants[0][0]
    name_seg  = def_constants[0][2]

    nsegs = arr_segs.shape[0]
    nquads = nsegs // nsegs_in_quad
    nquads_tot = len(quad_orientation_deg)

    arr_quads     = np.empty(shape=(nquads, nsegs_in_quad, 9), dtype=np.float64)
    arr_quads_daq = np.empty(shape=(nquads, nsegs_in_quad, 9), dtype=np.float64)
    arr_quad_center = np.empty(shape=(nquads, 2), dtype=np.float64)
    #list_of_pars =[]

    for s in range(nsegs):
        iorgn = arr_iorgn[s]
        arr1seg = arr_segs[s,:,:] # 4]
        q = arr1seg[0,4]
        nsegq = s%nsegs_in_quad
        twister = segment_metrology_constants(arr1seg, iorgn)
        arr_quads[q, nsegq,:] = twister

        #seg_optical_frame_pars = (name_quad, q, name_seg, nsegq) + twister
        #list_of_pars.append(seg_optical_frame_pars)

    # subtract mean Z offset
    logger.debug('arr_quads.shape' + str(arr_quads.shape))
    z_mean = np.mean(arr_quads[:,:,2])
    logger.info('subtract z mean: %.3f' % z_mean)
    arr_quads[:,:,2] -= z_mean

    #for pars in list_of_pars:
    #    print FMT % pars

    # Transform segment optical metrology coordinates to the quad frame

    for q in range(nquads):

        #logger.debug('======== quad %2d optical metrology coordinates of segments:\n%s'%\
        #             (q, str(arr_quads[q,:,:2])))

        # find quad center
        quad_center = np.mean(arr_quads[q,:,:2], axis=0)
        arr_quad_center[q,:] = quad_center
        #print 'quad_center', quad_center

        #apply offset of panel coordinates from optical metrology frame to quad center
        for s in range(nsegs_in_quad):
            arr_quads[q,s,:2] -= quad_center
            if not usez:
                arr_quads[q,s,2] = 0 #Z
                arr_quads[q,s,7] = 0 # tilt Y
                arr_quads[q,s,8] = 0 # tilt X

        #print 'offset quad center:\n', arr_quads[q,:,:2]

        quad_deg = quad_orientation_deg[q]

        #print 'apply quad orientation angle', quad_deg

        for s in range(nsegs_in_quad):
            v = arr_quads[q,s,:3]
            v_rot = rotate_vector_xy(v, quad_deg)
            arr_quads[q,s,:3] = v_rot[:3] # z is not changing
            arr_quads[q,s,3] += quad_deg  # account for quad rotartion
            arr_quads[q,s,3] %= 360
            #xrot, yrot = rotation(x, y, quad_deg)

            seg_daq = segnums_in_daq[q][s]
            arr_quads_daq[q,seg_daq,:] = arr_quads[q,s,:]

        #print 'rotated quad:\n', arr_quads[q,:,:4]
        #print 'rotated quad daq:\n', arr_quads_daq[q,:,:4]

    #print '\noptical metrology constants'
    #for q in range(nquads):
    #    for s in range(nsegs_in_quad):
    #        print FMT % ((name_quad, q, name_seg, s) + tuple(arr_quads_daq[q,s,:]))

    camera_center = np.mean(arr_quad_center, axis=0)
    logger.info('\nnumber of quads in metrology %d of total expected in camera %d' % (nquads, nquads_tot))
    logger.debug('\nquad centers from metrology:\n%s\n'  % str(arr_quad_center)+\
                 '\naveraged over quads center:\n%s'     % str(camera_center)+\
                 '\n\ncamera center from input pars:\n%s'% str(center_offset_optmet))

    #print 'camera_center evaluated as an averaged quad center:\n', camera_center

    _center_offset_optmet = center_offset_optmet if nquads != nquads_tot else camera_center

    #print '\nmake list of constants'
    list_geo_recs = []
    for rec in def_constants:
        name_par, ip, name_obj, io = rec[:4]
        rec_new = (name_quad, ip, name_seg, io) + tuple(arr_quads_daq[ip,io,:])\
                  if ip<nquads and io<nsegs and name_par==name_quad and name_obj==name_seg else\
                  rec[:4] + tuple(arr_quad_center[io] - np.array(_center_offset_optmet)) + rec[6:]\
                  if io<nquads and name_obj==name_quad else\
                  rec

        list_geo_recs.append(list(rec_new))

    #list_geo_recs[-1][7] = <z-angler>
    list_geo_recs[-1][4:10] = center_ip_twister

    logger.info('constants form list_geo_recs:\n%s' % str_geo_constants(list_geo_recs))
    return list_geo_recs


def geometry_constants_v1(arr_segs, arr_iorgn, def_constants, segnums_in_daq, center_ip_twister, usez):
                       #, nsegs_in_quad, quad_orientation_deg, center_offset):
    """Returns list/tuple of per-segment geometry constants
    """
    #logger.debug('%s\nIn %s' % (60*'-', sys._getframe().f_code.co_name))

    name_subd = def_constants[-2][0]
    name_seg  = def_constants[0][2]
    ind_subd  = 0

    logger.debug('arr_segs:\n%s' % str(arr_segs))

    nsegs = arr_segs.shape[0]

    logger.debug('arr_segs.shape: %s' % str(arr_segs.shape))
    logger.debug('nsegs: %d' % nsegs)

    list_of_twisters =[]
    for s in range(nsegs):

        iorgn = arr_iorgn[s]
        arr1seg = arr_segs[s,:,:]
        q = arr1seg[0,4]
        logger.debug('seg %2d quad %2d origin point %d:\n%s' % (s, q, iorgn, arr1seg))

        #nsegq = s%nsegs_in_quad
        twister = segment_metrology_constants(arr1seg, iorgn)
        logger.debug('twister: %s' % str(twister))
        list_of_twisters.append(twister)

    arr_of_twisters = np.array(list_of_twisters)
    #logger.debug('arr_of_twisters:\n%s' % str(arr_of_twisters))

    correct_twisters(arr_of_twisters, usez)

    # combine list of segment rercords
    dict_geo_recs = {}
    for s in range(nsegs):
        seg_daq = segnums_in_daq[s]
        twister = tuple(arr_of_twisters[s])
        seg_optical_frame_pars = (name_subd, ind_subd, name_seg, seg_daq) + twister
        dict_geo_recs[seg_daq] = seg_optical_frame_pars

    list_geo_recs = [dict_geo_recs[seg_daq] for seg_daq in range(nsegs)]
    # add last record
    list_geo_recs.append(last_record_v1(name_subd, ind_subd, center_ip_twister))

    #list_of_pars_str = [FMT % pars for pars in list_of_pars]
    logger.debug('constants form list_geo_recs:\n%s' % str_geo_constants(list_geo_recs))

    return list_geo_recs


def last_record_v1(name_subd, ind_subd, ip_twister):
    return ('IP', 0, name_subd, ind_subd) + ip_twister + (0, 0, 0)
    #return ('IP', 0, name_subd, ind_subd, 0, 0,  1000000, 0, 0, 0, 0, 0, 0)


def str_geo_constants(lst):
    """Returns str from tuple/liat of geometry constants.
    """
    return '\n'.join([FMT % tuple(rec) for rec in lst])


def str_comment(comments):
    return '\n# '+'\n# '.join(['COMMENT:%02d %s'%(i,s) for i,s in enumerate(comments)])


def str_geo_constants_hat():
    #from CalibManager.GlobalUtils import get_login, get_current_local_time_stamp
    from time import localtime, strftime #, gmtime, clock, time, sleep
    import getpass
    return \
        '\n# DATE_TIME %s' % strftime('%Y-%m-%d %H:%M:%S %Z', localtime())+\
        '\n# USER %s' % getpass.getuser()+\
        '\n# CALIB_TYPE geometry'\
        '\n# PARAM:01 PARENT     - name and version of the parent object'\
        '\n# PARAM:02 PARENT_IND - index of the parent object'\
        '\n# PARAM:03 OBJECT     - name and version of the object'\
        '\n# PARAM:04 OBJECT_IND - index of the new object'\
        '\n# PARAM:05 X0         - x-coordinate [um] of the object origin in the parent frame'\
        '\n# PARAM:06 Y0         - y-coordinate [um] of the object origin in the parent frame'\
        '\n# PARAM:07 Z0         - z-coordinate [um] of the object origin in the parent frame'\
        '\n# PARAM:08 ROT_Z      - object design rotation angle [deg] around Z axis of the parent frame'\
        '\n# PARAM:09 ROT_Y      - object design rotation angle [deg] around Y axis of the parent frame'\
        '\n# PARAM:10 ROT_X      - object design rotation angle [deg] around X axis of the parent frame'\
        '\n# PARAM:11 TILT_Z     - object tilt angle [deg] around Z axis of the parent frame'\
        '\n# PARAM:12 TILT_Y     - object tilt angle [deg] around Y axis of the parent frame'\
        '\n# PARAM:13 TILT_X     - object tilt angle [deg] around X axis of the parent frame'\
        '\n'\
        '\n# HDR PARENT IND     OBJECT IND    X0[um]   Y0[um]   Z0[um]   ROT-Z  ROT-Y  ROT-X     TILT-Z    TILT-Y    TILT-X'\
        '\n'


def default_constants_epix10ka2m_v0():
    # HDR PARENT IND     OBJECT IND    X0[um]   Y0[um]   Z0[um]   ROT-Z ROT-Y ROT-X     TILT-Z   TILT-Y   TILT-X
    SENSOR   = 'EPIX10KA:V1'
    QUAD     = 'QUAD'
    CAMERA   = 'CAMERA'
    IP       = 'IP'
    return (
        (QUAD  , 0,    SENSOR, 0,    -20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 0,    SENSOR, 1,     20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 0,    SENSOR, 2,    -20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 0,    SENSOR, 3,     20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\

        (QUAD  , 1,    SENSOR, 0,    -20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 1,    SENSOR, 1,     20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 1,    SENSOR, 2,    -20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 1,    SENSOR, 3,     20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\

        (QUAD  , 2,    SENSOR, 0,    -20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 2,    SENSOR, 1,     20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 2,    SENSOR, 2,    -20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 2,    SENSOR, 3,     20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\

        (QUAD  , 3,    SENSOR, 0,    -20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 3,    SENSOR, 1,     20150,    20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 3,    SENSOR, 2,    -20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\
        (QUAD  , 3,    SENSOR, 3,     20150,   -20950,        0,     180,     0,     0,    0,  0,  0),\

        (CAMERA, 0,    QUAD  , 0,    -38450,    42850,        0,      90,     0,     0,    0,  0,  0),\
        (CAMERA, 0,    QUAD  , 1,     42850,    38450,        0,       0,     0,     0,    0,  0,  0),\
        (CAMERA, 0,    QUAD  , 2,     38450,   -42850,        0,     270,     0,     0,    0,  0,  0),\
        (CAMERA, 0,    QUAD  , 3,    -42850,   -38450,        0,     180,     0,     0,    0,  0,  0),\

        (IP    , 0,    CAMERA, 0,         0,        0,   100000,      90,     0,     0,    0,  0,  0)\
    )


def default_constants_epix10ka2m_v1():
    # HDR PARENT IND  OBJECT IND    X0[um]   Y0[um]   Z0[um]     ROT-Z   ROT-Y   ROT-X TILT-Z TILT-YTILT-X
    SENSOR   = 'EPIX10KA:V1'
    CAMERA   = 'CAMERA'
    IP       = 'IP'
    return (
      (CAMERA,  0,  SENSOR,  0,    -59032,    23573,        0,     270,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  1,    -59026,    64317,        0,     270,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  2,    -17405,    23567,        0,     270,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  3,    -17039,    63692,        0,     270,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  4,     23423,    59147,        0,     180,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  5,     64071,    59039,        0,     180,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  6,     23779,    17334,        0,     180,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  7,     63997,    17181,        0,     180,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  8,     59152,   -23403,        0,      90,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR,  9,     58942,   -64143,        0,      90,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR, 10,     17468,   -23658,        0,      90,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR, 11,     17029,   -63871,        0,      90,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR, 12,    -23591,   -59310,        0,       0,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR, 13,    -64155,   -59056,        0,       0,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR, 14,    -23622,   -17240,        0,       0,      0,      0,   0,   0,   0),\
      (CAMERA,  0,  SENSOR, 15,    -63991,   -17169,        0,       0,      0,      0,   0,   0,   0),\
      (    IP,  0,  CAMERA,  0,         0,        0,   100000,      90,      0,      0,   0,   0,   0)\
    )


class OpticalMetrologyEpix10ka2M():
    """Optical metrology measurements processing for Epix10ka2M"""

    def __init__(self, parser):
        self._name = self.__class__.__name__
        self.parser = parser
        self.init_parameters()
        logger.warning('VERSION: %d' % self.vers)
        if self.vers == 0: self.proc_optical_metrology_data_v0()
        else: self.proc_optical_metrology_data_v1()

    def init_parameters(self):
        (popts, pargs) = self.parser.parse_args()
        self.ifname = pargs[0] if len(pargs) else popts.ifn # popts['ifn']
        #self.ifname = popts.ifn # popts['ifn']
        self.ofname = popts.ofn # popts['ofn']
        self.xc     = popts.xc
        self.yc     = popts.yc
        self.xcip   = popts.xcip
        self.ycip   = popts.ycip
        self.zcip   = popts.zcip
        self.azip   = popts.azip
        self.rot    = popts.rot
        self.qoff   = popts.qoff
        self.loglev = popts.log
        self.vers   = popts.vers
        self.usez   = popts.usez
        self.docorr = popts.docorr

        msg = 'Command: %s' % ' '.join(sys.argv)
        logger.info(msg)

    def proc_optical_metrology_data_v0(self):
        """v0 process metrology using quads
        """
        NSEGS_IN_QUAD_EPIX10KA2M = 4

        irot = self.rot%4 # rotation index in the range [0,3] showing location of q0

                                    #     Q0        Q1          Q2            Q3
        METROLOGY_SEGNUMS_IN_DAQ = (((0,1,3,2), (2,0,1,3), (3,2,0,1), (1,3,2,0)),\
                                    ((2,0,1,3), (3,2,0,1), (1,3,2,0), (0,1,3,2)),\
                                    ((3,2,0,1), (1,3,2,0), (0,1,3,2), (2,0,1,3)),\
                                    ((1,3,2,0), (0,1,3,2), (2,0,1,3), (3,2,0,1)))[irot]

        DET_ORIENTATION_DEG = (90,180,270,0)[irot]

        CENTER_IP_TWISTER = (self.xcip, self.ycip, self.zcip, self.azip, 0, 0)

        QUAD_ORIENTATION_DEG =\
           ((  0, 90,180,270),\
            ( 90,180,270,  0),\
            (180,270,  0, 90),\
            (270,  0, 90,180))[(self.rot+self.qoff)%4] #[self.rot%4]

        # metrology point index in range [0,3] for "origin" - 0 pixel in DAQ
        SEG_XY_ORIGIN_EPIX10KA2M =\
           ((1,1,1,1, 2,2,2,2, 3,3,3,3, 0,0,0,0),\
            (2,2,2,2, 3,3,3,3, 0,0,0,0, 1,1,1,1),\
            (3,3,3,3, 0,0,0,0, 1,1,1,1, 2,2,2,2),\
            (0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3))[irot]

        DEF_CONSTANTS = default_constants_epix10ka2m_v0()
        CENTER_OFFSET_OPTMET = (self.xc, self.yc)

        arr_points = read_optical_metrology_file(fname=self.ifname, apply_offset_and_tilt_correction=self.docorr)
        logger.debug('Array of points:\n%s' % str(arr_points))

        arr_segs = make_table_of_segments(arr_points) #, self.qoff)
        logger.debug('Array of segments:\n%s' % str(arr_segs))

        check_points_numeration(arr_segs)

        print_quality_check_tables(arr_segs, SEG_XY_ORIGIN_EPIX10KA2M)

        logger.info('default constants:\n%s' % str_geo_constants(DEF_CONSTANTS))

        lst = geometry_constants_v0(arr_segs, SEG_XY_ORIGIN_EPIX10KA2M, NSEGS_IN_QUAD_EPIX10KA2M,\
                                 QUAD_ORIENTATION_DEG, METROLOGY_SEGNUMS_IN_DAQ, DEF_CONSTANTS,\
                                 CENTER_OFFSET_OPTMET, CENTER_IP_TWISTER, self.usez)
        cons = str_geo_constants(lst)
        cmts = str_comment(('detector:Epix10ka2M experiment:abcd01234',\
                            'constants generated from optical metrology',\
                            'processor version %d - panels-quads-detector' % self.vers))
        hat  = str_geo_constants_hat()

        geo_cons = '%s%s\n%s' % (cmts, hat, cons)

        logger.info('geometry constants:\n%s' % geo_cons)

        dname = os.path.dirname(self.ofname)
        create_directory(dname, mode=0o2775)
        save_textfile(geo_cons, self.ofname, accmode=0o664)
        logger.info('geometry constants saved in file %s' % self.ofname)


    def proc_optical_metrology_data_v1(self):
        """v0 process metrology withoiut quads

        DAQ panel numbers at different orientations of the detector in optical metrology

        rot = 0  ^y
        Q0  1  3 |  4  5  Q1
            0  2 |  6  7
        ---------+----------> x
           15 14 | 10  8
        Q3 13 12 | 11  9  Q2

        rot = 1  ^y
        Q3 13 15 |  0  1  Q0
           12 14 |  2  3
        ---------+----------> x
           11 10 |  6  4
        Q2  9  8 |  7  5  Q1

        rot = 2  ^y
        Q2  9 11 | 12 13  Q3
            8 10 | 14 15
        ---------+----------> x
            7  6 |  2  0
        Q1  5  4 |  3  1  Q0

        rot = 3  ^y
        Q1  5  7 |  8  9  Q2
            4  6 | 10 11
        ---------+----------> x
            3  2 | 14 12
        Q0  1  0 | 15 13  Q3
        """
        irot = self.rot%4 # rotation index in the range [0,3] showing location of q0

        NSEGS_IN_QUAD_EPIX10KA2M = 4
                                    #     Q0        Q1          Q2            Q3
        METROLOGY_SEGNUMS_IN_DAQ = ((0,1,3,2,  6,4,5,7,  11,10,8,9,  13,15,14,12),\
                                    (2,0,1,3,  7,6,4,5,  9,11,10,8,  12,13,15,14),\
                                    (3,2,0,1,  5,7,6,4,  8,9,11,10,  14,12,13,15),\
                                    (1,3,2,0,  4,5,7,6,  10,8,9,11,  15,14,12,13))[irot]

        # metrology point index in range [0,3] for "origin" - 0 pixel in DAQ
        SEG_XY_ORIGIN_EPIX10KA2M =\
           ((1,1,1,1, 2,2,2,2, 3,3,3,3, 0,0,0,0),\
            (2,2,2,2, 3,3,3,3, 0,0,0,0, 1,1,1,1),\
            (3,3,3,3, 0,0,0,0, 1,1,1,1, 2,2,2,2),\
            (0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3))[irot]

        DEF_CONSTANTS = default_constants_epix10ka2m_v1()
        #CENTER_OFFSET_OPTMET = (self.xc, self.yc)
        CENTER_IP_TWISTER = (self.xcip, self.ycip, self.zcip, self.azip, 0, 0)

        arr_points = read_optical_metrology_file(fname=self.ifname, apply_offset_and_tilt_correction=self.docorr)
        logger.debug('Array of points:\n%s' % str(arr_points))

        arr_segs = make_table_of_segments(arr_points, self.qoff)
        logger.debug('Array of segments:\n%s' % str(arr_segs))

        check_points_numeration(arr_segs)

        print_quality_check_tables(arr_segs, SEG_XY_ORIGIN_EPIX10KA2M)

        logger.info('default constants:\n%s' % str_geo_constants(DEF_CONSTANTS))

        lst = geometry_constants_v1(arr_segs, SEG_XY_ORIGIN_EPIX10KA2M, DEF_CONSTANTS,\
                      METROLOGY_SEGNUMS_IN_DAQ, CENTER_IP_TWISTER, self.usez)

        #====================
        #sys.exit('TEST EXIT')
        #====================

        cons = str_geo_constants(lst)
        cmts = str_comment(('detector:Epix10ka2M experiment:abcd01234',\
                            'constants generated from optical metrology',\
                            'processor version %d - panels only, no quads' % self.vers))
        hat  = str_geo_constants_hat()

        geo_cons = '%s%s\n%s' % (cmts, hat, cons)

        logger.info('geometry constants:\n%s' % geo_cons)

        dname = os.path.dirname(self.ofname)
        create_directory(dname, mode=0o2775)
        save_textfile(geo_cons, self.ofname, accmode=0o664)
        logger.info('geometry constants saved in file %s' % self.ofname)


if __name__ == "__main__":
    sys.exit('Try command> optical_metrology_epix10ka2m')

# EOF

