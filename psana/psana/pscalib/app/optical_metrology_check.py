#!/usr/bin/env python

""" Check optical measurements for 4-point/segment (e.g. cspad 2x1)

@author Mikhail Dubrovin
adopted to lcls2 on 2025-08-05 from CalibManager/app/optical_metrology_check
"""

import os
import sys
from math import atan2, degrees, sqrt, fabs, pi
import numpy as np

from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays, GeometryAccess
import psana.pyalgos.generic.NDArrUtils as ndu # import info_ndarr, print_ndarr, divide_protected
reshape_to_3d = ndu.reshape_to_3d
info_ndarr = ndu.info_ndarr
print_ndarr = ndu.print_ndarr
divide_protected = ndu.divide_protected

INFO, WARNING, ERROR, CRITICAL, DEBUG = (1,2,4,8,16)

class OpticalMetrologyCheck(object):
    """Check optical metrology measurements"""

    def __init__(self):

        self._name = self.__class__.__name__
        self.init_parameters()

        self.read_optical_metrology_file()
        self.make_table_of_segments()
        self.check_points_numeration()
        self.print_quality_check_tables()
        if self.plt:
            self.plot_metrology_data()
        if self.his:
            self.fill_histograms()
            self.plot_histograms(self.ofpref, do_save=True, hwin_x0y0=(0,50))


    def init_parameters(self):
        self.parser = option_parser()
        (popts, pargs) = self.parser.parse_args()
        self.ifname = pargs[0] if len(pargs) else popts.ifn # popts['ifn']
        self.ofpref = popts.ofp # popts['ofp']
        self.ofname = '%s.txt' % self.ofpref
        self.ofplot = '%s.png' % self.ofpref
        self.pixsize = popts.psz
        self.txy = popts.txy
        self.toz = popts.toz
        self.vrb = popts.vrb
        self.plt = popts.plt
        self.his = popts.his
        self.gfn = popts.gfn

        if self.vrb & DEBUG: print('In %s' % sys._getframe().f_code.co_name)
        if self.vrb & INFO: print_command_line()
        if self.vrb & INFO: print_command_line_parameters(self.parser)


    def read_optical_metrology_file(self):
        """Reads the metrology.txt file with original optical measurements
        """
        if self.vrb & INFO: print('%s\nIn %s: %s' % (60*'-', sys._getframe().f_code.co_name, self.ifname))

                                 # quad 0:3
                                   # point 1:32
                                      # record: point, X, Y, Z 0:3
        #self.arr_opt = np.zeros( (self.nquads,self.npoints+1,4), dtype=numpy.int32 )

        self.arr_opt = []

        if not os.path.lexists(self.ifname):
            raise IOError('File "%s" is not available' % self.ifname)

        quad = 0

        file = open(self.ifname, 'r')

        for linef in file:

            line = linef.strip('\n').strip()

            if not line:
                if self.vrb & INFO: print('EMPTY LINE IS IGNORED')
                continue   # discard empty strings

            if line[0] == '#': # comment
                if self.vrb & INFO: print('COMMENT IS IGNORED: "%s"' % line)
                continue

            list_of_fields = line.split()
            field0 = list_of_fields[0]

            if field0.lower() == 'quad': # Treat quad header lines
                quad = int(list_of_fields[1])
                if self.vrb & INFO: print('IS RECOGNIZED QUAD: %d' % quad)
                continue

            if field0.lower() in ('point', 'sensor'): # Treat the title lines
                if self.vrb & INFO: print('TITLE LINE:     %s' % line)
                continue

            if not field0.lstrip("-+").isdigit():  # is 1-st field digital?
                if self.vrb & INFO: print('RECORD IS IGNORED 1 due to unexpected format of the line: %s' % line)
                continue

            if len(list_of_fields) != 4: # Ignore lines with non-expected number of fields
                if self.vrb & INFO: print('len(list_of_fields) =', len(list_of_fields), end=' ')
                if self.vrb & INFO: print('RECORD IS IGNORED 2 due to unexpected format of the line: %s' % line)
                continue

            n, x, y, z = [int(v) for v in list_of_fields]

            if self.vrb & INFO: print('ACCEPT RECORD: %3d %7d %7d %7d ' % (n, x, y, z))

            self.arr_opt.append((n, x, y, z, quad))

        file.close()

        #if self.vrb & DEBUG: print '\nArray of alignment info:\n', self.arr_opt

        self.arr = np.array(self.arr_opt, dtype= np.int32)
        #print self.arr


    def make_table_of_segments(self):
        """Reshape optical metrology table to arr_segs;
           arr_segs.shape=(nsegs, 4(points-per-segment), 4(n, x, y, z))
           NOTE: npoints may not nesserely be dividable by 4...

           Input: self.arr
           Output: self.arr_segs
        """
        if self.vrb & DEBUG: print('In %s' % (sys._getframe().f_code.co_name))

        npoints = self.arr.shape[0]
        nsegs = npoints//4
        self.arr_segs = np.empty(shape=(nsegs, 4, 5), dtype=np.int64)

        npoints = nsegs*4
        for i in range(npoints):
            nseg = i//4 # [0, npoints/4]
            npoi = i%4 # [0,3]
            #print 'XXX nseg: %d  npoi: %d' % (nseg, npoi)
            self.arr_segs[nseg, npoi,:] = self.arr[i,:]

        if self.vrb & DEBUG: print(self.arr_segs)


    def check_points_numeration(self):
        if self.vrb & INFO: print('%s\n%s' % (60*'-', sys._getframe().f_code.co_name))

        if is_correct_numeration(self.arr_segs[:,:,0].flatten()): print('OK - points in table are sequential')
        else: print('WARNING - numeration of points in table is NOT sequential or started from non-x4 number')

        nsegs = self.arr_segs.shape[0]
        for nseg in range(nsegs):
            pnums = self.arr_segs[nseg,:,0]
            print('Measured segment %2d  point numbers: (%3d %3d %3d %3d)'%\
                   (nseg, pnums[0], pnums[1], pnums[2], pnums[3]), end=' ')
            if is_correct_numeration(pnums): print('OK - points in segment are sequential')
            else: print('WARNING - numeration of points in segment is NOT sequential or started from non-x4 number')


    def test_loop_over_segments(self):
        nsegs = self.arr_segs.shape[0]
        for nseg in range(nsegs):
            arr1seg = self.arr_segs[nseg,:,:4]
            x_um, y_um, z_um = self.segment_center_coordinates(arr1seg)
            self.evaluate_length_width_angle(arr1seg)
            self.evaluate_deviation_from_flatness(arr1seg)
            if self.vrb & DEBUG:
                print('segment center x, y, z (um) = %.1f, %.1f, %.1f' % (x_um, y_um, z_um))


    def print_quality_check_tables(self):
        if self.vrb & DEBUG: print('In %s' % (sys._getframe().f_code.co_name))

        txt = self.txt_qc_table_xy()
        print('X-Y quality check for optical metrology measurements \n%s' % txt)

        txt = self.txt_qc_table_z()
        print('Z quality check for optical metrology measurements \n%s' % txt)


    def segment_center_coordinates(self, arr1seg):
        """Returns segment center coordinates x, y, z in micrometer
           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: x_um, y_um, z_um - segment center coordinates
        """
        #n, x_um, y_um, z_um = 0.25 * arr1seg.sum(axis=0)
        return 0.25 * arr1seg.sum(axis=0)[1:4]


    def get_segment_vectors(self, arr1seg, iorgn=0):
        """Returns segment vectors relative to its origin point (def iorgn=0).
           In perfect world origin is a one of [0,3] corner nearest to pixel(0,0).
           However,
           for quality check it is not necessary.

           1) makes dictionary of 3 vectors from segment origin corner to 3 other corners,
           2) orders dictionary by vector length and assign them to vS1, vL1, and vD1,
           3) find complementary vectors vS2, vL2, vD2 and return results.

           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: vS1, vS2, vL1, vL2, vD1, vD2
        """
        dic_v = {}
        for i in range(4):
            if i == iorgn: continue
            v = arr1seg[i,1:] - arr1seg[iorgn,1:]
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


    def evaluate_short_long_average(self):
        self.dZSA = 0.5 * (self.dZS1 + self.dZS2)
        self.dZLA = 0.5 * (self.dZL1 + self.dZL2)
        self.SA   = 0.5 * (self.S1   + self.S2)
        self.LA   = 0.5 * (self.L1   + self.L2)
        self.dSA  = 0.5 * (self.dS1  + self.dS2)
        self.dLA  = 0.5 * (self.dL1  + self.dL2)


    def evaluate_length_width_angle(self, arr1seg):
        """
           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: x_um, y_um, z_um - segment center coordinates
        """
        #if self.vrb & DEBUG: print 'In %s' % (sys._getframe().f_code.co_name)

        ix, iy, iz = 0, 1, 2

        vS1, vS2, vL1, vL2, vD1, vD2 = self.get_segment_vectors(arr1seg, iorgn=0)

        orient = degrees(atan2(vL1[iy], vL1[ix]))

        gate_deg = 10
        segm_n90 = 0 if fabs(orient)     < gate_deg else\
                   1 if fabs(orient-90)  < gate_deg else\
                   2 if fabs(orient-180) < gate_deg else\
                   2 if fabs(orient+180) < gate_deg else\
                   3
        #print 'XXX segment orientation: %5.1f  n90:%d' % (orient, segm_n90)
        #return

        if segm_n90 in (0,2):
            # horizontal segment

            self.S1   = vS1[iy]
            self.S2   = vS2[iy]

            self.dS1  = vL1[iy]
            self.dS2  = vL2[iy]

            self.L1   = vL1[ix]
            self.L2   = vL2[ix]

            self.dL1  = vS1[ix]
            self.dL2  = vS2[ix]

            self.dZS1 = vS1[iz]
            self.dZS2 = vS2[iz]
            self.dZL1 = vL1[iz]
            self.dZL2 = vL2[iz]

            self.evaluate_short_long_average()

            self.XSize = fabs(self.LA)
            self.YSize = fabs(self.SA)
            self.dZX   = self.dZLA
            self.dZY   = self.dZSA

        if segm_n90 in (1,3):
            # vertical segment

            self.S1   = vS1[ix]
            self.S2   = vS2[ix]

            self.dS1  = vL1[ix]
            self.dS2  = vL2[ix]

            self.L1   = vL1[iy]
            self.L2   = vL2[iy]

            self.dL1  = vS1[iy]
            self.dL2  = vS2[iy]

            self.dZS1 = vS1[iz]
            self.dZS2 = vS2[iz]
            self.dZL1 = vL1[iz]
            self.dZL2 = vL2[iz]

            self.evaluate_short_long_average()

            self.XSize = fabs(self.SA)
            self.YSize = fabs(self.LA)
            self.dZX   = self.dZSA
            self.dZY   = self.dZLA

        self.D1   = sqrt(np.sum(np.square(vD1)))
        self.D2   = sqrt(np.sum(np.square(vD2)))
        self.dD   = self.D1 - self.D2

        self.ddS  = self.dS1 - self.dS2
        self.ddL  = self.dL1 - self.dL2

        self.ddZS = self.dZS1 - self.dZS2
        self.ddZL = self.dZL1 - self.dZL2

        self.rotXYDegree = segm_n90 * 90
        self.rotXZDegree = 0
        self.rotYZDegree = 0

        vLA = 0.5*(vL1+vL2)
        vSA = 0.5*(vS1+vS2)

        tiltXY = atan2(vLA[iy], vLA[ix])
        tiltXZ = atan2(self.dZX, self.XSize)
        tiltYZ = atan2(self.dZY, self.YSize)

        #vLlen = sqrt(np.sum(np.square(vLA)))
        #vSlen = sqrt(np.sum(np.square(vSA)))
        #tiltXZ = atan2(vLA[iz], vLlen)
        #tiltYZ = atan2(vSA[iz], vSlen)

        if abs(tiltXY)>0.1 and tiltXY<0: tiltXY += 2*pi # move angle in range [0,2pi]

        self.tiltXYDegree = degrees(tiltXY) - self.rotXYDegree
        self.tiltXZDegree = degrees(tiltXZ) - self.rotXZDegree
        self.tiltYZDegree = degrees(tiltYZ) - self.rotYZDegree

        #print 'rotXY: %f' % self.rotXYDegree
        #print 'tiltXY: %f' % self.tiltXYDegree


    def txt_qc_table_xy(self):
        """Returns (str)  text of the quality check table
           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: text of the quality check table
        """
        if self.vrb & DEBUG: print('%s' % (sys._getframe().f_code.co_name))

        sepline = '%s\n' % (124*'-')
        txt = sepline
        txt += 'segm:        S1      S2     dS1     dS2        L1      L2     dL1     dL2    angle(deg)   D1      D2      dD   d(dS)   d(dL)\n'
        txt += sepline
        wrg = '\n'

        nsegs = self.arr_segs.shape[0]
        for nseg in range(nsegs):
            arr1seg = self.arr_segs[nseg,:,:4]
            self.evaluate_length_width_angle(arr1seg)

            txt += 'segm:%2d  %6d  %6d  %6d  %6d    %6d  %6d  %6d  %6d   %8.5f  %6d  %6d  %6d  %6d  %6d\n' % \
                (nseg, self.S1, self.S2, self.dS1, self.dS2, \
                       self.L1, self.L2, self.dL1, self.dL2, \
                       self.tiltXYDegree, \
                       self.D1, self.D2, self.dD, self.ddS, self.ddL)
            if fabs(self.dD)  > self.txy: wrg += '  WARNING segm %2d:  |%d| > %.1f\n' % (nseg, self.dD,  self.txy)
            if fabs(self.ddS) > self.txy: wrg += '  WARNING segm %2d:  |%d| > %.1f\n' % (nseg, self.ddS, self.txy)
            if fabs(self.ddL) > self.txy: wrg += '  WARNING segm %2d:  |%d| > %.1f\n' % (nseg, self.ddL, self.txy)
        txt += sepline + wrg
        return txt


    def txt_qc_table_z(self):
        """Returns (str) text of the quality check table
           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: text of the quality check table
        """
        if self.vrb & DEBUG: print('%s' % (sys._getframe().f_code.co_name))

        sepline = '%s\n' % (137*'-')
        txt = sepline
        txt += 'segm:        SA      LA   XSize   YSize    dZS1  dZS2  dZL1  dZL2    dZSA  dZLA  ddZS  ddZL     dZX   dZY   angXZ(deg) angYZ(deg) dz3(um)\n'
        txt += sepline
        wrg = '\n'

        nsegs = self.arr_segs.shape[0]
        for nseg in range(nsegs):
            arr1seg = self.arr_segs[nseg,:,:4]
            self.evaluate_length_width_angle(arr1seg)
            self.evaluate_deviation_from_flatness(arr1seg)

            txt += 'segm:%2d  %6d  %6d  %6d  %6d   %5d %5d %5d %5d   %5d %5d %5d %5d   %5d %5d  %8.5f   %8.5f   %8.3f\n' % \
                (nseg, self.SA,   self.LA,   self.XSize, self.YSize, \
                       self.dZS1, self.dZS2, self.dZL1,  self.dZL2, \
                       self.dZSA, self.dZLA, self.ddZS,  self.ddZL, \
                       self.dZX,  self.dZY,  self.tiltXZDegree, self.tiltYZDegree, \
                       self.arr_dev_um)
            if fabs(self.arr_dev_um) > self.toz: wrg += '  WARNING segm %2d:  |%.1f| > %.1f\n' % (nseg, self.arr_dev_um, self.toz)
        txt += sepline + wrg
        return txt #+'\n'


    def evaluate_deviation_from_flatness(self, arr1seg):
        """Evaluates deviation from segment flatness in micron self.arr_dev_um.
           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: self.arr_dev_um - segment corner 3 deviation from flatness
        """
        #if self.vrb & DEBUG: print 'In %s' % (sys._getframe().f_code.co_name)

        ix, iy, iz = 0, 1, 2

        vS1, vS2, vL1, vL2, vD1, vD2 = self.get_segment_vectors(arr1seg, iorgn=0)

        v21, v31, v41 = vL1, vD1, vS1
        #print v21, v31, v41,

        vort = np.array(np.cross(v21, v41), dtype=np.double) # vort = [v21 x v41]        - vector product
        norm = sqrt(np.sum(vort*vort))                       # norm = |vort|             - length of the vector vort
        vort_norm = vort / norm                              # vort_norm = vort / |vort| - normalized vector orthogonal to the plane
        dev = np.sum(v31*vort_norm)                          # dev = (v31 * vort_norm)   - scalar product

        self.arr_dev_um = dev

        #print '  vort_norm=', vort_norm, '  norm =', norm, '  dev =', dev
        #print '  vort_norm=', vort_norm, '  norm =', norm, '  dev =', dev
        #if self.vrb & DEBUG: print 'quad:%1d, segm:%2d,  dz3[um]: %8.3f' % (quad, segm, self.arr_dev_um[quad,segm])


#--------- GRAPHICS -----------

    def plot_metrology_data(self, offset=2000):
        import matplotlib.pyplot as plt;   global plt
        import matplotlib.lines  as lines; global lines

        if self.vrb & DEBUG: print('In %s' % (sys._getframe().f_code.co_name))

        cmin, cmax = self.arr[:,1].min(), self.arr[:,1].max()
        rmin, rmax = self.arr[:,2].min(), self.arr[:,2].max()

        if self.vrb & DEBUG: print('rmin, rmax =', rmin, rmax)
        if self.vrb & DEBUG: print('cmin, cmax =', cmin, cmax)

        fig = plt.figure(figsize=(8,8), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        axes = fig.add_axes((0.12, 0.08, 0.85, 0.88))
        #axes  = fig.add_subplot(111)

        axes.set_xlim((cmin-offset, cmax+offset))
        axes.set_ylim((rmin-offset, rmax+offset))
        axes.set_xlabel(u'x, \u00B5m', fontsize=14)
        axes.set_ylabel(u'y, \u00B5m', fontsize=14)
        axes.set_title('Check %s' % self.ifname, color='k', fontsize=14)

        self.plot_points(axes)
        self.plot_centers(axes)
        self.plot_geometry_file(axes)

        plt.show()

        fig.savefig(self.ofplot)
        print('Image saved in file: %s' % self.ofplot)


    def plot_points(self, axes):
        arr_segs = self.arr_segs
        nsegs = arr_segs.shape[0]
        for nseg in range(nsegs):
            xlp = arr_segs[nseg,:, 1]; xp = xlp.tolist(); xp.append(xlp[0])
            ylp = arr_segs[nseg,:, 2]; yp = ylp.tolist(); yp.append(ylp[0])
            line = lines.Line2D(xp, yp, linewidth=1, color='r')
            axes.add_artist(line)

            print('segment %02d: xp:' % nseg, xp)

            for point in range(4):
                n, x, y, z, q = arr_segs[nseg, point,:]
                plt.text(x, y, str(n), fontsize=8, color='k', ha='left', rotation=45)


    def plot_centers(self, axes):
        nsegs = self.arr_segs.shape[0]
        for nseg in range(nsegs):
            arr1seg = self.arr_segs[nseg,:,:4]
            x_um, y_um, z_um = self.segment_center_coordinates(arr1seg)
            #if self.vrb & 256: print 'segment center x, y, z (um) = %.1f, %.1f, %.1f' % (x_um, y_um, z_um)
            plt.text(x_um, y_um, str(nseg), fontsize=8, color='k', ha='left', rotation=0)


    def plot_geometry_file(self, axes):
        print('geometry file name: %s' % str(self.gfn))
        if self.gfn is None: return

        geo = GeometryAccess(self.gfn, 0)
        #iX, iY = geo.get_pixel_coord_indexes()
        X, Y, Z = geo.get_pixel_coords() # oname=None, oindex=0, do_tilt=True)
        arrx = reshape_to_3d(X)
        arry = reshape_to_3d(Y)

        #print_ndarr(arrx, 'X pixel coordinates')
        #print_ndarr(arry, 'Y pixel coordinates')

        # find DAQ segment 0 center in geometry
        xc = arrx[0]
        yc = arry[0]
        seg0cgeo_x = np.mean((xc[0,0], xc[0,-1], xc[-1,-1], xc[-1,0]))
        seg0cgeo_y = np.mean((yc[0,0], yc[0,-1], yc[-1,-1], yc[-1,0]))
        print('seg0cgeo_x: %.1f' % seg0cgeo_x)
        print('seg0cgeo_y: %.1f' % seg0cgeo_y)

        # find DAQ segment 0 center in optical geometry (3), or (15), ...
        seg0copm_x = np.mean(self.arr_segs[15,:, 1])
        seg0copm_y = np.mean(self.arr_segs[15,:, 2])
        print('seg0copm_x: %.1f' % seg0copm_x)
        print('seg0copm_y: %.1f' % seg0copm_y)

        xoffset = seg0copm_x - seg0cgeo_x # 527000
        yoffset = seg0copm_y - seg0cgeo_y # 939250

        nsegs = arrx.shape[0]
        for s in range(16):
            xc = arrx[s] + xoffset
            yc = arry[s] + yoffset
            xcor = (xc[0,0], xc[0,-1], xc[-1,-1], xc[-1,0], xc[0,0])
            ycor = (yc[0,0], yc[0,-1], yc[-1,-1], yc[-1,0], yc[0,0])

            print('segment %02d: x corner coordinates: ' % s, xcor)

            line = lines.Line2D(xcor, ycor, linewidth=1, color='b')
            axes.add_artist(line)

#-------- HISTOGRAMS ----------

    def fill_histograms(self):
        """
           Input: arr1seg - array of segment data arr1seg.shape=(4points, 4(n, x, y, z))
           Output: text of the quality check table
        """
        if self.vrb & DEBUG: print('%s' % (sys._getframe().f_code.co_name))

        self.lst_ds_um = []
        self.lst_dl_um = []
        self.lst_dd_um = []
        self.lst_dz_um = []
        self.lst_tilt_z = []
        self.lst_tilt_y = []
        self.lst_tilt_x = []

        nsegs = self.arr_segs.shape[0]
        for nseg in range(nsegs):
            arr1seg = self.arr_segs[nseg,:,:4]
            self.evaluate_length_width_angle(arr1seg)
            self.evaluate_deviation_from_flatness(arr1seg)

            self.lst_ds_um.append(self.ddS)
            self.lst_dl_um.append(self.ddL)
            self.lst_dd_um.append(self.dD)
            self.lst_dz_um.append(self.arr_dev_um)
            self.lst_tilt_z.append(self.tiltXYDegree)
            self.lst_tilt_y.append(self.tiltXZDegree)
            self.lst_tilt_x.append(self.tiltYZDegree)


    def h1d(sp, hlst, bins=None, amp_range=None, weights=None, color='magenta', show_stat=True, log=False,\
        figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80), title='Title', xlabel='x', ylabel='y', titwin=None, fnm='fnm.png'):
        """Wrapper for hist1d, move, and save methods, using common store parameters
        """
        fig, axhi, hi = hist1d(np.array(hlst), bins, amp_range, weights, color, show_stat,\
                           log, figsize, axwin, title, xlabel, ylabel, titwin)

        move(sp.hwin_x0y0[0], sp.hwin_x0y0[1])
        save('%s-%s' % (sp.prefix, fnm), sp.do_save)
        return fig, axhi, hi


    def plot_histograms(sp, prefix='plot', do_save=True, hwin_x0y0 = (0,0)):
        """Plots/saves histograms for intensiry of all and selected peaks in ARC and EQU regions
        """
        sp.prefix    = prefix
        sp.do_save   = do_save
        sp.hwin_x0y0 = hwin_x0y0

        sp.h1d(np.array(sp.lst_ds_um), bins=100, amp_range=(-100,100), \
           title ='Short sides difference', xlabel='dS[um]', ylabel='Segments',\
           fnm='his-ds-um.png')

        sp.h1d(np.array(sp.lst_dl_um), bins=100, amp_range=(-100,100), \
           title ='Long sides difference', xlabel='dL[um]', ylabel='Segments',\
           fnm='his-dl-um.png')

        sp.h1d(np.array(sp.lst_dd_um), bins=100, amp_range=(-100,100), \
           title ='Segment diagonals difference', xlabel='dD[um]', ylabel='Segments',\
           fnm='his-dd-um.png')

        sp.h1d(np.array(sp.lst_dz_um), bins=100, amp_range=(-100,100), \
           title ='Segment non-flatness dz', xlabel='dZ[um]', ylabel='Segments',\
           fnm='his-dz-um.png')

        sp.h1d(np.array(sp.lst_tilt_z), bins=100, amp_range=(-2,2), \
            title ='Segment tilt around z', xlabel='tilt z [deg]', ylabel='Segments',\
           fnm='his-tilt-z.png')

        sp.h1d(np.array(sp.lst_tilt_y), bins=100, amp_range=(-2,2), \
            title ='Segment tilt around y', xlabel='tilt y [deg]', ylabel='Segments',\
           fnm='his-tilt-y.png')

        sp.h1d(np.array(sp.lst_tilt_x), bins=100, amp_range=(-2,2), \
            title ='Segment tilt around x', xlabel='tilt x [deg]', ylabel='Segments',\
           fnm='his-tilt-x.png')

        plt.show()


#------- METHODS FROM ---------
#from pyimgalgos.GlobalGraphics import hist1d, move, save #, show

def add_stat_text(axhi, weights, bins):
    #mean, rms, err_mean, err_rms, neff = proc_stat(weights,bins)
    mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = proc_stat(weights,bins)
    pm = r'$\pm$'
    txt  = 'Entries=%d\nMean=%.2f%s%.2f\nRMS=%.2f%s%.2f\n' % (sum_w, mean, pm, err_mean, rms, pm, err_rms)
    txt += r'$\gamma1$=%.3f  $\gamma2$=%.3f' % (skew, kurt)
    #txt += '\nErr of err=%8.2f' % (err_err)
    xb,xe = axhi.get_xlim()
    yb,ye = axhi.get_ylim()
    #x = xb + (xe-xb)*0.84
    #y = yb + (ye-yb)*0.66
    #axhi.text(x, y, txt, fontsize=10, color='k', ha='center', rotation=0)
    x = xb + (xe-xb)*0.98
    y = yb + (ye-yb)*0.95

    if axhi.get_yscale() == 'log':
        #print 'axhi.get_yscale():', axhi.get_yscale()
        log_yb, log_ye = log10(yb), log10(ye)
        log_y = log_yb + (log_ye-log_yb)*0.95
        y = 10**log_y

    axhi.text(x, y, txt, fontsize=10, color='k',
              horizontalalignment='right',
              verticalalignment='top',
              rotation=0)


def proc_stat(weights, bins):
    center = np.array([0.5*(bins[i] + bins[i+1]) for i,w in enumerate(weights)])

    sum_w  = weights.sum()
    if sum_w <= 0: return  0, 0, 0, 0, 0, 0, 0, 0, 0

    sum_w2 = (weights*weights).sum()
    neff   = sum_w*sum_w/sum_w2 if sum_w2>0 else 0
    sum_1  = (weights*center).sum()
    mean = sum_1/sum_w
    d      = center - mean
    d2     = d * d
    wd2    = weights*d2
    m2     = (wd2)   .sum() / sum_w
    m3     = (wd2*d) .sum() / sum_w
    m4     = (wd2*d2).sum() / sum_w

    #sum_2  = (weights*center*center).sum()
    #err2 = sum_2/sum_w - mean*mean
    #err  = sqrt(err2)

    rms  = sqrt(m2) if m2>0 else 0
    rms2 = m2

    err_mean = rms/sqrt(neff)
    err_rms  = err_mean/sqrt(2)

    skew, kurt, var_4 = 0, 0, 0

    if rms>0 and rms2>0:
        skew  = m3/(rms2 * rms)
        kurt  = m4/(rms2 * rms2) - 3
        var_4 = (m4 - rms2*rms2*(neff-3)/(neff-1))/neff if neff>1 else 0
    err_err = sqrt(sqrt(var_4)) if var_4>0 else 0
    #print  'mean:%f, rms:%f, err_mean:%f, err_rms:%f, neff:%f' % (mean, rms, err_mean, err_rms, neff)
    #print  'skew:%f, kurt:%f, err_err:%f' % (skew, kurt, err_err)
    return mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w


def hist1d(arr, bins=None, amp_range=None, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80), \
           title=None, xlabel=None, ylabel=None, titwin=None):
    """Makes historgam from input array of values (arr), which are sorted in number of bins (bins) in the range (amp_range=(amin,amax))
    """
    #print 'hist1d: title=%s, size=%d' % (title, arr.size)
    if arr.size==0: return None, None, None
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    if   titwin is not None: fig.canvas.set_window_title(titwin)
    elif title  is not None: fig.canvas.set_window_title(title)
    axhi = fig.add_axes(axwin)
    hbins = bins if bins is not None else 100
    hi = axhi.hist(arr.flatten(), bins=hbins, range=amp_range, weights=weights, color=color, log=log) #, log=logYIsOn)
    if amp_range is not None: axhi.set_xlim(amp_range) # axhi.set_autoscale_on(False) # suppress autoscailing
    if title  is not None: axhi.set_title(title, color='k', fontsize=20)
    if xlabel is not None: axhi.set_xlabel(xlabel, fontsize=14)
    if ylabel is not None: axhi.set_ylabel(ylabel, fontsize=14)
    if show_stat:
        weights, bins, patches = hi
        add_stat_text(axhi, weights, bins)
    return fig, axhi, hi


def save(fname='img.png', do_save=True, pbits=0o377):
    if not do_save: return
    if pbits & 1: print('Save plot in file: %s' % fname)
    plt.savefig(fname)


#def save_fig(fig, fname='img.png', do_save=True, pbits=0377):
#    if not do_save: return
#    if pbits & 1: print 'Save plot in file: %s' % fname
#    fig.savefig(fname)


def move(x0=200,y0=100):
    plt.get_current_fig_manager().window.move(x0, y0)


#def move_fig(fig, x0=200, y0=100):
#    fig.canvas.manager.window.move(x0, y0)


def is_correct_numeration(mylst):
    for i, v in enumerate(mylst):
        if i==0:
            if (v-1)%4 != 0: return False
            continue
        if v != (mylst[i-1]+1): return False
    return True


#--------- CL PARSER ----------

def print_command_line(cmt='Command:\n'):
    """Prints command line"""
    print(cmt, ' '.join(sys.argv))


def print_command_line_parameters(parser):
    """Prints input arguments and optional parameters"""
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options
    print('Command:\n ', ' '.join(sys.argv)+\
          '\nArgument list: %s\nOptional parameters:\n' % str(args)+\
          '  <key>      <value>              <default>')
    for k,v in opts.items():
        print('  %s %s %s' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20)))


def usage():
    return '\nCommand to run:'+\
           '\n  %prog'+\
           ' -i <input-file-name> -o <output-file-name> -p -s <pixel-seze-um> -v <verbosity-bitword>'+\
           '\n\n  Example:'+\
           '\n  %prog -i optical-metrology.txt -o results/opmet-2017-04-18 -p'+\
           '\n  alternative:'+\
           '\n  %prog optical-metrology.txt'


def option_parser():

    from optparse import OptionParser

    d_ifn = './optical_metrology.txt'
    d_ofp = './optmet'
    d_psz = 109.92
    d_plt = True
    d_his = False
    d_txy = 60
    d_toz = 100
    d_vrb = 15
    d_gfn = None

    h_ifn   = 'input file name, default = %s' % d_ifn
    h_ofp   = 'output file(s) prefix, default = %s' % d_ofp
    h_psz   = 'pixel size (um)-micrometer, default = %s' % d_psz
    h_plt   = 'plot image, default = %s' % str(d_plt)
    h_his   = 'plot histograms, default = %s' % str(d_his)
    h_txy   = 'tolerance to deviation in x-y plane [um], default = %.2f' % d_txy
    h_toz   = 'tolerance to deviation in z [um], default = %.2f' % d_toz
    h_vrb   = 'verbosity, default = %s' % str(d_vrb)
    h_gfn   = 'input geometry file name, default = %s' % d_gfn

    parser = OptionParser(description='Check optical metrology measurements', usage=usage())
    parser.add_option('-i', '--ifn', default=d_ifn, action='store', type='string', help=h_ifn)
    parser.add_option('-o', '--ofp', default=d_ofp, action='store', type='string', help=h_ofp)
    parser.add_option('-s', '--psz', default=d_psz, action='store', type='float',  help=h_psz)
    parser.add_option('-x', '--txy', default=d_txy, action='store', type='float',  help=h_txy)
    parser.add_option('-z', '--toz', default=d_toz, action='store', type='float',  help=h_toz)
    parser.add_option('-p', '--plt', default=d_plt, action='store_false',          help=h_plt)
    parser.add_option('-H', '--his', default=d_his, action='store_true',           help=h_his)
    parser.add_option('-v', '--vrb', default=d_vrb, action='store', type='int',    help=h_vrb)
    parser.add_option('-g', '--gfn', default=d_gfn, action='store', type='string', help=h_gfn)

    return parser

def do_main():
    omc = OpticalMetrologyCheck()
    sys.exit()

if __name__ == '__main__':
    do_main()

# EOF
