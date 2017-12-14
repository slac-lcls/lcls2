#------------------------------
"""
Class :py:class:`TDNodeRecord` helps to retreive and use peak data in processing
================================================================================

Usage::

    # Imports
    from pyimgalgos.TDCheetahPeakRecord import TDCheetahPeakRecord

    # Usage

    # make object
    rec = TDCheetahPeakRecord(line)

    # access peak attributes
    rec.line
    rec.fields        

    rec.frameNumber
    rec.runnum
    rec.tstamp
    rec.fid
    rec.photonEnergyEv
    rec.wavelengthA
    rec.GMD
    rec.peak_index
    rec.peak_x_raw
    rec.peak_y_raw
    rec.peak_r_assembled
    rec.peak_q
    rec.peak_resA
    rec.nPixels
    rec.totalIntensity
    rec.maxIntensity
    rec.sigmaBG
    rec.SNR
    rec.tsec

    # print attributes
    rec.print_peak_data()
    rec.print_peak_data_short()
    rec.print_attrs()

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
  - `Analysis of data for cxif5315 <https://confluence.slac.stanford.edu/display/PSDMInternal/Analysis+of+data+for+cxif5315>`_.

Created in 2015 by Mikhail Dubrovin
"""

#--------------------------------

import math
from pyimgalgos.GlobalUtils import convertCheetahEventName, src_from_rc8x8

#------------------------------

class TDCheetahPeakRecord :

    def __init__(sp, line) :  
        """Parse the string of parameters to values
        """
        ## frameNumber, eventName,              photonEnergyEv, wavelengthA, GMD,      peak_index, peak_x_raw, peak_y_raw, peak_r_assembled, peak_q,   peak_resA, nPixels, totalIntensity, maxIntensity, sigmaBG,   SNR
        #5, LCLS_2015_Feb22_r0169_022047_197ee, 6004.910515,    2.064714,    4.262349, 29997,      508.884796, 19.449471,  441.314606,       1.741234, 5.743053,  5,       361.105774,     112.819145,   19.236982, 18.771435

        sp.line   = line[:-1] #.rstrip('\n') # .replace(',',' ')
        sp.fields = sp.line.split()

        s_frameNumber, s_eventName, s_photonEnergyEv, s_wavelengthA, s_GMD, s_peak_index, s_peak_x_raw, s_peak_y_raw,\
        s_peak_r_assembled, s_peak_q, s_peak_resA, s_nPixels, s_totalIntensity, s_maxIntensity, s_sigmaBG, s_SNR =\
        sp.fields[0:16]

        sp.frameNumber, sp.photonEnergyEv, sp.wavelengthA        = int(s_frameNumber), float(s_photonEnergyEv), float(s_wavelengthA)
        sp.GMD, sp.peak_index, sp.peak_x_raw, sp.peak_y_raw      = float(s_GMD), int(s_peak_index), float(s_peak_x_raw), float(s_peak_y_raw)
        sp.peak_r_assembled, sp.peak_q, sp.peak_resA, sp.nPixels = float(s_peak_r_assembled), float(s_peak_q), float(s_peak_resA), int(s_nPixels)
        sp.totalIntensity, sp.maxIntensity, sp.sigmaBG, sp.SNR   = float(s_totalIntensity), float(s_maxIntensity), float(s_sigmaBG), float(s_SNR)

        sp.runnum, sp.tstamp, sp.tsec, sp.s_fid = convertCheetahEventName(s_eventName)
        sp.fid = int(sp.s_fid, 16)

        #sp.seg, sp.row, sp.col = src_from_rc8x8(sp.peak_y_raw, sp.peak_x_raw)

        sp.line = line
        sp.empty = sp.empty_line()
        
#------------------------------
    
    def empty_line(sp) :
       #header = '# Exp     Run  Date       Time      time(sec)   time(nsec) fiduc'
       #addhdr = '  Evnum  Reg  Seg  Row  Col  Npix      Amax      Atot   rcent   ccent '+\
       #         'rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son  imrow   imcol     x[um]     y[um]     r[um]  phi[deg]'
       #fmt = '%8s  %3d  %10s %8s  %10d  %9d  %6d'+\
       #      ' %7d  %3s  %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f'+\
       #      ' %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f'+\
       #      ' %6d  %6d  %8.0f  %8.0f  %8.0f  %8.2f'
       #z=0
       #return fmt % ('exp', z, 'date', 'time', z,z,z,z,'N/A',z,z,z,z,z,z,z,z,z,z, z,z,z,z,z,z,z,z,z,z, z,z,z)       
       return 'empty_line'

#------------------------------

    def print_peak_data_short(sp) :
        """Prints short subset of data
        """    
        print '%d %4d  %s     %s  %.6f  %.6f  %.6f  %d  %.6f  %.6f  %.6f  %.6f  %.6f  %d  %.6f  %.6f  %.6f  %.6f  %d  %d' % \
              (sp.frameNumber, sp.runnum, sp.tstamp, sp.s_fid,\
               sp.photonEnergyEv, sp.wavelengthA,\
               sp.GMD, sp.peak_index, sp.peak_x_raw, sp.peak_y_raw,\
               sp.peak_r_assembled, sp.peak_q, sp.peak_resA, sp.nPixels,\
               sp.totalIntensity, sp.maxIntensity, sp.sigmaBG, sp.SNR,\
               sp.tsec, sp.fid)

#------------------------------

    def print_seg_row_col(sp) :
        """prints peak seg, row, col
        """    
        s, r, c = src_from_rc8x8(sp.peak_y_raw, sp.peak_x_raw)
        print 'seg: %d, row: %.1f, col: %.1f' % (s, r, c)

#------------------------------

    def seg_row_col(sp) :
        """returns peak (int) seg, (int or float) row, (int or float) col
        """    
        return src_from_rc8x8(sp.peak_y_raw, sp.peak_x_raw)

#------------------------------

    def print_peak_data(sp) :
        """Prints input data string(line)
        """    
        for field in sp.fields : print field,
        print ''

#------------------------------

    def print_attrs(sp) :
        msg = 'Attributes of %s - there is no extra attributes.' % (sp.__class__.__name__)
        #msg += ', line:  \n%s' % (sp.line)
        print msg

#------------------------------

    def print_short(sp) :
        """Alias for interface method
        """
        sp.print_peak_data_short()

#------------------------------
#------------------------------
#----------- TEST - -----------
#------------------------------

if __name__ == "__main__" :
    pass

#------------------------------
