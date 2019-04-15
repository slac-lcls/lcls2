#------------------------------
"""
Class :py:class:`TDPeakRecord` helps to retreive and use peak data in processing
================================================================================

Usage::

    # Imports
    from pyimgalgos.TDPeakRecord import TDPeakRecord

    # Usage

    # make object
    pk = TDPeakRecord(line)

    # access peak attributes
    exp  = pk.exp  # (str)   experiment name
    run  = pk.run  # (int)   run number
    son  = pk.son  # (float) S/N for pixel with maximal intensity 
    sonc = pk.sonc # (float) S/N for all pixels included in the peak
    line = pk.line # (str)   entire record with peak data
    ...

    # Information available through the TDPeakRecord object pk
    # ____________________________________________________
    # pk.exp, pk.run, pk.evnum, pk.reg
    # pk.date, pk.time, pk.tsec, pk.tnsec, pk.fid
    # pk.seg, pk.row, pk.col, pk.amax, pk.atot, pk.npix
    # pk.rcent, pk.ccent, pk.rsigma, pk.csigma
    # pk.rmin, pk.rmax, pk.cmin, pk.cmax
    # pk.bkgd, pk.rms, pk.son
    # pk.imrow, pk.imcol
    # pk.x, pk.y, pk.r, pk.phi
    # pk.sonc
    # pk.dphi000
    # pk.dphi180
    # pk.line
    # pk.nsplit

    # get evaluated parameters
    # pk.peak_signal()
    # pk.peak_noise()
    # pk.peak_son()

    # for peak records with fit information:
    # pk.fit_phi, pk.fit_beta
    # pk.fit_phi_err, pk.fit_beta_err
    # pk.fit_chi2, pk.fit_ndof, pk.fit_prob
            
    # print attributes
    pk.print_peak_data()
    pk.print_peak_data_short()
    pk.print_attrs()

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
#import numpy as np
#from time import strftime, localtime #, gmtime

#------------------------------

class TDPeakRecord :

    def __init__(sp, line, pixel_size = 109.92) :  
        """Parse the string of parameters to values
        """
        ## Exp     Run  Date       Time      time(sec)   time(nsec) fiduc  Evnum  Reg  Seg  Row  Col  Npix      Amax      Atot   rcent   ccent rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son  imrow   imcol     x[um]     y[um]     r[um]  phi[deg]
        #cxif5315  169  2015-02-22 02:20:47  1424600447  494719789  104424     1  EQU   17  170   51    38     168.6    2309.2   169.8    51.6   3.09    1.45  165  176   46   57   -2.90   27.99    6.12    586     499     -8027      -949      8082   -173.26

        sp.pixel_size = pixel_size
        
        sp.fields = line.rstrip('\n').split()
        sp.nfields = nfields = len(sp.fields)

        if nfields in (29,30,36,37) : # r1 peak record: discarded s_rmin, s_rmax, s_cmin, s_cmax, + bkgd corrected amax, atot, son
            s_exp, s_run, s_date, s_time, s_time_sec, s_time_nsec,\
            s_fid, s_evnum, s_reg, s_seg, s_row, s_col, s_npix, s_amax, s_atot,\
            s_rcent, s_ccent, s_rsigma, s_csigma,\
            s_bkgd, s_rms, s_son, s_imrow, s_imcol, s_x, s_y, s_r, s_phi, s_egamma =\
            sp.fields[0:29]

            sp.exp, sp.run, sp.evnum, sp.reg = s_exp, int(s_run), int(s_evnum), s_reg
            sp.date, sp.time, sp.tsec, sp.tnsec, sp.fid = s_date, s_time, int(s_time_sec), int(s_time_nsec), int(s_fid)
            sp.seg, sp.row, sp.col, sp.amax, sp.atot, sp.npix = int(s_seg), int(s_row), int(s_col), float(s_amax), float(s_atot), int(s_npix)
            sp.rcent, sp.ccent, sp.rsigma, sp.csigma = float(s_rcent), float(s_ccent), float(s_rsigma), float(s_csigma)
            sp.rmin, sp.rmax, sp.cmin, sp.cmax = 0, 0, 0, 0
            sp.bkgd, sp.rms, sp.son = float(s_bkgd), float(s_rms), float(s_son)
            sp.imrow, sp.imcol = int(s_imrow), int(s_imcol)
            sp.x, sp.y, sp.r, sp.phi = float(s_x), float(s_y), float(s_r)/sp.pixel_size, float(s_phi)
            sp.sonc = sp.son
            sp.egamma = float(s_egamma)            

        sp.dphi000 = sp.phi
        sp.dphi180 = sp.phi - 180 if sp.phi > -90 else sp.phi + 180 # +360-180

        sp.line = line
        sp.empty = sp.empty_line()

        # get extended parameters for peak record with fit parameters
        if nfields in (30,37) :
            s_nsplit = sp.fields[29]
            sp.nsplit = int(s_nsplit)

        if nfields in (36,37) :
            s_fit_phi, s_fit_beta, s_fit_phi_err, s_fit_beta_err, s_fit_chi2, s_fit_ndof, s_fit_prob = sp.fields[nfields-7:nfields]
            sp.fit_phi, sp.fit_beta = float(s_fit_phi), float(s_fit_beta)
            sp.fit_phi_err, sp.fit_beta_err = float(s_fit_phi_err), float(s_fit_beta_err)
            sp.fit_chi2, sp.fit_ndof, sp.fit_prob = float(s_fit_chi2), int(s_fit_ndof), float(s_fit_prob)
        
#------------------------------
    
    def empty_line(sp) :
       #header = '# Exp     Run  Date       Time      time(sec)   time(nsec) fiduc'
       #addhdr = '  Evnum  Reg  Seg  Row  Col  Npix      Amax      Atot   rcent   ccent '+\
       #         'rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son  imrow   imcol     x[um]     y[um]     r[um]  phi[deg]'

       z = 0

       if sp.nfields == 33 :

         print 'XXX: nfields = ', sp.nfields 


         fmt = '%8s  %3d  %10s %8s  %10d  %9d  %6d' +\
               ' %7d  %3s  %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f' +\
               ' %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f' +\
               ' %6d  %6d  %8.0f  %8.0f  %8.0f  %8.2f  %3d'
         return fmt % ('exp', z, 'date', 'time', z,z,z,z,'N/A',z,z,z,z,z,z,z,z,z,z, z,z,z,z,z,z,z,z,z,z, z,z,z,z)       

       else : #if sp.nfields == 30 : removed rmin rmax cmin cmax
         fmt = '%8s  %3d  %10s %8s  %10d  %9d  %6d' +\
               ' %7d  %3s  %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f' +\
               '  %6.2f  %6.2f  %6.2f' +\
               ' %6d  %6d  %8.0f  %8.0f  %8.0f  %8.2f  %9.3f  %3d'
         return fmt % ('exp', z, 'date', 'time', z,z,z,z,'N/A',z,z,z,z,z,z,z,z,z,z, z,z,z,z,z,z, z,z,z,z,z)       


#------------------------------

    def print_peak_data_short(sp) :
        """Prints short subset of data
        """    
        print '%7d %s %3d %3d %3d %7.1f %7.1f %3d %6d %6d %7.1f %7.1f' % \
              (sp.evnum, sp.reg, sp.seg, sp.row, sp.col, sp.amax, sp.atot, sp.npix, sp.x, sp.y, sp.r, sp.phi)   

#------------------------------

    def print_peak_data(sp) :
        """Prints input data string(line)
        """    
        for field in sp.fields : print field,
        print ''

#------------------------------

    def peak_signal(sp) :
        """Evaluates corrected signal subtracting the background
        """
        return sp.atot-sp.bkgd*sp.npix

#------------------------------

    def peak_noise(sp) :
        """Evaluates corrected rms noise for all pixels in the peak
        """
        return sp.rms*math.sqrt(sp.npix)

#------------------------------

    def peak_son(sp) :
        """Evaluates corrected value of the S/N ratio based on entire peak intensity
        """
        N = sp.peak_noise()
        return sp.peak_signal()/N if N > 0 else 0

#------------------------------

    def print_attrs(sp) :
        msg = 'Attributes of %s, pixel size[um] =%8.2f' % (sp.__class__.__name__, sp.pixel_size)
        #msg += ', line:  \n%s' % (sp.line)
        print msg

#------------------------------

    def print_short(sp) :
        """Alias for interface method
        """
        sp.print_peak_data_short()

#--------------------------------
#-----------  TEST  -------------
#--------------------------------

def test_tdpeakrecord() :
    from pyimgalgos.TDFileContainer import TDFileContainer
    from pyimgalgos.TDPeakRecord    import TDPeakRecord
    import sys

    fname = sys.argv[1] if len(sys.argv) > 1 else 'peaks-cxif5315-r0169-2017-05-24.txt'

    fc = TDFileContainer(fname, indhdr='Evnum', objtype=TDPeakRecord) #, pbits=256)
    fc.print_content(nlines=20)

    return

    counter = 0
    for evnum in fc.group_num_iterator() :
        
        counter += 1
        if counter > 10 : break

        event = fc.next()
        lst_peaks = event.get_objs()

        print '%s Event# %6d %s' % (4*'_', evnum, 4*'_')
        #print '%s\n %s\n%s\n%s' % (71*'_', sp.fc.hdr[:70], lst_peaks[0].line[:71], sp.fc.hdr[72:])

        for peak in lst_peaks :
            #print peak.line.rstrip('\n')[73:]
            print peak.line

#--------------------------------

if __name__ == "__main__" :
    test_tdpeakrecord()

#--------------------------------
