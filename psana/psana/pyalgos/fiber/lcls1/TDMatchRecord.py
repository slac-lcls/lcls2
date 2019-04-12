#------------------------------
"""
Class :py:class:`TDMatchRecord` helps to retreive and use peak data in processing
=================================================================================

Usage::

    # import
    from pyimgalgos.TDMatchRecord import TDMatchRecord

    # make object
    rec = TDMatchRecord(line)

    # access record attributes
    index, beta, omega, h, k, l, dr, R, qv, qh, P =
    rec.index, rec.beta, rec.omega, rec.h, rec.k, rec.l, rec.dr, rec.R, rec.qv, rec.qh, rec.P
    line = rec.line

    # print attributes
    rec.print_short()

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

Created in 2015 by Mikhail Dubrovin"""

#--------------------------------

from pyimgalgos.TDPeakRecord import TDPeakRecord
from pyimgalgos.TDNodeRecord import TDNodeRecord

from time import strftime, localtime #, gmtime, time

#------------------------------

class TDMatchRecord(TDPeakRecord, TDNodeRecord) :

    def __init__(sp, line, pixel_size=109.92) :
        """Parse the string of parameters to values
        """
        #sp.fields = line.rstrip('\n').split()
        #TDPeakRecord.__init__(sp, line)

        sp.pixel_size = pixel_size
        sp.fields_mr = line[:-1].split()

        sp.nfields_mr = nfields = len(sp.fields_mr)

        if sp.nfields_mr == 47 :

          TDNodeRecord.__init__(sp, line[284:])

          s_state, s_phi_fit, s_beta_fit, s_qh_fit, s_qv_fit, s_dqh_fit,\
          s_exp, s_run, s_time_sec, s_time_nsec,\
          s_fid, s_evnum, s_reg, s_seg, s_row, s_col, s_npix, s_amax, s_atot,\
          s_rcent, s_ccent, s_rsigma, s_csigma, s_rmin, s_rmax, s_cmin, s_cmax,\
          s_bkgd, s_rms, s_son, s_imrow, s_imcol, s_x, s_y, s_r, s_phi =\
          sp.fields_mr[0:36]

          sp.state, sp.phi_fit, sp.beta_fit = s_state, float(s_phi_fit), float(s_beta_fit)
          sp.qh_fit, sp.qv_fit, sp.dqh_fit = float(s_qh_fit), float(s_qv_fit), float(s_dqh_fit)
          sp.exp, sp.run, sp.evnum, sp.reg = s_exp, int(s_run), int(s_evnum), s_reg
          sp.tsec, sp.tnsec, sp.fid = int(s_time_sec), int(s_time_nsec), int(s_fid)
          sp.seg, sp.row, sp.col, sp.amax, sp.atot, sp.npix = int(s_seg), int(s_row), int(s_col), float(s_amax), float(s_atot), int(s_npix)
          sp.rcent, sp.ccent, sp.rsigma, sp.csigma = float(s_rcent), float(s_ccent), float(s_rsigma), float(s_csigma)
          sp.rmin, sp.rmax, sp.cmin, sp.cmax = int(s_rmin), int(s_rmax), int(s_cmin), int(s_cmax)
          sp.bkgd, sp.rms, sp.son = float(s_bkgd), float(s_rms), float(s_son)
          sp.imrow, sp.imcol = int(s_imrow), int(s_imcol)
          sp.x, sp.y, sp.r, sp.phi = float(s_x), float(s_y), float(s_r)/sp.pixel_size, float(s_phi)

        if sp.nfields_mr == 43 :

          TDNodeRecord.__init__(sp, line[262:])

          s_state, s_phi_fit, s_beta_fit, s_qh_fit, s_qv_fit, s_dqh_fit,\
          s_exp, s_run, s_time_sec, s_time_nsec,\
          s_fid, s_evnum, s_reg, s_seg, s_row, s_col, s_npix, s_amax, s_atot,\
          s_rcent, s_ccent, s_rsigma, s_csigma,\
          s_bkgd, s_rms, s_son, s_imrow, s_imcol, s_x, s_y, s_r, s_phi =\
          sp.fields_mr[0:32]
          #s_rmin, s_rmax, s_cmin, s_cmax = '0','0','0','0' # fields removed from peak record

          sp.state, sp.phi_fit, sp.beta_fit = s_state, float(s_phi_fit), float(s_beta_fit)
          sp.qh_fit, sp.qv_fit, sp.dqh_fit = float(s_qh_fit), float(s_qv_fit), float(s_dqh_fit)
          sp.exp, sp.run, sp.evnum, sp.reg = s_exp, int(s_run), int(s_evnum), s_reg
          sp.tsec, sp.tnsec, sp.fid = int(s_time_sec), int(s_time_nsec), int(s_fid)
          sp.seg, sp.row, sp.col, sp.amax, sp.atot, sp.npix = int(s_seg), int(s_row), int(s_col), float(s_amax), float(s_atot), int(s_npix)
          sp.rcent, sp.ccent, sp.rsigma, sp.csigma = float(s_rcent), float(s_ccent), float(s_rsigma), float(s_csigma)
          #sp.rmin, sp.rmax, sp.cmin, sp.cmax = int(s_rmin), int(s_rmax), int(s_cmin), int(s_cmax)
          sp.bkgd, sp.rms, sp.son = float(s_bkgd), float(s_rms), float(s_son)
          sp.imrow, sp.imcol = int(s_imrow), int(s_imcol)
          sp.x, sp.y, sp.r, sp.phi = float(s_x), float(s_y), float(s_r)/sp.pixel_size, float(s_phi)
        
        sp.set_date_time(sp.tsec) #sp.date, sp.time = '2015-11-13', '16:00:00'
        sp.sonc = sp.peak_son()
        sp.dphi000 = sp.phi
        sp.dphi180 = sp.phi - 180 if sp.phi > -90 else sp.phi + 180 # +360-180

        sp.line  = line
        sp.empty = sp.empty_line()
        #print line
        
#------------------------------
    
    def set_date_time(sp, tsec=None) :
        tstamp = strftime('%Y-%m-%d %H:%M:%S', localtime(tsec))
        sp.date, sp.time = tstamp.split()

#------------------------------
    
    def empty_line(sp) :
        fmt = '%6d  %7.2f %7.2f %3d %3d %3d %9.6f %9.6f %9.6f %9.6f %9.6f'
        z = 0
        return fmt % (z,z,z,z,z,z,z,z,z,z,z)       

#------------------------------

    def print_short(sp) :
        """Prints short subset of data
        """    
        print '%6d  %7.2f  %7.2f  %2d %2d %2d    %9.6f  %9.6f  %9.6f  %9.6f  %9.6f' % \
              (sp.index, sp.beta, sp.omega, sp.h, sp.k, sp.l, sp.dr, sp.R, sp.qv, sp.qh, sp.P)   

#------------------------------
#--------------------------------
#-----------  TEST  -------------
#--------------------------------

if __name__ == "__main__" :
    pass

#--------------------------------
