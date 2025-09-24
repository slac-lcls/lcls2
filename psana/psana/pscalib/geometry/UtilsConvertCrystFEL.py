#!/usr/bin/env python
"""
Class :py:class:`UtilsConvertCrystFEL`
======================================

Usage::

    from psana.pscalib.geometry.UtilsConvertCrystFEL import CrystFELGeometryParser

# CrystFEL per asic info:

p15a1/fs = -0.000000x +1.000000y
p15a1/ss = -1.000000x +0.000000y
p15a1/res = 10000.000             # resolution 1m / <pixel-size>
p15a1/corner_x = 2.500000         # in number of pixels
p15a1/corner_y = -628.500000      # in number of pixels
p15a1/coffset = 1.000000          # z-offset correction
p15a1/min_fs = 192
p15a1/max_fs = 383
p15a1/min_ss = 5280
p15a1/max_ss = 5455
p15a1/no_index = 0                # exclude panels from indexing
p15a1/coffset = -0.186288         # z[m] panel offset

@author: Mikhail Dubrovin
2020-09-03 - created for lcls1
2025-08-14 - adopted to lcls2
"""

import os
import sys
import numpy as np
from math import atan2, degrees, sqrt
#from Detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES
#import logging
logger = logging.getLogger(__name__)

#from PSCalib.SegGeometryStore import sgs
from psana.pscalib.geometry.SegGeometryStore import sgs

#import PSCalib.GlobalUtils as gu
import psana.detector.Utils as gu #  save_textfile, load_textfile, get_login, str_tstamp
#from Detector.GlobalUtils import print_ndarr, info_ndarr
#from PSCalib.GlobalUtils import CFRAME_LAB, CFRAME_PSANA


def info_vector(v, msg='', fmt='%10.1f'):
    return msg + ', '.join([fmt%e for e in v])


def str_to_int_or_float(s):
    v = float(s)
    if v%1 == 0: v=int(v)
    return v


def sfields_to_xyz_vector(flds):
    """ converts ['+0.002583x', '-0.999997y', '+0.000000z'] to (0.002583, -0.999997, 0.000000)
    """
    v = (float(flds[0].strip('x')), float(flds[1].strip('y')))
    z = float(flds[2].strip('z')) if len(flds)==3 else 0
    v += (z,)
    return v


def angle_and_tilt(a):
    """for angle in range [-180,180] returns nearest design angle and tilt.
       output angle range is shifted to positive [0,360]
    """
    desangles = np.array((-180,-90, 0, 90, 180))
    difangles = a-desangles
    absdifang = np.absolute(difangles)
    imin = np.where(absdifang == np.amin(absdifang))[0]
    angle, tilt = desangles[imin], difangles[imin]
    return (angle if angle>=0 else angle+360), tilt


def unit_vector_pitch_angle_max_ind(u):
    """unit vector pitch (axis transverse direction in x-y plane) angle
    """
    absu = np.absolute(u)
    imax = np.where(absu == np.amax(absu))[0]
    pitch = degrees(atan2(u[2],u[imax]))
    pitch = (pitch+180) if pitch<-90 else (pitch-180) if pitch>90 else pitch
    return pitch, imax


def vector_lab_to_psana(v):
    """both-way conversion of vectors between LAB and PSANA coordinate frames
    """
    assert len(v)==3
    return np.array((-v[1], -v[0], -v[2]))


def tilt_xy(uf, us, i, k):

    tilt_f, imaxf = unit_vector_pitch_angle_max_ind(uf)
    tilt_s, imaxs = unit_vector_pitch_angle_max_ind(us)
    vmaxf = uf[imaxf]
    vmaxs = us[imaxs]

    logger.debug('== panel %02d %s  tilts f:%.5f s:%.5f' % (i, k, tilt_f, tilt_s)\
      + info_vector(uf, '\n  uf ', '%10.5f')\
      + info_vector(us, '\n  us ', '%10.5f')\
      + '\n  ** imax f:%d s:%d   vmax f:%.5f s:%.5f' % (imaxf, imaxs, vmaxf, vmaxs))

    tilt_x, tilt_y = (tilt_s, tilt_f) if imaxf==0 else (tilt_f, tilt_s)
    return tilt_x, -tilt_y


def str_is_segment_and_asic(s):
    """ check if s looks like str 'q0a2' or 'p12a7'
        returns 'p0.2' or 'p12.7' or False
    """
    if not isinstance(s, str)\
    or len(s)<2: return False
    flds = s[1:].split('a')
    return False if len(flds) !=2 else\
           'p%sa%s' % (flds[0], flds[1]) if all([f.isdigit() for f in flds]) else\
           False


def header_psana(list_of_cmts=[], dettype='N/A'):
    comments = '\n'.join(['# CFELCMT:%02d %s'%(i,s) for i,s in enumerate(list_of_cmts)])
    return\
    '\n# TITLE      Geometry constants converted from CrystFEL by genuine psana'\
    +'\n# DATE_TIME  %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S %Z')\
    +'\n# AUTHOR     %s' % gu.get_login()\
    +'\n# CWD        %s' % gu.get_cwd()\
    +'\n# HOST       %s' % gu.get_hostname()\
    +'\n# COMMAND    %s' % ' '.join(sys.argv)\
    +'\n# RELEASE    %s' % gu.get_enviroment('CONDA_DEFAULT_ENV')\
    +'\n# CALIB_TYPE geometry'\
    +'\n# DETTYPE    %s' % dettype\
    +'\n# DETECTOR   N/A'\
    '\n# METROLOGY  N/A'\
    '\n# EXPERIMENT N/A'\
    +'\n%s' % comments\
    +'\n#'\
    '\n# HDR PARENT IND        OBJECT IND     X0[um]   Y0[um]   Z0[um]   ROT-Z ROT-Y ROT-X     TILT-Z   TILT-Y   TILT-X'


DETTYPE_TO_PARS = {\
  'epix10ka': ('EPIX10KA:V2','p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,'\
                             'p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0'),\
  'jungfrau': ('JUNGFRAU:V2','p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0'),\
  'cspad'   : ('SENS2X1:V1', 'p0a0,p0a2,p0a4,p0a6,p0a8,p0a10,p0a12,p0a14,'\
                             'p1a0,p1a2,p1a4,p1a6,p1a8,p1a10,p1a12,p1a14,'\
                             'p2a0,p2a2,p2a4,p2a6,p2a8,p2a10,p2a12,p2a14,'\
                             'p3a0,p3a2,p3a4,p3a6,p3a8,p3a10,p3a12,p3a14'),\
  'cspadv2' : ('SENS2X1:V1', 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,'\
                             'p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0,'\
                             'p16a0,p17a0,p18a0,p19a0,p20a0,p21a0,p22a0,p23a0,'\
                             'p24a0,p25a0,p26a0,p27a0,p28a0,p29a0,p30a0,p31a0'),\
  'pnccd'   : ('MTRX:V2:512:512:75:75', 'p0a0,p1a0,p2a0,p3a0'),\
}


class CrystFELGeometryParser:
    """ :py:class:`CrystFELGeometryParser`
    """

    def __init__(self, args):
        self._class_name = self.__class__.__name__

        self.args = args
        #self.dic_args = vars(args)

        self.fname = args.fname
        self.ofname = args.ofname
        self.dettype = args.dettype
        self.valid = False

        self.load_crystfel_file()
        self.print_list_of_comments()
        self.print_list_of_ignored_records()
        self.print_dict_of_pars()


    def _parse_line_as_parameter(self, line):
        assert isinstance(line, str), 'line is not a str object'

        fields = line.split()
        nfields = len(fields)

        if fields[1] != '=':
            self.list_of_ignored_records.append(line)
            logger.warning('line does not contain "=", presumably not a parameter, ignored')
            return

        logger.debug('   fields: %s'% str(fields))

        keys = fields[0].split('/') # ex: p15a3/corner_y

        nkeys = len(keys)
        if nkeys==1:
            if nfields>3:
                self.list_of_ignored_records.append(line)
                logger.warning('number of fields >3, ignored')
                return
            k0 = keys[0]
            self.dict_of_pars[k0] = float(fields[2]) if k0 in ('res', 'adu_per_eV', 'coffset') else\
                ' '.join(fields[2:])

        elif nkeys==2:
            k0, k1 = keys
            resp = str_is_segment_and_asic(k0)
            if resp: k0=resp
            v = '' if nfields<3 else\
                sfields_to_xyz_vector(fields[2:]) if k1 in ('fs','ss') else\
                int(fields[2]) if k1 in ('max_ss', 'min_ss', 'max_fs', 'min_fs', 'no_index') else\
                int(fields[2]) if k1 in ('max_x', 'min_x', 'max_y', 'min_y') else\
                float(fields[2]) if k1 in ('res', 'corner_x', 'corner_y', 'adu_per_eV', 'coffset') else\
                float(fields[2]) if k1 in ('xfs', 'yfs', 'xss', 'yss') else\
                ' '.join(fields[2:]) # str_to_int_or_float(fields[2])
            if k0 in self.dict_of_pars.keys():
                self.dict_of_pars[k0][k1] = v
            else:
                self.dict_of_pars[k0] = {k1:v,}

        else:
            self.list_of_ignored_records.append(line)
            logger.warning('field[0]: %s contains unexpected number of keys, ignored' % fields[0])
            return


    def str_list_of_comments(self):
        return 'List of comments\n'\
            + '\n'.join(self.list_of_comments)


    def str_list_of_ignored_records(self):
        return 'List of ignored records\n'\
            + '\n'.join(self.list_of_ignored_records)


    def str_dict_of_pars(self):
        keys = sorted(self.dict_of_pars.keys())
        msg = 'dict of parameters with top keys: %s' % ' '.join(keys)
        for k in keys:
            v = self.dict_of_pars[k]
            if isinstance(v,dict):
                msg += '\n%s: %s' % (k, str_is_segment_and_asic(k))
                for k2,v2 in v.items(): msg += '\n    %s: %s' % (k2,v2)
            else: msg += '\n%s: %s' % (k,v)
        return msg


    def print_list_of_comments(self):
        logger.info(self.str_list_of_comments())


    def print_list_of_ignored_records(self):
        logger.info(self.str_list_of_ignored_records())


    def print_dict_of_pars(self):
        logger.debug(self.str_dict_of_pars())


    def load_crystfel_file(self, fname=None):

        if fname is not None: self.fname = fname
        assert os.path.exists(self.fname), 'geometry file "%s" does not exist' % self.fname

        self.valid = False

        self.list_of_comments = []
        self.list_of_ignored_records = []
        self.dict_of_pars = {}

        logger.debug('Load file: %s' % self.fname)

        f=open(self.fname,'r')
        for linef in f:
            line = linef.strip('\n')
            logger.debug(line)

            if not line.strip(): continue # discard empty strings
            if line[0] == ';':            # accumulate list of comments
                self.list_of_comments.append(line)
                continue

            self._parse_line_as_parameter(line)

        f.close()

        self.valid = True


    def crystfel_to_geometry(self, pars):
        """pattern for conversion: /reg/g/psdm/detector/data2_test/geometry/geo-cspad-xpp.data
        """
        logger.info('crystfel_to_geometry - converts geometry constants from CrystFEL to psana format')

        segname, panasics = pars
        sg = sgs.Create(segname=segname, pbits=0)
        logger.warning('TBE crystfel_to_geometry with segshape: %s' % str(sg.shape())
                       +'\nstr of asics to reconstruct panels geometry:\n%s' % str(panasics))

        X,Y,Z = sg.pixel_coord_array()


        PIX_SIZE_UM = sg.get_pix_size_um()
        M_TO_UM = 1e6
        xc0, yc0, zc0 = X[0,0], Y[0,0], Z[0,0]
        rc0 = sqrt(xc0*xc0+yc0*yc0+zc0*zc0)
        logger.info('distance from panel center to pixel [0,0], um r: %.1f in panel frame x: %.1f y: %.1f z: %.1f' % (rc0, xc0, yc0, zc0))
        logger.info('PIX_SIZE_UM %.2f' % PIX_SIZE_UM)

        zoffset_m = self.dict_of_pars.get('coffset', 0) # in meters

        recs = header_psana(list_of_cmts=self.list_of_comments, dettype=self.dettype)

        segz = np.array([self.dict_of_pars[k].get('coffset', 0) for k in panasics.split(',')])
        logger.info('segment z [m]: %s' % str(segz))
        meanroundz = round(segz.mean()*1e3)*1e-3 # round z to 1mm
        logger.info(   'mean(z), m: %.6f' % meanroundz)
        zoffset_m += meanroundz

        for i,k in enumerate(panasics.split(',')):
            dicasic = self.dict_of_pars[k]
            uf = np.array(dicasic.get('fs', None), dtype=np.float32) # unit vector f
            us = np.array(dicasic.get('ss', None), dtype=np.float32) # unit vector s
            vf = uf*abs(xc0)
            vs = us*abs(yc0)
            x0pix = dicasic.get('corner_x', 0) # The units are pixel widths of the current panel
            y0pix = dicasic.get('corner_y', 0)
            z0m   = dicasic.get('coffset', 0)
            adu_per_eV = dicasic.get('adu_per_eV', 1)
            logger.debug('%s: %s\n   uf: %s  us: %s  vf: %s  vs: %s'%\
                         (k, str(dicasic), str(uf), str(us), str(vf), str(vs)))

            v00center = vf + vs
            v00corner = np.array((x0pix*PIX_SIZE_UM, y0pix*PIX_SIZE_UM, (z0m - zoffset_m)*M_TO_UM))
            vcent = v00corner + v00center

            logger.debug('== panel %s' % k\
              + info_vector(v00center, '\n  v(00->center)')\
              + info_vector(v00corner, '\n  v(00corner)  ')\
              + info_vector(vcent,     '\n  center       '))

            angle_deg = degrees(atan2(uf[1],uf[0]))
            angle_z, tilt_z = angle_and_tilt(angle_deg)
            tilt_x, tilt_y = tilt_xy(uf,us,i,k)

            logger.warning('TBD signs of tilt_x, tilt_y')

            recs += '\nDET:VC         0  %12s  %2d' % (segname, i)\
                  + '   %8d %8d %8d %7.0f     0     0   %8.5f %8.5f %8.5f'%\
                    (vcent[0], vcent[1], vcent[2], angle_z, tilt_z, tilt_y, tilt_x)
        recs += '\nIP             0    DET:VC       0          0        0'\
                ' %8d       0     0     0    0.00000  0.00000  0.00000' % (zoffset_m*M_TO_UM)
        logger.info('geometry constants in psana format:\n\n%s' % recs)

        f=open(self.ofname,'w')
        f.write(recs)
        f.close()
        logger.info('psana geometry constrants saved in file %s' % self.ofname)


    def convert_crystfel_to_geometry(self):
        pars = DETTYPE_TO_PARS.get(self.dettype.lower(), None)
        if pars is None: sys.exit('NON_IMPLEMENTED DETECTOR TYPE: %s' % dettype)
        self.crystfel_to_geometry(pars)


def convert_crystfel_to_geometry(args):
    cgp = CrystFELGeometryParser(args)
    cgp.convert_crystfel_to_geometry()
    sys.exit('TEST EXIT')


if __name__ == "__main__":

    class TestArguments:
        def __init__(self, tname, dettype, fname, ofname, loglev):
            self.tname   = tname
            self.dettype = dettype
            self.fname   = fname
            self.ofname  = ofname
            self.loglev  = loglev
            self.dump_test_arguments()

        def str_test_arguments(self):
            return '\n'.join(['%10s: %s' % (name, str(getattr(self, name, None)))\
                   for name in dir(self) if name[0]!='_' and not callable(getattr(self, name, None))])

        def dump_test_arguments(self):
            logger.info('dir(TestArguments):' + ' '.join(dir(self)))
            logger.info('TestArguments:\n%s' % self.str_test_arguments())


    def test_converter(*args):
        targs = TestArguments(*args)
        convert_crystfel_to_geometry(targs)


    scrname = sys.argv[0].rsplit('/')[-1]
    #dirdt = '/reg/g/psdm/detector/data_test/'
    dirdt = '/sdf/group/lcls/ds/ana/detector/data2_test/'
    fname_epix10ka2m = dirdt + 'geometry/crystfel/geo-mfxc00318-epix10ka2m.1-0013-z0-mirror.geom'
    fname_cspad      = dirdt + 'geometry/crystfel/geo-cxig0915-cspad-ds1-crystfel.geom'
    fname_cspadv2    = dirdt + 'geometry/crystfel/geo-cspadv2-test-cframe-psana.geom'
    fname_pnccd      = dirdt + 'geometry/crystfel/geo-amox26916-pnccd-front-108-psana-crystfel.geom'
    fname_jungfrau   = dirdt + 'geometry/crystfel/geo-jungfrau-8-test-cframe-psana.geom'

    d_tname   = '0'
    d_dettype = 'cspad' # 'epix10ka' 'pnccd' 'jungfrau'
    d_fname   = fname_cspad
    d_ofname  = 'geo-psana.txt'
    d_loglev  ='INFO'

    usage = '\nE.g.: %s' % scrname\
      + '\n  or: %s -t <test-number: 1,2,3,4,5,...>' % (scrname)\
      + '\n  or: %s -d epix10ka -f %s -o geo_crystfel.txt -l DEBUG' % (scrname, d_fname)

    import argparse

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('-t', '--tname',   default=d_tname,   type=str, help='test number: 1/2/3/4/5 = epix10ka/jungfrau/cspad/cspadv2(xpp)/pnccd def=%s'%d_tname)
    parser.add_argument('-d', '--dettype', default=d_dettype, type=str, help='detector type, i.e. epix10ka, jungfrau, cspad, pnccd, def=%s' % d_dettype)
    parser.add_argument('-f', '--fname',   default=d_fname,   type=str, help='input geometry file name, def=%s' % d_fname)
    parser.add_argument('-o', '--ofname',  default=d_ofname,  type=str, help='output file name, def=%s' % d_ofname)
    parser.add_argument('-l', '--loglev',  default=d_loglev,  type=str, help='logging level name, one of (INFO, DEBUG, ERROR,...), def=%s' % d_loglev)

    args = parser.parse_args()
    print('Arguments:') #%s\n' % str(args))
    for k,v in vars(args).items(): print('  %12s : %s' % (k, str(v)))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=DICT_NAME_TO_LEVEL[args.loglev])

    tname = args.tname

    if   tname=='0': convert_crystfel_to_geometry(args)
    elif tname=='1': test_converter(tname, 'epix10ka', fname_epix10ka2m, 'geo-epix10ka-psana-from-crystel.txt', args.loglev)
    elif tname=='2': test_converter(tname, 'jungfrau', fname_jungfrau,   'geo-jungfrau-psana-from-crystel.txt', args.loglev)
    elif tname=='3': test_converter(tname, 'cspad',    fname_cspad,      'geo-cspad-psana-from-crystel.txt', args.loglev)
    elif tname=='4': test_converter(tname, 'cspadv2',  fname_cspadv2,    'geo-cspadv2-psana-from-crystel.txt', args.loglev)
    elif tname=='5': test_converter(tname, 'pnccd',    fname_pnccd,      'geo-pnccd-psana-from-crystel.txt', args.loglev)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)

    sys.exit('END OF %s' % scrname)

# EOF
