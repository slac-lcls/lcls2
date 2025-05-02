#### !/usr/bin/env python

"""
:py:class:`CalibConstants` - global constants for Calib project
===================================================================

Usage ::

    # python lcls2/psana/pscalib/calib/CalibConstants.py

    from psana.pscalib.calib.CalibConstants import *

    for k,v in dic_det_type_to_name.items() : print('%16s : %s' % (str(k), str(v)))

See:
 * :py:class:`CalibBase`
 * :py:class:`CalibConstant`

For more detail see `Calibration Store <https://confluence.slac.stanford.edu/display/PCDS/MongoDB+evaluation+for+calibration+store>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2018-02-02 by Mikhail Dubrovin
"""

import os
import sys
import numpy as np

#import kerberos
from krtc import KerberosTicket
from urllib.parse import urlparse
import getpass

URL_ENV = os.environ.get('LCLS_CALIB_HTTP', None)
URL     = 'https://pswww.slac.stanford.edu/calib_ws' if URL_ENV is None else URL_ENV
URL_KRB = 'https://pswww.slac.stanford.edu/ws-kerb/calib_ws/'
#URL     = 'https://psdmint.sdf.slac.stanford.edu/calib_ws' if URL_ENV is None else URL_ENV
#URL_KRB = 'https://psdmint.sdf.slac.stanford.edu/ws-kerb/calib_ws/'
HOST = 'psdb02' # psdb01/02/03/04 'psdbdev01' # 'psanaphi103'
PORT = 9307
USERLOGIN = getpass.getuser()
USERNAME = USERLOGIN
USERPW = 'pw-should-be-provided-somehow'
DBNAME_PREFIX = 'cdb_'
DETNAMESDB = '%sdetnames' % DBNAME_PREFIX
#MAX_DETNAME_SIZE = 55
#MAX_DETNAME_SIZE = 40
MAX_DETNAME_SIZE = 20
OPER = os.getenv('CALIBDB_AUTH')

try: KRBHEADERS = KerberosTicket("HTTP@" + urlparse(URL_KRB).hostname).getAuthHeaders()
#except kerberos.GSSError as e:
except Exception as e:
    #msg = e.message if hasattr(e, 'message') else str(e)
    #print('Exception:', msg)
    #sys.exit('Fix kerberos ticket - use command kinit')
    KRBHEADERS = None

TSFORMAT = '%Y-%m-%dT%H:%M:%S%z' # e.g. 2018-02-07T08:40:28-0800
TSFORMAT_SHORT = '%Y%m%d%H%M%S'  # e.g. 20180207084028

# Enumerated and named parameters

PEDESTALS     = 0
PIXEL_STATUS  = 1
PIXEL_RMS     = 2
PIXEL_GAIN    = 3
PIXEL_MASK    = 4
PIXEL_BKGD    = 5
COMMON_MODE   = 6
GEOMETRY      = 7
PIXEL_OFFSET  = 8
PIXEL_DATAST  = 9
CODE_GEOMETRY = 10
LASINGOFFREFERENCE = 11
USERS_MASK    = 12
STATUS_EXTRA  = 13
PIXEL_MAX     = 14
PIXEL_MIN     = 15

ctype_tuple = (
    (PEDESTALS,      'pedestals',     np.float32, 'p'),
    (PIXEL_STATUS,   'pixel_status',  np.uint64 , 's'),
    (PIXEL_RMS,      'pixel_rms',     np.float32, 'r'),
    (PIXEL_GAIN,     'pixel_gain',    np.float32, 'g'),
    (PIXEL_MASK,     'pixel_mask',    np.uint8  , 'm'),
    (PIXEL_BKGD,     'pixel_bkgd',    np.float32, 'b'),
    (COMMON_MODE,    'common_mode',   np.float64, 'c'),
    (GEOMETRY,       'geometry',      str       , 'G'),
    (PIXEL_OFFSET,   'pixel_offset',  np.float32, 'o'),
    (PIXEL_DATAST,   'pixel_datast',  np.uint16 , 'd'),
    (CODE_GEOMETRY,  'code_geometry', str       , 'N'),
    (USERS_MASK,     'users_mask',    np.uint8  , 'u'),
    (STATUS_EXTRA,   'status_extra',  np.uint64 , 'e'),
    (PIXEL_MAX,      'pixel_max',     np.uint64 , 'x'),
    (PIXEL_MIN,      'pixel_min',     np.uint64 , 'n'),
    (LASINGOFFREFERENCE, 'lasingoffreference', 'hdf5', 'N'),
)

list_calib_types  = [rec[0] for rec in ctype_tuple] # (PEDESTALS, PIXEL_STATUS,...)
list_calib_names  = [rec[1] for rec in ctype_tuple] # ('pedestals','pixel_status',...)
list_calib_dtypes = [rec[2] for rec in ctype_tuple] # (np.float32,  np.uint64,...)
list_calib_chars  = [rec[3] for rec in ctype_tuple] # ('p','s','r',...)

dic_calib_type_to_name  = dict(zip(list_calib_types, list_calib_names))
dic_calib_name_to_type  = dict(zip(list_calib_names, list_calib_types))
dic_calib_type_to_dtype = dict(zip(list_calib_types, list_calib_dtypes))
dic_calib_name_to_dtype = dict(zip(list_calib_names, list_calib_dtypes))
dic_calib_char_to_name  = dict(zip(list_calib_chars, list_calib_names))

UNDEFINED   =  0
CSPAD       =  1
CSPAD2X2    =  2
PRINCETON   =  3
PNCCD       =  4
TM6740      =  5
OPAL1000    =  6
OPAL2000    =  7
OPAL4000    =  8
OPAL8000    =  9
ORCAFL40    = 10
EPIX        = 11
EPIX10K     = 12
EPIX100A    = 13
FCCD960     = 14
ANDOR       = 15
ACQIRIS     = 16
IMP         = 17
QUARTZ4A150 = 18
RAYONIX     = 19
EVR         = 20
FCCD        = 21
TIMEPIX     = 22
FLI         = 23
PIMAX       = 24
ANDOR3D     = 25
JUNGFRAU    = 26
ZYLA        = 27
EpicsPVAM   = 28
EPIX10KA    = 29

det_tuple = (
    (UNDEFINED   , 'UNDEFINED'),
    (CSPAD       , 'Cspad'),
    (CSPAD2X2    , 'Cspad2x2'),
    (PRINCETON   , 'Princeton'),
    (PNCCD       , 'pnCCD'),
    (TM6740      , 'Tm6740'),
    (OPAL1000    , 'Opal1000'),
    (OPAL2000    , 'Opal2000'),
    (OPAL4000    , 'Opal4000'),
    (OPAL8000    , 'Opal8000'),
    (ORCAFL40    , 'OrcaFl40'),
    (EPIX        , 'Epix'),
    (EPIX10K     , 'Epix10k'),
    (EPIX100A    , 'Epix100a'),
    (FCCD960     , 'Fccd960'),
    (ANDOR       , 'Andor'),
    (ACQIRIS     , 'Acqiris'),
    (IMP         , 'Imp'),
    (QUARTZ4A150 , 'Quartz4A150'),
    (RAYONIX     , 'Rayonix'),
    (EVR         , 'Evr'),
    (FCCD        , 'Fccd'),
    (TIMEPIX     , 'Timepix'),
    (FLI         , 'Fli'),
    (PIMAX       , 'Pimax'),
    (ANDOR3D     , 'Andor3d'),
    (JUNGFRAU    , 'Jungfrau'),
    (ZYLA        , 'Zyla'),
    (EpicsPVAM    , 'ControlsCamera'),
    (EPIX10KA    , 'Epix10ka')
)

list_det_types = [rec[0] for rec in det_tuple]
list_det_names = [rec[1] for rec in det_tuple]

dic_det_type_to_name = dict(zip(list_det_types, list_det_names))
dic_det_name_to_type = dict(zip(list_det_names, list_det_types))


if __name__ == "__main__":
  def test_constants():
    print('URL_ENV  : %s' % str(URL_ENV )\
      + '\nURL      : %s' % str(URL     )\
      + '\nURL_KRB  : %s' % str(URL_KRB )\
      + '\nHOST     : %s' % str(HOST    )\
      + '\nPORT     : %s' % str(PORT    )\
      + '\nUSERNAME : %s' % str(USERNAME)\
      + '\nUSERPW   : %s' % str(USERPW  ))

if __name__ == "__main__":

    print('%s\ndic_calib_type_to_name:'%(50*'_'))
    for k,v in dic_calib_type_to_name.items(): print('%16s : %s' % (str(k), str(v)))

    print('%s\ndic_calib_type_to_name:'%(50*'_'))
    for k,v in dic_calib_name_to_type.items(): print('%16s : %s' % (str(k), str(v)))

    print('%s\ndic_det_type_to_name:'%(50*'_'))
    for k,v in dic_det_type_to_name.items(): print('%16s : %s' % (str(k), str(v)))

    print('%s\ndic_det_name_to_type:'%(50*'_'))
    for k,v in dic_det_name_to_type.items(): print('%16s : %s' % (str(k), str(v)))

    print('%s\n'%(50*'_'))
    test_constants()

# EOF
