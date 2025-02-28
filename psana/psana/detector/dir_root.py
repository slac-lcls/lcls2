"""
   Usage:
   from psana.detector.dir_root import DIR_ROOT, DIR_REPO

   DIR_ROOT:
   /reg/g/psdm             # lcls
   /cds/group/psdm         # lcls2
   /sdf/group/lcls/ds/ana/ # s3df
"""
import os

HOSTNAME = os.getenv('HOSTNAME', None)  # ex: pslogin02
if HOSTNAME is None:
    import socket
    HOSTNAME = socket.gethostname()

DIR_ROOT = os.getenv('DIR_PSDM', '/sdf/group/lcls/ds/ana/')  # /sdf/group/lcls/ds/ana/ on s3df OR /cds/group/psdm on pcds
DIR_LOG_AT_START    = os.path.join(DIR_ROOT, 'detector/logs/atstart/')          # /cds/group/psdm/detector/logs/atstart
DIR_REPO            = os.path.join(DIR_ROOT, 'detector/calib2/constants')       # common repository
DIR_REPO_EPIX10KA   = DIR_REPO
DIR_REPO_EPIXM320   = DIR_REPO
DIR_REPO_JUNGFRAU   = DIR_REPO
DIR_REPO_CALIBMAN   = DIR_REPO                                                  # prev: /cds/group/psdm/detector/calib2/constants/logs
DIR_DATA_TEST       = os.path.join(DIR_ROOT, 'detector/data2_test')             # /cds/group/psdm/detector/data2_test/

DIR_DATA = os.getenv('SIT_PSDM_DATA', '/sdf/data/lcls/ds')  # /sdf/data/lcls/ds/
DIR_FFB = os.path.join(DIR_DATA, '../drpsrcf/ffb').replace('/ds/../','/')  # '/sdf/data/lcls/drpsrcf/ffb'
# EOF
