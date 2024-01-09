"""
   see on psana
   /cds/sw/ds/ana/conda1/manage/bin/psconda.sh  # lcls1
   /cds/sw/ds/ana/conda2/manage/bin/psconda.sh  # lcls2
   see on s3df
   /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh # lcls1
   /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh # lcls2

   DIR_ROOT for repositories and logfiles through the environment variable:
   /reg/g/psdm      # lcls
   /cds/group/psdm  # lcls2
   /sdf/group/psdm  # s3df ???

   DIR_PSDM
   /cds/group/psdm # on psana lcls2
   /cds/group/psdm # on sdflogin lcls2

   SIT_PSDM_DATA
   /sdf/data/lcls/ds/

   ffb data
   /sdf/data/lcls/drpsrcf/ffb/

from psana.detector.dir_root import DIR_ROOT, DIR_REPO
"""
import os

HOSTNAME = os.getenv('HOSTNAME', None)  # ex: pslogin02
if HOSTNAME is None:
    import socket
    HOSTNAME = socket.gethostname()
#print('TEST dir_root.HOSTNAME %s' % HOSTNAME)

DIR_ROOT = os.getenv('DIR_PSDM')  # /cds/group/psdm
DIR_LOG_AT_START    = os.path.join(DIR_ROOT, 'detector/logs/atstart/')          # /cds/group/psdm/detector/logs/atstart
DIR_REPO            = os.path.join(DIR_ROOT, 'detector/calib2/constants')       # common repository
DIR_REPO_EPIX10KA   = DIR_REPO
#DIR_REPO_EPIX10KA   = os.path.join(DIR_ROOT, 'detector/gains2/epix10ka/panels') # /cds/group/psdm/detector/gains2/epix10ka/panels
#DIR_REPO_DARK_PROC  = DIR_REPO
#DIR_REPO_DARK_PROC  = os.path.join(DIR_ROOT, 'detector/calib2')                 # /cds/group/psdm/detector/calib2
DIR_DATA_TEST       = os.path.join(DIR_ROOT, 'detector/data2_test')             # /cds/group/psdm/detector/data2_test/
DIR_REPO_CALIBMAN   = DIR_REPO                                                  # prev: /cds/group/psdm/detector/calib2/constants/logs
#DIR_LOG_CALIBMAN    = os.path.join(DIR_ROOT, 'detector/logs/calibman/lcls2')   # /cds/group/psdm/detector/logs/calibman/lcls2

# for s3df
DIR_DATA = os.getenv('SIT_PSDM_DATA', '/sdf/data/lcls/ds')  # /sdf/data/lcls/ds/
DIR_FFB = os.path.join(DIR_DATA, '../drpsrcf/ffb').replace('/ds/../','/')  # '/sdf/data/lcls/drpsrcf/ffb'
# EOF
