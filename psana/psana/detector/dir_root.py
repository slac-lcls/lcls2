"""define DIR_ROOT for repositories and logfiles through the environment variable:
   /reg/g/psdm  # lcls
   /cds/group/psdm  # lcls2
   /sdf/group/psdm/etc  # s3df
"""
import os

HOSTNAME = os.getenv('HOSTNAME', None)  # ex: pslogin02
if HOSTNAME is None:
    import socket
    HOSTNAME = socket.gethostname()
print('TEST dir_root.HOSTNAME %s' % HOSTNAME)

DIR_ROOT = os.getenv('DIR_PSDM')  # /cds/group/psdm
DIR_LOG_AT_START = os.path.join(DIR_ROOT, 'logs/atstart/')  # /cds/group/psdm/logs/atstart
CALIB_REPO_EPIX10KA = os.path.join(DIR_ROOT, 'detector/gains2/epix10ka/panels')

#DIR_REPO = os.path.join(DIR_ROOT, 'detector/calib/constants/')
#DIR_ROOT_DATA = '/reg/d/psdm'  # dcs

# EOF
