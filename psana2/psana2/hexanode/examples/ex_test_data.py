
""" Acess to test xtc2 data files:
    from psana2.hexanode.examples.ex-test-data import DIR_DATA_TEST
"""
import os
DIR_ROOT = os.getenv('DIR_PSDM')  # /cds/group/psdm ON psana OR /sdf/group/lcls/ds/ana/ ON s3df
DIR_DATA_TEST = os.path.join(DIR_ROOT, 'detector/data2_test/xtc')
# EOF
