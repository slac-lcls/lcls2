#!/usr/bin/env python
"""
/reg/d/psdm/AMO/amox27716/results/xiangli/run85_opal_1000.xtc2 copied to:
detnames -i /reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0085-opal1k.xtc2
----------------------------------------
Name    | Det Type | Data Type | Version
----------------------------------------
runinfo | runinfo  | runinfo   | 0_0_1
opal    | ele_opal | raw       | 1_2_3
----------------------------------------

cdb add -e testexper -d opal1000_test -c pop_rbfs -r 50 -f /reg/g/psdm/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.pkl -i pkl -u dubrovin
cdb get -d opal1000_test -c pop_rbfs -r 50 -i json -f my_pop_rbfs
cdb get -d opal1000_test -c pop_rbfs -r 50 -i pkl -f my_pop_rbfs
cdb add -e amox27716 -d ele_opal -c pop_rbfs -r 50 -f /reg/g/psdm/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.pkl -i pkl -u dubrovin
cdb add -e amox27716 -d ele_opal -c pop_rbfs -r 50 -f /reg/g/psdm/detector/calib/misc/calib-amox27716-r50-opal-pop-rbfs-xiangli.json -i json -u dubrovin

Access opal1k camera in xtc2 files
"""

import sys
from psana2.pyalgos.generic.NDArrUtils import print_ndarr
from psana import DataSource
from psana2.hexanode.examples.ex_test_data import DIR_DATA_TEST

FNAME = '%s/%s' % (DIR_DATA_TEST, 'data-amox27716-r0085-opal1k.xtc2')
#FNAME = '/sdf/group/lcls/ds/ana/detector/data2_test/xtc/data-amox27716-r0085-opal1k.xtc2'

print('e.g.: [python] %s [test-number]' % sys.argv[0])

def test_opal_data_access() :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'

    print('DIRECT ACCESS CALIBRATION CONSTANTS')

    from psana2.pscalib.calib.MDBWebUtils import calib_constants
    data, doc = calib_constants('ele_opal_1234', exp='amox27716', ctype='pop_rbfs', run=85)
    print('direct consatnts access meta:\n', doc)
    print('\ndirect consatnts access data:')
    for k,v in data.items() : print_ndarr(v, '%03d : '%k)

    print('DETECTOR INTERFACE ACCESS CALIBRATION CONSTANTS')

    ds = DataSource(files=FNAME)
    orun = next(ds.runs())
    camera = orun.Detector('opal')

    print('test_xtcav_data    expt: %s runnum: %d\n' % (orun.expt, orun.runnum))

    for nev,evt in enumerate(orun.events()):
        if nev>10 : break
        print('Event %03d'%nev, end='')
        ####print_ndarr(camera.raw.array(evt), '  camera.raw.array(evt):')
        #print_ndarr(camera.img(evt), '  camera.img(evt):')
        #print('XXXXX', evt._dgrams[0].xtcav[0].raw.raw)
        #print('XXXXX', dir(evt._dgrams[0].opal[0].raw.img))

        print('***',camera.raw.image(evt))
        break

    calib_data, calib_meta = camera.calibconst.get('pop_rbfs')
    print('camera.calibconst.get   calib_meta', calib_meta)


if __name__ == "__main__":
    test_opal_data_access()

# EOF

