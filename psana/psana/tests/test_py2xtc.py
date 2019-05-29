import dgramCreate as dc
import numpy as np
import os

def test_py2xtc(tmp_path):

    config = {}
    detname = 'spi_cspad'
    dettype = 'cspad'
    serial_number = '1234'
    namesid = 0

    nameinfo = dc.nameinfo(detname,dettype,serial_number,namesid)
    alg = dc.alg('raw',[1,2,3])

    cydgram = dc.CyDgram()

    image_array = np.array([[1,2,3,4],[9,8,7,6]])
    orientations_array = np.array([4,3,2,1])

    fname = os.path.join(tmp_path,'junk.xtc2')

    f = open(fname,'wb')
    for i in range(3):
        my_data = {
            'image': image_array+i,
            'orientations': orientations_array+i
        }

        cydgram.addDet(nameinfo, alg, my_data)
        timestamp = 0
        pulseid = 0
        transitionid = 0
        xtc_bytes = cydgram.get(timestamp,pulseid,transitionid)
        f.write(xtc_bytes)
    f.close()

    from psana import DataSource
    ds = DataSource(files=fname)
    myrun = next(ds.runs())
    for nevt,evt in enumerate(myrun.events()):
        assert np.array_equal(evt._dgrams[0].spi_cspad[0].raw.image,image_array+nevt+1)
        assert np.array_equal(evt._dgrams[0].spi_cspad[0].raw.orientations,orientations_array+nevt+1)
