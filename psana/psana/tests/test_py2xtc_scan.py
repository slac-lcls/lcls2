import dgramCreate as dc
import numpy as np
import os

# _transitionId is a subset of the TransitionId.hh enum
_transitionId = {
    'ClearReadout'      : 0,
    'Reset'             : 1,
    'Configure'         : 2,
    'Unconfigure'       : 3,
    'BeginRun'          : 4,
    'EndRun'            : 5,
    'BeginStep'         : 6,
    'EndStep'           : 7,
    'Enable'            : 8,
    'Disable'           : 9,
    'SlowUpdate'        : 10,
    'L1Accept'          : 12,
}

def test_py2xtc_step(tmp_path):

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

    runinfo_detname = 'runinfo'
    runinfo_dettype = 'runinfo'
    runinfo_detid = ''
    runinfo_namesid = 1
    runinfo_nameinfo = dc.nameinfo(runinfo_detname,runinfo_dettype,
                                   runinfo_detid,runinfo_namesid)
    runinfo_alg = dc.alg('runinfo',[0,0,1])
    runinfo_data = {
        'expt': 'xpptut15',
        'runnum': 14
    }

    fname = os.path.join(tmp_path,'junk.xtc2')

    f = open(fname,'wb')
    for i in range(4):
        my_data = {
            'image': image_array+i,
            'orientations': orientations_array+i
        }

        cydgram.addDet(nameinfo, alg, my_data)
        # only do this for the first two dgrams: name info for config, and
        # the runinfo data for beginrun
        if i<2: cydgram.addDet(runinfo_nameinfo, runinfo_alg, runinfo_data)
        timestamp = i
        if (i==0):
            transitionid = _transitionId['Configure']
        elif (i==1):
            transitionid = _transitionId['BeginRun']
        else:
            transitionid = _transitionId['L1Accept']
        xtc_bytes = cydgram.get(timestamp,transitionid)
        f.write(xtc_bytes)
    f.close()

    from psana import DataSource
    ds = DataSource(files=fname)
    myrun = next(ds.runs())
    assert myrun.expt==runinfo_data['expt']
    assert myrun.runnum==runinfo_data['runnum']
    for nevt,evt in enumerate(myrun.events()):
        assert np.array_equal(evt._dgrams[0].spi_cspad[0].raw.image,image_array+nevt+2)
        assert np.array_equal(evt._dgrams[0].spi_cspad[0].raw.orientations,orientations_array+nevt+2)
    assert nevt>0 #make sure we get events

if __name__ == "__main__":
    import pathlib
    test_py2xtc_step(pathlib.Path('.'))
