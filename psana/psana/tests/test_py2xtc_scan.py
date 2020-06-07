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

def test_py2xtc_scan(tmp_path):

    config = {}
    detname = 'spi_cspad'
    dettype = 'cspad'
    serial_number = '1234'
    namesid = 0

    nameinfo = dc.nameinfo(detname,dettype,serial_number,namesid)
    alg = dc.alg('raw',[1,2,3])

    cydgram = dc.CyDgram()

    motor1_array = np.array([1.0, 2.0, 3.0])
    motor2_array = np.array([4.0, 5.0, 6.0])

    fname = os.path.join(tmp_path,'junk.xtc2')

    my_data = {
        'motor1': motor1_array,
        'motor2': motor2_array
    }
    timestamp = 0
    transitionid = _transitionId['Configure']

    f = open(fname,'wb')

    len_list = []
    for xx in range(4):
        cydgram.addDet(nameinfo, alg, my_data)
        xtc_bytes = cydgram.get(timestamp, transitionid)
        len_list.append(len(xtc_bytes))
        f.write(xtc_bytes)

    for xx in range(2):
        cydgram.addDet(nameinfo, alg, my_data)
        xtc_bytes = cydgram.getSelect(timestamp, transitionid, add_names=True, add_shapes_data=True)
        len_list.append(len(xtc_bytes))
        f.write(xtc_bytes)

    for xx in range(2):
        cydgram.addDet(nameinfo, alg, my_data)
        xtc_bytes = cydgram.getSelect(timestamp, transitionid, add_names=False, add_shapes_data=True)
        len_list.append(len(xtc_bytes))
        f.write(xtc_bytes)

    for xx in range(2):
        cydgram.addDet(nameinfo, alg, my_data)
        xtc_bytes = cydgram.getSelect(timestamp, transitionid, add_names=True, add_shapes_data=False)
        len_list.append(len(xtc_bytes))
        f.write(xtc_bytes)

    cydgram.addDet(nameinfo, alg, my_data)
    xtc_bytes = cydgram.getSelect(timestamp, transitionid, add_names=False, add_shapes_data=False)
    len_list.append(len(xtc_bytes))
    f.write(xtc_bytes)

    print("len_list: %s" % len_list)

    header_size = 24                                    # 24 = dgram (12) + xtc (12)
    assert len_list[0] > len_list[1]
    assert len_list[1] == len_list[2] == len_list[3]
    assert len_list[0] == len_list[4] == len_list[5]
    assert len_list[3] == len_list[6] == len_list[7]
    assert len_list[8] == len_list[9]
    assert len_list[10] == header_size
    assert len_list[8] + len_list[6] - header_size == len_list[0]

    f.close()

if __name__ == "__main__":
    import pathlib
    test_py2xtc_scan(pathlib.Path('.'))
