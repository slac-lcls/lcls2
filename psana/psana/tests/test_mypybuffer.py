from psana.mypybuffer import MyPyBuffer
import os

def test_run_mypybuffer():
    ifname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', 'dgramedit-test.xtc2')
    fd = os.open(ifname, os.O_RDONLY)
    size = 2643928                          # size of dgrampy-test.xtc2
    view = os.read(fd, size)

    mpb = MyPyBuffer(view)
    for dg in mpb.dgrams():
        print(dg.timestamp())

    os.close(fd)
    

if __name__ == "__main__":
    test_run_mypybuffer()
