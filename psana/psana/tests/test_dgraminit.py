from psana import dgram
import os

class TestDgramInit:

    def testInvalidEmptyDgram(self):
        raised = False
        try:
            d = dgram.Dgram()
        except RuntimeError:
            raised = True
        assert raised

    def testInvalidFileDescriptor(self):
        raised = False
        try:
            d = dgram.Dgram(file_descriptor=42)
        except OSError:
            raised = True
        assert raised

    def testInvalidSequentialRead(self):
        """ prevent reading data dgram without config """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(dir_path, "smd.xtc2")
        if os.path.isfile(full_path):
            fd = os.open(full_path, os.O_RDONLY)
            d = dgram.Dgram(file_descriptor=fd)
            raised = False
            try: 
                another_d = dgram.Dgram(file_descriptor=fd)
            except StopIteration:
                raised = True
            assert raised

def run():
    test = TestDgramInit()
    test.testInvalidEmptyDgram()
    test.testInvalidFileDescriptor()
    test.testInvalidSequentialRead() 

if __name__ == "__main__":
    run()
