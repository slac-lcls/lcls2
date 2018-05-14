from psana import dgram
import os

class TestDgramInit:
    emptyResults = (0, 0, 0x4000000)

    def testCreateEmptyDgram(self):
        d = dgram.Dgram()
        results = (d._offset, d._file_descriptor, memoryview(d).shape[0])
        assert self.emptyResults == results

    def testInvalidFileDescriptor(self):
        raised = False
        try:
            d = dgram.Dgram(file_descriptor=42)
        except OSError:
            raised = True
        assert raised

    def testInvalidSequentialRead(self):
        fd = os.open("smd.xtc", os.O_RDONLY)
        d = dgram.Dgram(file_descriptor=fd)
        raised = False
        try: 
            another_d = dgram.Dgram(file_descriptor=fd)
        except StopIteration:
            raised = True
        assert raised

def run():
    test = TestDgramInit()
    test.testCreateEmptyDgram()
    test.testInvalidFileDescriptor()
    test.testInvalidSequentialRead()
