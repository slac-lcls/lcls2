# Polymorphic factory methods.
from __future__ import generators
import re
import numpy as np

def Detector(name, config):
    det = getattr(config,name)
    df = eval(str(det.dettype) + '._Factory()')
    return df.create(name, config)

class DetectorBase(object):
    def __init__(self, name, config):
        self.detectorName = name
        self.config = config

    def __searchAttr__(self, softwareName):
        self.dataAttr = []
        def children(grandparent, parent):
            tree.append(parent)
            _parent = getattr(grandparent, parent)
            try:
                if "software" in vars(_parent) and "version" in vars(_parent):
                    if softwareName == getattr(_parent, "software"):
                        self.dataAttr.append('.'.join(tree))

                for i, child in enumerate(vars(_parent)):
                    children(_parent, child)
                tree.pop()
            except:
                tree.pop()
                pass

        tree = []
        # this should only look at self.detectorName
        for detname in vars(self.config):
            children(self.config, detname)

    def __sorted_nicely__(self, l):
        """ Sort the given iterable in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def name(self): return self.detectorName

class cspad(DetectorBase):
    """
    cspad reader
    """
    def __init__(self, name, config):
        super(cspad, self).__init__(name, config)
        self.softwareName = "cspadseg"
        self.__searchAttr__(self.softwareName)

    class _Factory:
        def create(self, name, config): return cspad(name, config)

    def raw(self, evt, verbose=0):
        evtStr = 'evt.' + self.dataAttr[0]
        evtAttr = eval(evtStr)  # evt.cspad0.raw.arrayRaw
        evtAttr = evtAttr.reshape((2,3,3)) # This mini cspad has 2x3x3 pixels
        return evtAttr

    def calib(self, data, verbose=0): print("cspad.calib")
    def image(self, data, verbose=0): print("cspad.image")


class Hsd(DetectorBase):
    """
    High Speed Digitizer (HSD) reader
    """
    def __init__(self, name, config):
        super(Hsd, self).__init__(name, config)
        self.softwareName = "hsd"
        self.__searchAttr__(self.softwareName)

    class _Factory:
        def create(self, name, config): return Hsd(name, config)

    def __adcVal__(self, pulseId,_adc,_pid):
        dclks = (pulseId - _pid)*1348
        adc = (_adc + dclks) & 2047
        #print("pulseId,_pid,_adc,dclks,adc: ", pulseId, _pid, _adc, dclks, adc)
        return adc

    def __adcNext__(self, adc):
        return (adc+1) & 2047

    def raw(self, evt):
        mychan = []
        [mychan.append('evt.'+i) for i in self.dataAttr]
        mychan = self.__sorted_nicely__(set(mychan))
        # Return all channels
        arr = None
        for chan in mychan:
            if arr is None:
                arr = self.__rawChannel__(eval(chan))
            else:
                arr = np.vstack((arr, self.__rawChannel__(eval(chan))))
        return arr

    def __rawChannel__(self, data, verbose=0):
        unsignedSize = 4
        if verbose: print("---event header---")
        eventHeader = data[:32] # bytes
        pulseId = int.from_bytes(eventHeader[0:4], byteorder='little', signed=False)
        if verbose: print("pulseId [%u]" % pulseId)
        wordCount = int.from_bytes(eventHeader[20:24], byteorder='little', signed=False)
        if verbose: print("wordCount [%u]" % wordCount)
        eh_samples = int.from_bytes(eventHeader[24:28], byteorder='little', signed=False)
        eh_samples = eh_samples & 262143
        if verbose: print("eh_samples [%u]" % eh_samples)
        sync = int.from_bytes(eventHeader[28:32], byteorder='little', signed=False)
        sync = sync & 7
        if verbose: print("sync [%u]" % sync)

        if verbose: print("---stream header---")
        streamHeader = data[32:32+4*unsignedSize]  # bytes
        samples = int.from_bytes(streamHeader[0:4], byteorder='little', signed=False)
        samples = samples & 2147483647
        if verbose: print("samples [%04u]" % samples)
        boffs = int.from_bytes(streamHeader[4:8], byteorder='little', signed=False)
        boffs = boffs >> 0 & 255
        if verbose: print("boffs [%u]" % boffs)
        eoffs = int.from_bytes(streamHeader[4:8], byteorder='little', signed=False)
        eoffs = eoffs >> 8 & 255
        if verbose: print("eoffs [%u]" % eoffs)
        buffer = int.from_bytes(streamHeader[4:8], byteorder='little', signed=False)
        buffer = buffer >> 16
        if verbose: print("buffer [%u]" % buffer)
        toffs = int.from_bytes(streamHeader[8:12], byteorder='little', signed=False)
        if verbose: print("toffs [%04u]" % toffs)
        baddr = int.from_bytes(streamHeader[12:16], byteorder='little', signed=False)
        baddr = baddr & 65535
        if verbose: print("baddr [%04x]" % baddr)
        eaddr = int.from_bytes(streamHeader[12:16], byteorder='little', signed=False)
        eaddr = eaddr >> 16
        if verbose: print("eaddr [%04x]" % eaddr)

        if verbose: print("---raw---")
        end = samples - 8 + eoffs
        if verbose: print(end)
        rawStart = 48
        rawSize = end*2
        raw = data[rawStart:rawStart+rawSize]  # bytes
        for i in np.arange(0,16,2):
            rawChunk = int.from_bytes(raw[i:i+2], byteorder='little', signed=False)
            if verbose: print("rawChunk: %04x" % rawChunk)

        if verbose: print("---raw stream validate---")
        _pid = pulseId # only set once
        _adc = int.from_bytes(raw[boffs*2:boffs*2+2], byteorder='little', signed=False) # only set once
        if verbose: print("_pid _adc: %u %u" % (_pid, _adc))
        adc = self.__adcVal__(pulseId, _adc, _pid)
        if verbose: print("expected adc: %u %x" % (adc, adc))
        i = boffs
        p = int.from_bytes(raw[i*2:i*2+2], byteorder='little', signed=False)
        if verbose: print("adc: %u %x" % (p, p))
        if verbose: print("i p adc: %d %x %x" %(i, p, adc))
        myRaw = np.zeros((end-i,))
        counter = 0
        myRaw[counter] = adc
        counter += 1
        # update adc
        adc = self.__adcNext__(p)
        i += 1
        arr = np.empty((end-i+1,))
        lenP = 0
        p = int.from_bytes(raw[i * 2:i * 2 + 2], byteorder='little', signed=False)
        arr[lenP] = p
        lenP += 1
        if verbose: print("expected adc: %u %x" % (adc, adc))
        if verbose: print("adc: %u %x" % (p, p))
        ntest = 0
        nerror = 0
        ncorrect = 0
        printMax = 20

        while(i<end):
            ntest += 1
            p = int.from_bytes(raw[i * 2:i * 2 + 2], byteorder='little', signed=False)
            arr[lenP] = p
            lenP += 1
            if adc == p:
                ncorrect += 1
                if verbose:
                    if ncorrect < printMax: print("CORRECT: Match at index %d : adc [%x] expected [%x]" % (i, p, adc))
            else:
                nerror += 1
                if verbose:
                    if nerror < printMax: print("ERROR: Mismatch at index %d : adc [%x] expected [%x]" % (i, p, adc))
            myRaw[counter] = adc
            adc = self.__adcNext__(p)
            i+=1
            counter+=1

        if verbose: print("RawStream::validate %u/%u errors" %(nerror,ntest))

        if verbose: print("---stream header---")
        off = rawStart + samples*2
        streamHeader_fex = data[off:off+4*unsignedSize] # bytes
        for i in range(4):
            a = int.from_bytes(data[off+i*4:off+(i+1)*4], byteorder='little', signed=False)
            if verbose: print("%d %d %08x"%(off+i*4,off+(i+1)*4,a))

        samples_fex = int.from_bytes(streamHeader_fex[0:4], byteorder='little', signed=False)
        samples_fex = samples_fex & 2147483647
        if verbose: print("samples [%04u]" % samples_fex)
        boffs_fex = int.from_bytes(streamHeader_fex[4:8], byteorder='little', signed=False)
        boffs_fex = boffs_fex >> 0 & 255
        if verbose: print("boffs [%u]" % boffs_fex)
        eoffs_fex = int.from_bytes(streamHeader_fex[4:8], byteorder='little', signed=False)
        eoffs_fex = eoffs_fex >> 8 & 255
        if verbose: print("eoffs [%u]" % eoffs_fex)
        buffer_fex = int.from_bytes(streamHeader_fex[4:8], byteorder='little', signed=False)
        buffer_fex = buffer_fex >> 16
        if verbose: print("buffer [%u]" % buffer_fex)
        toffs_fex = int.from_bytes(streamHeader_fex[8:12], byteorder='little', signed=False)
        if verbose: print("toffs [%04u]" % toffs_fex)
        baddr_fex = int.from_bytes(streamHeader_fex[12:16], byteorder='little', signed=False)
        baddr_fex = baddr_fex & 65535
        if verbose: print("baddr [%04x]" % baddr_fex)
        eaddr_fex = int.from_bytes(streamHeader_fex[12:16], byteorder='little', signed=False)
        eaddr_fex = eaddr_fex >> 16
        if verbose: print("eaddr [%04x]" % eaddr_fex)

        for i in np.arange(0, 16, 2):
            fex = data[off + 4 * unsignedSize+i:off + 4 * unsignedSize+i+2]  # bytes
            if verbose: print("%04x" % int.from_bytes(fex, byteorder='little', signed=False))

        # Validate feature extraction
        end = samples_fex-8+eoffs_fex-boffs_fex
        end_j = samples-8+eoffs-boffs
        if verbose: print("fex %u %u %u %u" %(samples_fex, eoffs_fex, boffs_fex, samples_fex-8+eoffs_fex-boffs_fex))
        if verbose: print("raw %u %u %u %u" % (samples, eoffs, boffs, samples-8+eoffs-boffs))

        end_fex = samples_fex - 8 + eoffs_fex
        if verbose: print(end_fex)
        fexStart = off+4*unsignedSize
        fexSize = end_fex * 2
        fex = data[fexStart:fexStart + fexSize]  # bytes
        if verbose: print("final: ",fexStart + fexSize)
        p_thr = int.from_bytes(fex[boffs_fex * 2:boffs_fex * 2 + 2], byteorder='little', signed=False)
        p_raw = int.from_bytes(raw[boffs * 2:boffs * 2 + 2], byteorder='little', signed=False)
        if verbose: print("p_thr p_raw %u %u" %(p_thr,p_raw))
        i = j = 0
        ntest = nerror = ncorrect = 0
        myFex = []
        if (p_thr & 32768):
            i += 1
            j += 1
            if verbose: print("inc i and j")
        while(i < end and j < end_j):
            p_thr = int.from_bytes(fex[(boffs_fex+i) * 2:(boffs_fex+i) * 2 + 2], byteorder='little', signed=False)
            p_raw = int.from_bytes(raw[(boffs + j) * 2:(boffs + j) * 2 + 2], byteorder='little', signed=False)
            if (p_thr & 32768):
                if verbose: print("skipping j from %d to %d %d"%(j, j+(p_thr&32767), p_thr))
                j += p_thr&32767
            else:
                ntest += 1
                if (p_thr == p_raw):
                    ncorrect += 1
                    if verbose:
                        if ncorrect < printMax: print("CORRECT: Match at index thr[%d], raw[%d] : adc thr[%x] raw[%x]" % (i, j, p_thr, p_raw))
                    myFex.append(p_thr)
                else:
                    nerror += 1
                    if verbose:
                        if nerror < printMax: print("ERROR: Mismatch at index thr[%d], raw[%d] : adc thr[%x] raw[%x]" % (i, j, p_thr, p_raw))

                j += 1
            i += 1
        if verbose: print("offset: ",fexStart + fexSize)
        if verbose: print("ThrStream::validate %u/%u errors" % (nerror, ntest))
        return arr
