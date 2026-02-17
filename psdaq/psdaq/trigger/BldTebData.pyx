from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t, uint8_t
cimport psdaq.trigger.GmdTebData         as _GmdTebData
cimport psdaq.trigger.XGmdTebData        as _XGmdTebData
cimport psdaq.trigger.PhaseCavityTebData as _PhaseCavityTebData
cimport psdaq.trigger.EBeamTebData       as _EBeamTebData
cimport psdaq.trigger.GasDetTebData      as _GasDetTebData
cimport psdaq.trigger.BldTebData         as _BldTebData

cdef class GmdTebData:

     cdef _GmdTebData.GmdTebData* cptr

     def __cinit__(self):
         pass

     def milliJoulesPerPulse(self):
         return self.cptr.milliJoulesPerPulse

cdef class XGmdTebData:

     cdef _XGmdTebData.XGmdTebData* cptr

     def __cinit__(self):
         pass

     def milliJoulesPerPulse(self):
         return self.cptr.milliJoulesPerPulse

     def POSY(self):
         return self.cptr.POSY

cdef class PhaseCavityTebData:

     cdef _PhaseCavityTebData.PhaseCavityTebData* cptr

     def __cinit__(self):
         pass

     def fitTime1(self):
         return self.cptr.fitTime1

     def fitTime2(self):
         return self.cptr.fitTime2

     def charge1(self):
         return self.cptr.charge1

     def charge2(self):
         return self.cptr.charge2

cdef class EBeamTebData:

     cdef _EBeamTebData.EBeamTebData* cptr

     def __cinit__(self):
         pass

     def l3Energy(self):
         return self.cptr.l3Energy

cdef class GasDetTebData:

     cdef _GasDetTebData.GasDetTebData* cptr

     def __cinit__(self):
         pass

     def f11ENRC(self):
         return self.cptr.f11ENRC

     def f12ENRC(self):
         return self.cptr.f12ENRC

     def f21ENRC(self):
         return self.cptr.f21ENRC

     def f22ENRC(self):
         return self.cptr.f22ENRC

     def f63ENRC(self):
         return self.cptr.f63ENRC

     def f64ENRC(self):
         return self.cptr.f64ENRC

cdef class BldTebData:
    cdef Py_buffer buf
    cdef _BldTebData.BldTebData* cptr
    cdef int _bufOwner
    cdef _GmdTebData.GmdTebData*                 gmdp
    cdef _XGmdTebData.XGmdTebData*               xgmdp
    cdef _PhaseCavityTebData.PhaseCavityTebData* pcavp
    cdef _PhaseCavityTebData.PhaseCavityTebData* pcavsp
    cdef _EBeamTebData.EBeamTebData*             ebeamp
    cdef _EBeamTebData.EBeamTebData*             ebeamsp
    cdef _GasDetTebData.GasDetTebData*           gasdetp

    def __cinit__(self, view = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            if PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)<0:
               print('Error extracting BldTebData buffer')
            self.cptr = <_BldTebData.BldTebData *>(self.buf.buf)
            self.gmdp     = self.cptr.gmd()
            self.xgmdp    = self.cptr.xgmd()
            self.pcavp    = self.cptr.pcav()
            self.pcavsp   = self.cptr.pcavs()
            self.ebeamp   = self.cptr.ebeam()
            self.ebeamsp  = self.cptr.ebeams()
            self.gasdetp  = self.cptr.gasdet()

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def gmd(self):
        if self.gmdp:
           g = GmdTebData()
           g.cptr = self.gmdp
           return g
        return None

    def xgmd(self):
        if self.xgmdp:
           g = XGmdTebData()
           g.cptr = self.xgmdp
           return g
        return None

    def pcav(self):
        if self.pcavp:
           g = PhaseCavityTebData()
           g.cptr = self.pcavp
           return g
        return None

    def pcavs(self):
        if self.pcavsp:
           g = PhaseCavityTebData()
           g.cptr = self.pcavsp
           return g
        return None

    def ebeam(self):
        if self.ebeamp:
           g = EBeamTebData()
           g.cptr = self.ebeamp
           return g
        return None

    def ebeams(self):
        if self.ebeamsp:
           g = EBeamTebData()
           g.cptr = self.ebeamsp
           return g
        return None

    def gasdet(self):
        if self.gasdetp:
           g = GasDetTebData()
           g.cptr = self.gasdetp
           return g
        return None
