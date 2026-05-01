from libc.stdint cimport uint64_t, uint32_t, uint8_t
cimport psdaq.trigger.GmdTebData           as _gmd
cimport psdaq.trigger.XGmdTebData          as _xgmd
cimport psdaq.trigger.PhaseCavityTebData   as _pcav
cimport psdaq.trigger.EBeamTebData         as _ebeam
cimport psdaq.trigger.GasDetTebData        as _gasdet

cdef extern from 'psdaq/trigger/src/BldTebData.hh' namespace "Pds::Trg":
    cdef cppclass BldTebData:
        BldTebData(uint64_t  sources_) except +
        _gmd.GmdTebData*          gmd()
        _xgmd.XGmdTebData*        xgmd()
        _pcav.PhaseCavityTebData* pcav()
        _pcav.PhaseCavityTebData* pcavs()
        _ebeam.EBeamTebData*      ebeam()
        _ebeam.EBeamTebData*      ebeams()
        _gasdet.GasDetTebData*    gasdet()
        uint32_t offset_(uint32_t)
        uint64_t sources
