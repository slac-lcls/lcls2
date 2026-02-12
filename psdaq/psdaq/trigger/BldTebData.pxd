from libc.stdint cimport uint64_t, uint32_t, uint8_t
cimport GmdTebData           as _gmd
cimport XGmdTebData          as _xgmd
cimport PhaseCavityTebData   as _pcav
cimport EBeamTebData         as _ebeam
cimport GasDetTebData        as _gasdet

cdef extern from 'psdaq/trigger/BldTebData.hh' namespace "Pds::Trg":
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
