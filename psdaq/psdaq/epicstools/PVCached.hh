#ifndef Pds_PVCached_hh
#define Pds_PVCached_hh

#include "psdaq/epicstools/EpicsPVA.hh"

namespace Pds_Epics {
  class PVCached : public Pds_Epics::EpicsPVA {
  public:
    PVCached( const char* name ) : 
      Pds_Epics::EpicsPVA(name), _changed(true), _cache(1) {}
    PVCached( const char* name, unsigned nelem ) : 
      Pds_Epics::EpicsPVA(name, nelem), _changed(true), _cache(nelem) {}
  public:
    void putC(double v) { 
      if (v!=_cache[0] || _changed) {
        _changed = false;
        putFrom<double>(_cache[0]=v); 
      }
    }
    void putC(double v, unsigned i) {
      if (v!=_cache[i]) {
        _cache[i]=v;
        _changed=true;
      }
    }
    void push() {
      if (_changed) {
        _changed=false;
        //  Create a unique copy because put deletes the original
        pvd::shared_vector<double> cache(_cache);
        cache.make_unique();
        putFromVector<double>(freeze(cache));
      }
    }
  private:
    bool    _changed;
    pvd::shared_vector<double> _cache;
  };
};

#endif
