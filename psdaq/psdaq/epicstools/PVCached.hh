/**
 * @file
 * @brief PVCached, an EpicsPVA that does not send a value again if it has not changed.
 */
#ifndef Pds_PVCached_hh
#define Pds_PVCached_hh

#include "psdaq/epicstools/EpicsPVA.hh"

namespace Pds_Epics {
  /** EpicsPVA that caches the last values put, so that unchanged values are not sent again. */
  class PVCached : public Pds_Epics::EpicsPVA {
  public:
    /** Connect to name for a scalar PV; the cache holds 0 and the first putC() always sends. */
    PVCached( const char* name ) : 
      Pds_Epics::EpicsPVA(name), _changed(true), _cache(1)
    { _cache[0]=0; }
    /** Connect to name for an array PV of nelem elements, all cached as 0; the first push() always sends. */
    PVCached( const char* name, unsigned nelem ) : 
      Pds_Epics::EpicsPVA(name, nelem), _changed(true), _cache(nelem) 
    { for(unsigned i=0; i<nelem; i++) _cache[i]=0; }
  public:
    /** Put v (a blocking put) if it differs from the cached value or on the first call, and cache it. */
    void putC(double v) { 
      if (v!=_cache[0] || _changed) {
        _changed = false;
        putFrom<double>(_cache[0]=v); 
      }
    }
    /** Store v in element i of the cache and mark the array as changed if it differs; nothing is sent until push(). */
    void putC(double v, unsigned i) {
      if (v!=_cache[i]) {
        _cache[i]=v;
        _changed=true;
      }
    }
    /** If the array is marked as changed, put a copy of the cached array (a blocking put) and clear the mark. */
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
