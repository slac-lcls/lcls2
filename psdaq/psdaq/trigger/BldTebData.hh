#ifndef Pds_Trg_BldTebData_hh
#define Pds_Trg_BldTebData_hh

#include <string>
#include <cstdint>

namespace Pds {
  namespace Trg {
    class GmdTebData;
    class XGmdTebData;
    class PhaseCavityTebData;
    class EBeamTebData;
    class GasDetTebData;

    class BldTebData
    {
    public:
      enum BldSource { gmd_, xgmd_, pcav_, pcavs_,  gasdet_, ebeam_, ebeams_, NSOURCES };
      static BldSource lookup(const std::string& detName);
      static unsigned sizeof_();

      BldTebData(uint64_t sources_) : sources(sources_) {}

#define DATADEF(N,T)  T* N() \
        { return (sources>>N##_)&1 ? reinterpret_cast<T*>((char*)this+offset_(N##_)) : 0; }

      DATADEF(gmd   ,GmdTebData)
      DATADEF(xgmd  ,XGmdTebData)
      DATADEF(pcav  ,PhaseCavityTebData)
      DATADEF(pcavs ,PhaseCavityTebData)
      DATADEF(gasdet,GasDetTebData)
      DATADEF(ebeam ,EBeamTebData)
      DATADEF(ebeams,EBeamTebData)

      // GmdTebData&         gmd ();
      // XGmdTebData&        xgmd();
      // PhaseCavityTebData& pcav();

      unsigned offset_(unsigned);

      uint64_t sources;
    };
  };
};

#endif
