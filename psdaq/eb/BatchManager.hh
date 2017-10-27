#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "Batch.hh"

#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/service/SemLock.hh"

#include <stdlib.h>
#include <stdint.h>
#include <cstddef>
#include <string>


namespace XtcData {
  class Dgram;
};

namespace Pds {
  namespace Eb {

    class EbFtBase;

    class BatchManager
    {
    public:
      BatchManager(EbFtBase& outlet,
                   unsigned  src,       // Revisit: Should be a Src?
                   uint64_t  duration,
                   unsigned  batchDepth,
                   unsigned  maxEntries,
                   size_t    contribSize);
      virtual ~BatchManager();
    public:
      virtual void post(Batch*, void* arg) = 0;
    public:
      void         process(const XtcData::Dgram*, void* arg = (void*)0);
      void         postTo(Batch*, unsigned dst, unsigned slot);
      void         release(uint64_t id);
      void         shutdown();
      uint64_t     batchId(uint64_t id) const;
    private:
      void         _post(const Batch&);
      void         _batchInit(unsigned batchDepth, unsigned poolDepth);
    private:
      uint64_t     _startId(uint64_t id) const;
    private:
      unsigned     _src;                // ID of this node
      uint64_t     _duration;           // The lifetime of a batch (power of 2)
      uint64_t     _durationShift;      // Shift away insignificant bits
      uint64_t     _durationMask;       // Mask  off  insignificant bits
      size_t       _maxBatchSize;       // Maximum size of a batch
      GenericPoolW _pool;               // Pool of Batch objects
      BatchList    _inFlightList;       // Listhead of batches in flight
      SemLock      _inFlightLock;       // Lock for _inFlightList
      EbFtBase&    _outlet;             // LibFabric transport
    private:
      Batch*       _batch;              // Batch currently being assembled
    };
  };
};

#endif
