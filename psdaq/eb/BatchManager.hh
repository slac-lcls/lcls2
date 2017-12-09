#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include "Batch.hh"

#include "psdaq/service/GenericPoolW.hh"

#include <stdlib.h>
#include <stdint.h>
#include <cstddef>
#include <string>

namespace XtcData {
  class Dgram;
};

namespace Pds {
  namespace Eb {

    class BatchManager
    {
    public:
      BatchManager(uint64_t     duration,
                   unsigned     batchDepth,
                   unsigned     maxEntries,
                   size_t       maxSize);
      virtual ~BatchManager();
    public:
      virtual void post(Batch*) = 0;
    public:
      void*        batchRegion() const;
      size_t       batchRegionSize() const;
      Batch*       allocate(const XtcData::Dgram*);
      void         process(const XtcData::Dgram*);
      const Batch* batch(unsigned index) const;
      void         shutdown();
      uint64_t     batchId(uint64_t id) const;
      size_t       maxBatchSize() const;
    public:
      void         dump() const;
    private:
      void         _post(const Batch&);
      void         _batchInit(unsigned batchDepth, unsigned poolDepth);
    private:
      uint64_t     _startId(uint64_t id) const;
    private:
      uint64_t     _duration;           // The lifetime of a batch (power of 2)
      uint64_t     _durationShift;      // Shift away insignificant bits
      uint64_t     _durationMask;       // Mask  off  insignificant bits
      unsigned     _batchDepth;         // Depth of the batch pool
      unsigned     _maxEntries;         // Max number of entries per batch
      size_t       _maxBatchSize;       // Maximum size of a batch
      char*        _batchBuffer;        // Buffers for batches
      const XtcData::Dgram** _datagrams;// Array for holding on to datagrams
      GenericPoolW _pool;               // Pool of Batch objects
      Batch**      _batches;            // Lookup array of batches
    private:
      Batch*       _batch;              // Batch currently being assembled
    };
  };
};

#endif
