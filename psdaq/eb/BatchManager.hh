#ifndef Pds_Eb_BatchManager_hh
#define Pds_Eb_BatchManager_hh

#include <stdlib.h>
#include <stdint.h>
#include <cstddef>
#include <string>

#include "Endpoint.hh"
#include "Batch.hh"

#include "psdaq/service/GenericPoolW.hh"

#include "pdsData/xtc/ClockTime.hh"


namespace Pds {

  class Datagram;

  namespace Eb {

// Notational conveniences...     (Revisit: Why aren't these typedefs?)
#define StringList std::vector<std::string>
#define EpList     std::vector<Fabrics::Endpoint*>
#define MrList     std::vector<Fabrics::MemoryRegion*>
#define RaList     std::vector<Fabrics::RemoteAddress>

    class BatchManager
    {
    public:
      BatchManager(StringList&  remote,
                   std::string& port,
                   unsigned     src,       // Revisit: Should be a Src?
                   ClockTime    duration,
                   unsigned     batchDepth,
                   unsigned     iovPoolDepth,
                   size_t       contribSize);
      ~BatchManager();
    public:
      void         process(const Datagram&);
      void         postTo(const Batch*, unsigned dst, unsigned slot);
    private:
      void         _post(const Batch&);
      void         _batchInit(unsigned poolDepth);
      int          _connect(std::string&            remote,
                            std::string&            port,
                            char*                   pool,
                            size_t                  size,
                            Fabrics::Endpoint*&     ep,
                            Fabrics::MemoryRegion*& mr,
                            Fabrics::RemoteAddress& ra);
    private:
      uint64_t     _batchId(ClockTime&);
      uint64_t     _startTime(ClockTime&);
    private:
      EpList       _ep;                 // List of endpoints
      MrList       _mr;                 // List of memory regions
      RaList       _ra;                 // List of remote address descriptors
      unsigned     _src;                // ID of this node
      ClockTime    _duration;           // The lifetime of a batch (power of 2)
      uint64_t     _durationShift;      // Shift away insignificant bits
      uint64_t     _durationMask;       // Mask  off  insignificant bits
      unsigned     _numBatches;         // Maximum possible number of batches
      size_t       _maxBatchSize;       // Maximum size of a batch
      GenericPoolW _batchPool;          // Pool of Batch objects
      BatchList    _batchList;          // Listhead of batches in flight
    private:
      Batch*       _batch;              // Batch currently being assembled
    };
  };
};
