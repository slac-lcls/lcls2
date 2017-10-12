#ifndef Pds_Eb_BatchHandler_hh
#define Pds_Eb_BatchHandler_hh

#include "pdsdata/xtc/XtcIterator.hh"

#include <stdlib.h>
#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>


namespace Pds {

  class Datagram;

  namespace Eb {

    class BatchHandler : public Pds::XtcIterator
    {
    public:
      BatchHandler(std::string& port,
                   unsigned     nSources,
                   unsigned     nSlots,
                   unsigned     nBatches,
                   size_t       maxBatchSize);
      virtual ~BatchHandler();
    public:
      virtual int process(Pds::Xtc* xtc);
      virtual int process(Pds::Datagram* contribution) = 0;
    public:
      Datagram* pend();
      void      release(Datagram*);
    private:
      void      _process(Pds::Datagram* batch);
    private:
      unsigned  _numSources;            // Number of peers
      unsigned  _numBatches;            // Number of batches
      size_t    _maxBatchSize;          // Maximum size of each batch
      char*     _pool;                  // Pool where batches are written
      FtInlet   _inlet;                 // LibFabric transport
    };
  };
};
