#ifndef COLLECTOR_H
#define COLLECTOR_H

#include "drp.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/EbLfServer.hh"

// these parameters must agree with the server side
unsigned maxBatches = 1024; // size of the pool of batches
unsigned maxEntries = 8; // maximum number of events in a batch
unsigned BatchSizeInPulseIds = 8; // age of the batch. should never exceed maxEntries above, must be a power of 2

unsigned EbId = 0; // from 0-63, maximum number of event builders

class TheSrc : public XtcData::Src
{
public:
    TheSrc(XtcData::Level::Type level, unsigned id) :
        XtcData::Src(level)
    {
        _log |= id;
    }
};

class MyDgram : public XtcData::Dgram {
public:
    MyDgram(unsigned pulseId, uint64_t val, unsigned contributor_id);
private:
    uint64_t _data;
};

size_t maxSize = sizeof(MyDgram);

class MyBatchManager: public Pds::Eb::BatchManager {
public:
    MyBatchManager(Pds::Eb::EbLfClient& ebFtClient, unsigned contributor_id) :
        Pds::Eb::BatchManager(BatchSizeInPulseIds, maxBatches, maxEntries, maxSize),
        _ebLfClient(ebFtClient),
        _contributor_id(contributor_id)
    {}
    void post(const Pds::Eb::Batch* batch) {
      _ebLfClient.post(EbId, batch->datagram(), batch->extent(),
                       batch->index() * maxBatchSize(),
                       (_contributor_id << 24) + batch->index());
    }
private:
    Pds::Eb::EbLfClient& _ebLfClient;
    unsigned _contributor_id;
};

void collector(MemPool& pool, Parameters& para);
void eb_receiver(MyBatchManager& myBatchMan, MemPool& pool, Parameters& para);

#endif // COLLECTOR_H
