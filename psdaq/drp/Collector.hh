#ifndef COLLECTOR_H
#define COLLECTOR_H

#include "drp.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/EbLfServer.hh"

class TheSrc : public XtcData::Src
{
public:
    TheSrc(XtcData::Level::Type level, unsigned id) :
        XtcData::Src(level)
    {
        _log |= id;
    }
};

#pragma pack(push,4)
class MyDgram : public XtcData::Dgram {
public:
    MyDgram(XtcData::Sequence& sequence, uint64_t val, unsigned contributor_id);
private:
    uint64_t _data;
};
#pragma pack(pop)

class MyBatchManager : public Pds::Eb::BatchManager
{
public:
    MyBatchManager(Pds::Eb::EbLfClient& ebFtClient, unsigned contributor_id);
    void post(const Pds::Eb::Batch* batch);
    std::atomic<int> inflight_count;
private:
    Pds::Eb::EbLfClient& _ebLfClient;
    unsigned _contributor_id;
};

void collector(MemPool& pool, Parameters& para, MyBatchManager& myBatchMan);
void eb_receiver(MyBatchManager& myBatchMan, MemPool& pool, Parameters& para);

#endif // COLLECTOR_H
