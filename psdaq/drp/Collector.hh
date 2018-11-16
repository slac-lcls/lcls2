#ifndef COLLECTOR_H
#define COLLECTOR_H

#include "drp.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/eb/EbContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"
#include "psdaq/eb/MonContributor.hh"

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

class EbReceiver : public Pds::Eb::EbCtrbInBase
{
public:
    EbReceiver(const Parameters& para, MemPool& pool, Pds::Eb::MonContributor* mon);
    virtual ~EbReceiver() {};
    virtual void process(const XtcData::Dgram* result, const void* input);
private:
    MemPool& _pool;
    FILE* _xtcFile;
    Pds::Eb::MonContributor* _mon;
    unsigned nreceive;
};

void collector(MemPool& pool, Parameters& para, Pds::Eb::EbContributor&, Pds::Eb::MonContributor*);

#endif // COLLECTOR_H
