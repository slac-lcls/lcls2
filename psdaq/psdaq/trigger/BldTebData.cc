#include "psdaq/trigger/BldTebData.hh"
#include "psdaq/trigger/GmdTebData.hh"
#include "psdaq/trigger/XGmdTebData.hh"
#include "psdaq/trigger/PhaseCavityTebData.hh"
#include "psdaq/trigger/GasDetTebData.hh"
#include "psdaq/trigger/EBeamTebData.hh"

using namespace Pds::Trg;

static std::string _detNames[] = { "gmd",
                                   "xgmd",
                                   "pcav",
                                   "pcavs",
                                   "gasdet",
                                   "ebeam",
                                   "ebeams", };

static unsigned _sizes[] = { sizeof(GmdTebData), 
                             sizeof(XGmdTebData), 
                             sizeof(PhaseCavityTebData),
                             sizeof(PhaseCavityTebData),
                             sizeof(GasDetTebData),
                             sizeof(EBeamTebData),
                             sizeof(EBeamTebData) };

BldTebData::BldSource BldTebData::lookup(const std::string& detName)
{
    for(unsigned i=0; i<NSOURCES; i++)
        if (detName==_detNames[i])
            return BldSource(i);
    return NSOURCES;
}

unsigned BldTebData::sizeof_() 
{ 
    unsigned offset=sizeof(BldTebData);
    for(unsigned i=0; i<NSOURCES; i++)
        offset += _sizes[i];
    return offset;
}


unsigned BldTebData::offset_(unsigned src)
{
    uint64_t mask = sources;
    unsigned offset=sizeof(*this);
    for(unsigned i=0; i<src; i++)
        if ((mask>>i)&1)
            offset += _sizes[i];
    return offset;
}

/*
const GmdTebData& BldTebData::gmd() const
{
    return *reinterpret_cast<GmdTebData*>((char*)this+_offset(gmd_,sources));
}

const XGmdTebData& BldTebData::xgmd() const
{
    return *reinterpret_cast<XGmdTebData*>((char*)this+_offset(xgmd_,sources));
}
*/
