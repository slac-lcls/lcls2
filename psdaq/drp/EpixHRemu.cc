#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "EpixHRemu.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

class RawDef : public VarDef
{
public:
    enum index
    {
        raw
    };

    RawDef()
    {
        Alg raw("raw", 1, 0, 0);
        NameVec.push_back({"raw", Name::UINT16, 2, raw});
    }
};

class XtcCopyIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    XtcCopyIter(unsigned numWords, void*& buffer) :
        XtcIterator(),
        _numWords(numWords),
        _buffer(buffer)
    {
    }

    bool get_value(int i, Name& name, DescData& descdata)
    {
        int data_rank = name.rank();
        int data_type = name.type();
        logging::debug("%d: '%s' rank %d, type %d\n", i, name.name(), data_rank, data_type);

        std::string dscName(name.name());
        if ((dscName != "raw") || (data_type != Name::UINT16) || (data_rank != 2)) {
            return false;
        }

        unsigned maxWords = 1;
        char buffer[128];
        int n = 0;
        n += snprintf(&buffer[n], sizeof(buffer) - n, "'%s' ", name.name());
        n += snprintf(&buffer[n], sizeof(buffer) - n, "(shape:");
        auto shape = descdata.shape(name);
        for (unsigned w = 0; w < name.rank(); w++) {
            n += snprintf(&buffer[n], sizeof(buffer) - n, " %d",shape[w]);
            maxWords *= shape[w];
        }
        n += snprintf(&buffer[n], sizeof(buffer) - n, "):");
        unsigned numWords = _numWords;
        if (maxWords < numWords)
            numWords = maxWords;

        if (_buffer) {
            for (unsigned w = 0; w < 5; ++w) {
                n += snprintf(&buffer[n], sizeof(buffer) - n, " %04x", descdata.get_array<uint16_t>(i).data()[w]);
            }
            n += snprintf(&buffer[n], sizeof(buffer) - n, "\n");
            logging::debug("%s", buffer);
            memcpy(_buffer, descdata.get_array<uint16_t>(i).data(), numWords * sizeof(uint16_t));
        }

        return true;
    }

    int process(Xtc* xtc, const void* bufEnd)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc, bufEnd);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
            logging::debug("DetName: %s, Segment %d, DetType: %s, DetId: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
                           names.detName(), names.segment(), names.detType(), names.detId(),
                           alg.name(), alg.version(), (int)names.namesId());

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                logging::debug("Name: '%s' Type: %d Rank: %d\n",name.name(),name.type(), name.rank());
            }

            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            if (_namesLookup.count(namesId)<=0) {
                logging::critical("Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(unsigned)namesId);
                throw "invalid namesid";
                break;
            }
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            logging::debug("Found %d names for namesid 0x%x\n",names.num(),(unsigned)namesId);
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                if (get_value(i, name, descdata))  break;
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    NamesLookup _namesLookup;
    unsigned _numWords;
    void*& _buffer;
};


EpixHRemu::EpixHRemu(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool)
{
    // Cobble up a serial number
    const char* const serNo = "00deadbeef-0000000000-0000000000-0000000000-0000000000-0000000000-0000000000";
    para->serNo = serNo;

    if (para->kwargs.find("xtcfile") != para->kwargs.end()) {
        int fd = open(para->kwargs["xtcfile"].c_str(), O_RDONLY);
        if (fd < 0) {
            logging::critical("Unable to open file '%s': %m\n", para->kwargs["xtcfile"].c_str());
            abort();
        }

        // Use a different set of events for each segment
        // starting with the l1aOffset'th L1Accept found in the file
        unsigned nSkipL1A = __builtin_popcount(para->laneMask) * para->detSegment;
        if (para->kwargs.find("l1aOffset") != para->kwargs.end()) {
            nSkipL1A += std::stoul(para->kwargs["l1aOffset"]);
        }
        XtcFileIterator iter(fd, 0x4000000);
        void* copyBuffer = nullptr;
        XtcCopyIter xtcIter(numElems, copyBuffer);
        unsigned nevent = 0;
        for (unsigned i=0; i<PGP_MAX_LANES; i++) {
            if (para->laneMask & (1 << i)) {
                m_rawBuffer[i].resize(numElems * sizeof(uint16_t));
                Dgram* dg = iter.next();
                const void* bufEnd = ((char*)dg) + 0x4000000;
                while (dg) {
                    nevent++;
                    logging::debug("event %d, %11s transition: time 0x%8.8x.0x%8.8x, env 0x%08x, "
                                   "payloadSize %d damage 0x%x extent %d\n",
                                   nevent,
                                   TransitionId::name(dg->service()), dg->time.seconds(),
                                   dg->time.nanoseconds(),
                                   dg->env, dg->xtc.sizeofPayload(),dg->xtc.damage.value(),dg->xtc.extent);
                    if (dg->service() == TransitionId::L1Accept && dg->xtc.damage.value() == 0) {
                        if (nSkipL1A == 0) {
                            copyBuffer = m_rawBuffer[i].data();
                            xtcIter.iterate(&(dg->xtc), bufEnd);
                            break;
                        }
                        --nSkipL1A;
                    }
                    copyBuffer = nullptr;
                    xtcIter.iterate(&(dg->xtc), bufEnd);
                    dg = iter.next();
                }
            }
        }
        ::close(fd);
    }
}

unsigned EpixHRemu::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
    logging::info("EpixHRemu configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    Alg rawAlg("raw", 0, 0, 1);
    NamesId rawNamesId(nodeId,RawNamesIndex);
    Names& rawNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), rawAlg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId, m_para->detSegment);
    RawDef myRawDef;
    rawNames.add(xtc, bufEnd, myRawDef);
    m_namesLookup[rawNamesId] = NameIndex(rawNames);

    return 0;
}

unsigned EpixHRemu::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const json& runInfo)
{
    logging::info("EpixHRemu beginrun");
    return 0;
}

void EpixHRemu::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event)
{
    // raw data
    NamesId rawNamesId(nodeId,RawNamesIndex);
    DescribedData raw(dgram.xtc, bufEnd, m_namesLookup, rawNamesId);
    unsigned size = 0;

    //// Replace the dgram data with a whole EpixHR2x2 event which is organized as so:
    ////
    ////    A1   |   A3       (A1,A3) rotated 180deg
    //// --------+--------
    ////    A0   |   A2
    ////
    //unsigned nlanes = 0;
    //size_t dataSize = numElems * sizeof(uint16_t);
    //for (int i=0; i<PGP_MAX_LANES; i++) {
    //    if (event->mask & (1 << i)) {
    //        uint8_t* rawdata = m_rawBuffer[i].data();
    //        memcpy((uint8_t*)raw.data() + size, rawdata, dataSize);
    //        size += dataSize;
    //        nlanes++;
    //    }
    //}

    // Replace the dgram data by copying the ASIC 0 data
    // into the positions for all 4 ASICs
    unsigned nlanes;
    size_t dataSize = elemRowSize * sizeof(uint16_t);
    for (unsigned row = 0; row < elemRows; ++row) {
        nlanes = 0;
        for (int i=0; i<PGP_MAX_LANES; i++) {
            if (event->mask & (1 << i)) {
                uint8_t* rawdata = m_rawBuffer[i].data();
                for (unsigned asic = 0; asic < numAsics; ++asic) {
                    memcpy((uint8_t*)raw.data() + size, rawdata + row * dataSize * 2, dataSize);
                    size += dataSize;
                }
                nlanes++;
            }
        }
    }

    raw.set_data_length(size);
    //unsigned raw_shape[MaxRank] = {nlanes * 2 * elemRows, numAsics * elemRowSize / 2}; // 2x2 ASIC data
    unsigned raw_shape[MaxRank] = {elemRows, nlanes * numAsics * elemRowSize};         // 1x4 ASIC data
    raw.set_array_shape(RawDef::raw, raw_shape);
}

}
