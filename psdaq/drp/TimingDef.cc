#include "TimingDef.hh"

#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

using namespace XtcData;

namespace Drp {

    TimingDef::TimingDef()
    {
        NameVec.push_back({"pulseId"       , Name::UINT64, 0});
        NameVec.push_back({"timeStamp"     , Name::UINT64, 0});
        NameVec.push_back({"fixedRates"    , Name::UINT8 , 1});
        NameVec.push_back({"acRates"       , Name::UINT8 , 1});
        NameVec.push_back({"timeSlot"      , Name::UINT8 , 0});
        NameVec.push_back({"timeSlotPhase" , Name::UINT16, 0});
        NameVec.push_back({"ebeamPresent"  , Name::UINT8 , 0});
        NameVec.push_back({"ebeamDestn"    , Name::UINT8 , 0});
        NameVec.push_back({"ebeamCharge"   , Name::UINT16, 0});
        NameVec.push_back({"ebeamEnergy"   , Name::UINT16, 1});
        NameVec.push_back({"xWavelength"   , Name::UINT16, 1});
        NameVec.push_back({"dmod5"         , Name::UINT16, 0});
        NameVec.push_back({"mpsLimits"     , Name::UINT8 , 1});
        NameVec.push_back({"mpsPowerClass" , Name::UINT8 , 1});
        NameVec.push_back({"sequenceValues", Name::UINT16, 1});
    }

    void TimingDef::describeData(XtcData::Xtc&         xtc, 
                                 XtcData::NamesLookup& namesLookup, 
                                 XtcData::NamesId      namesId,
                                 const uint8_t*        data)
    {
        char cbuff[256]; char* cptr=cbuff;
        for(unsigned i=0; i<16; i++)
            cptr += sprintf(cptr,"%08x_",reinterpret_cast<const uint32_t*>(data)[i]);
        sprintf(cptr,"\n");
        logging::warning(cbuff);

        DescribedData ts(xtc, namesLookup, namesId);

        unsigned data_size = 968/8;
        memcpy(ts.data(), data, data_size);
        ts.set_data_length(data_size);

        ts.set_array_shape(TimingDef::fixedRates    ,(const unsigned[MaxRank]){10});
        ts.set_array_shape(TimingDef::acRates       ,(const unsigned[MaxRank]){ 6});
        ts.set_array_shape(TimingDef::ebeamEnergy   ,(const unsigned[MaxRank]){ 4});
        ts.set_array_shape(TimingDef::xWavelength   ,(const unsigned[MaxRank]){ 2});
        ts.set_array_shape(TimingDef::mpsLimits     ,(const unsigned[MaxRank]){16});
        ts.set_array_shape(TimingDef::mpsPowerClass ,(const unsigned[MaxRank]){16});
        ts.set_array_shape(TimingDef::sequenceValues,(const unsigned[MaxRank]){18});
    }

    void TimingDef::createDataNoBSA(XtcData::Xtc&         xtc,
                                    XtcData::NamesLookup& namesLookup,
                                    XtcData::NamesId      namesId,
                                    const uint8_t*        p)
    {
        char cbuff[256]; char* cptr=cbuff;
        for(unsigned i=0; i<16; i++)
            cptr += sprintf(cptr,"%08x_",reinterpret_cast<const uint32_t*>(p)[i]);
        sprintf(cptr,"\n");
        logging::warning(cbuff);

        CreateData cd(xtc, namesLookup, namesId);
        unsigned index = 0;
        unsigned shape[MaxRank];

#define BYTEFROMBIT(V,N,SH0) {                                          \
            shape[0] = N;                                               \
            Array<uint8_t> arrayT = cd.allocate<uint8_t>(index++, shape); \
            for(unsigned i=0; i<N; i++)                                 \
                arrayT.data()[N-1-i] = (V>>(SH0+i))&1;                  \
        }

#define COPYARRAY(T,N) {                                        \
            shape[0] = N;                                       \
            Array<T> arrayT = cd.allocate<T>(index++, shape);   \
            for(unsigned i=0; i<N; i++, p += sizeof(T))                 \
                arrayT.data()[i] = *reinterpret_cast<const T*>(p);      \
        }

        cd.set_value(index++, *reinterpret_cast<const uint64_t*>(p)); p += sizeof(uint64_t);  // pulseId
        cd.set_value(index++, *reinterpret_cast<const uint64_t*>(p)); p += sizeof(uint64_t);  // timeStamp
        BYTEFROMBIT((*reinterpret_cast<const uint16_t*>(p)),10, 0);
        BYTEFROMBIT((*reinterpret_cast<const uint16_t*>(p)), 6,10);
        p += sizeof(uint16_t);
        cd.set_value(index++, uint8_t(*p & 0x7));  // acTimeSlot
        cd.set_value(index++, uint16_t((*reinterpret_cast<const uint16_t*>(p) >> 3) & 0x7fff)); // acTimeSlotPhase
        p += sizeof(uint16_t);
        cd.set_value(index++, uint8_t(*p & 1));  // ebeamPresent
        cd.set_value(index++, uint8_t(*p >> 4)); // ebeamDestn
        p++;
        cd.set_value(index++, *reinterpret_cast<const uint16_t*>(p)); // ebeamCharge
        p += sizeof(uint16_t);
        COPYARRAY(uint16_t, 4);   // ebeamEnergy
        COPYARRAY(uint16_t, 2);   // xWavelength
        cd.set_value(index++, *reinterpret_cast<const uint16_t*>(p)); // dmod5
        p += sizeof(uint16_t);
        BYTEFROMBIT((*reinterpret_cast<const uint16_t*>(p)),16, 0); // mpsLimits
        {
            shape[0] = 16;
            Array<uint8_t> arrayT = cd.allocate<uint8_t>(index++, shape);
            for(unsigned i=0; i<16; i+=2, p++) {
                arrayT.data()[i+0] = *reinterpret_cast<const uint8_t*>(p)&0xf;
                arrayT.data()[i+1] = *reinterpret_cast<const uint8_t*>(p)>>4;
            }
        }
        COPYARRAY(uint16_t, 18);  // sequenceValues
#undef COPYARRAY
#undef BYTEFROMBIT        
    }

    void TimingDef::createDataETM(XtcData::Xtc&         xtc,
                                  XtcData::NamesLookup& namesLookup,
                                  XtcData::NamesId      namesId,
                                  const uint8_t*        pHdr,
                                  const uint8_t*        pEtm)
    {
        char cbuff[256]; char* cptr=cbuff;
        for(unsigned i=0; i<16; i++)
            cptr += sprintf(cptr,"%08x_",reinterpret_cast<const uint32_t*>(pEtm)[i]);
        sprintf(cptr,"\n");
        logging::warning(cbuff);

        CreateData cd(xtc, namesLookup, namesId);
        unsigned index = 0;
        unsigned shape[MaxRank];

#define BYTEFROMBIT(V,N,SH0) {                                          \
            shape[0] = N;                                               \
            Array<uint8_t> arrayT = cd.allocate<uint8_t>(index++, shape); \
            for(unsigned i=0; i<N; i++)                                 \
                arrayT.data()[N-1-i] = (V>>(SH0+i))&1;                  \
        }

#define COPYARRAY(T,N) {                                                \
            shape[0] = N;                                               \
            Array<T> arrayT = cd.allocate<T>(index++, shape);           \
            for(unsigned i=0; i<N; i++, pEtm += sizeof(T))              \
                arrayT.data()[i] = *reinterpret_cast<const T*>(pEtm);   \
        }
#define ZEROARRAY(T,N) {                                        \
            shape[0] = N;                                       \
            Array<T> arrayT = cd.allocate<T>(index++, shape);   \
            memset(arrayT.data(), 0, N*sizeof(T));              \
    }

        cd.set_value(index++, *reinterpret_cast<const uint64_t*>(pHdr));   // pulseId
        pHdr += sizeof(uint64_t);
        pEtm += sizeof(uint64_t);
        cd.set_value(index++, *reinterpret_cast<const uint64_t*>(pHdr));   // timeStamp
        pHdr += sizeof(uint64_t);
        BYTEFROMBIT((*reinterpret_cast<const uint16_t*>(pEtm)),10, 0);
        BYTEFROMBIT((*reinterpret_cast<const uint16_t*>(pEtm)), 6,10);
        pEtm += sizeof(uint16_t);
        cd.set_value(index++, *pEtm++);  // acTimeSlot
        cd.set_value(index++, uint16_t(0));        // acTimeSlotPhase
        cd.set_value(index++, uint8_t(*pEtm & 1));  // ebeamPresent
        cd.set_value(index++, uint8_t(*pEtm >> 4)); // ebeamDestn
        pEtm++;
        cd.set_value(index++, uint16_t(0)); // ebeamCharge
        ZEROARRAY(uint16_t, 4);   // ebeamEnergy
        ZEROARRAY(uint16_t, 2);   // xWavelength
        cd.set_value(index++, uint16_t(0)); //dmod5
        ZEROARRAY(uint8_t, 16);  // mpsLimits
        ZEROARRAY(uint8_t, 16);  // mpsPowerClasses
        COPYARRAY(uint16_t, 18);  // sequenceValues
    }
#undef ZEROARRAY
#undef COPYARRAY
#undef BYTEFROMBIT        
}
