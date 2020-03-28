#include "Wave8.hh"
#include "EventBatcher.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "DataDriver.h"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>

using namespace XtcData;
using namespace rapidjson;
using logging = psalg::SysLog;

using json = nlohmann::json;

namespace Drp {

  namespace W8 {
    class RawStream {
    public:
        static void varDef(VarDef& v,unsigned ch) {
            char name[32];
            // raw streams
            sprintf(name,"raw_%d",ch);
            v.NameVec.push_back(XtcData::Name(name, XtcData::Name::UINT16,1));
        }
        static void createData(CreateData& cd, unsigned& index, unsigned ch,
                               void* segptr, unsigned segsize) {
            unsigned shape[MaxRank];
            shape[0] = segsize>>1;
            Array<uint16_t> arrayT = cd.allocate<uint16_t>(index++, shape);
            memcpy(arrayT.data(), segptr, segsize);
        }
    };
    class IntegralStream {
    public:
        static void varDef(XtcData::VarDef& v) {
            char name[32];
            // v.NameVec.push_back(XtcData::Name("integralSize", XtcData::Name::UINT8));
            // v.NameVec.push_back(XtcData::Name("trigDelay"   , XtcData::Name::UINT8));
            v.NameVec.push_back(XtcData::Name("trigCount"   , XtcData::Name::UINT16));
            // v.NameVec.push_back(XtcData::Name("baselineSize", XtcData::Name::UINT16));
            for(unsigned i=0; i<8; i++) {
                sprintf(name,"integral_%d",i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::INT32));  // actually 24-bit signed, need to sign-extend
            }
            for(unsigned i=0; i<8; i++) {
                sprintf(name,"base_%d",i);
                v.NameVec.push_back(XtcData::Name(name, XtcData::Name::UINT16));
            }
        }
      
        static void createData(CreateData& cd, unsigned& index,
                               void* segptr, unsigned segsize) {
            IntegralStream& p = *new(segptr) IntegralStream;
            cd.set_value(index++, p._trigCount);
            for(unsigned i=0; i<8; i++) {
                uint32_t v = p._integral[i];
                if (v & (1<<23))   // sign-extend
                    v |= 0xff000000;
                cd.set_value(index++, (int32_t)v);
            }
            for(unsigned i=0; i<8; i++)
                cd.set_value(index++, p._base[i]);
            //            p._dump();
        }
        IntegralStream() {}
    private:
        void _dump() const {
          printf("integralSize %02x  trigDelay %02x  trigCount %04x  baseSize %04x\n",
                 _integralSize, _trigDelay, _trigCount, _baselineSize);
          for(unsigned i=0; i<8; i++)
              printf(" %d/%u", _integral[i], _base[i]);
          printf("\n");
        }
        uint8_t  _integralSize;
        uint8_t  _trigDelay;
        uint16_t _reserved;
        uint16_t _trigCount;
        uint16_t _baselineSize;
        int32_t  _integral[8];
        uint16_t _base[8];
    };
    class ProcStream {
    public:
        static void varDef(VarDef& v) {
            // v.NameVec.push_back(XtcData::Name("quadrantSel", XtcData::Name::UINT32));
            v.NameVec.push_back(XtcData::Name("trigCount"  , XtcData::Name::UINT32));
            v.NameVec.push_back(XtcData::Name("integral"   , XtcData::Name::DOUBLE));
            v.NameVec.push_back(XtcData::Name("posX"       , XtcData::Name::DOUBLE));
            v.NameVec.push_back(XtcData::Name("posY"       , XtcData::Name::DOUBLE));
        }
        static void createData(CreateData& cd, unsigned& index,
                               void* segptr, unsigned segsize) {
            ProcStream& p = *new(segptr) ProcStream;
            cd.set_value(index++, p._trigCount);
            cd.set_value(index++, p._integral);
            cd.set_value(index++, p._posX);
            cd.set_value(index++, p._posY);
            //            p._dump();
        }
        ProcStream() {}
    private:
        void _dump() const {
            printf("quadrantSel %08x  trigCount %08x  integral %f  posX %f  posY %f\n",
                   _quadrantSel, _trigCount, _integral, _posX, _posY);
        }
        uint32_t _quadrantSel;
        uint32_t _trigCount;
        double   _integral;
        double   _posX;
        double   _posY;
    };
    class Streams {
    public:
        static void defineData(Xtc& xtc, const char* detName, const char* detNum,
                               NamesLookup& lookup, NamesId& raw, NamesId& fex) {
          // set up the names for L1Accept data
          { Alg alg("raw", 0, 0, 1);
            Names& eventNames = *new(xtc) Names(detName, alg, 
                                                "wave8", detNum, raw);
            VarDef v;
            for(unsigned i=0; i<8; i++)
                RawStream::varDef(v,i);
            eventNames.add(xtc, v);
            lookup[raw] = NameIndex(eventNames); }
#if 1
          { Alg alg("fex", 0, 0, 1);
            Names& eventNames = *new(xtc) Names(detName, alg, 
                                                "wave8", detNum, fex);
            VarDef v;
            IntegralStream::varDef(v);
            ProcStream    ::varDef(v);
            eventNames.add(xtc, v);
            lookup[fex] = NameIndex(eventNames); }
#endif
        }
        static void createData(XtcData::Xtc&         xtc, 
                               XtcData::NamesLookup& lookup,
                               XtcData::NamesId&     rawId,
                               XtcData::NamesId&     fexId,
                               void**                streams,
                               const unsigned*       sizes) {
            CreateData raw(xtc, lookup, rawId);

            unsigned index=0;
            for(unsigned i=0; i<8; i++)
                RawStream::createData(raw,index,i,streams[i],sizes[i]);
#if 1
            index=0;
            CreateData fex(xtc, lookup, fexId);

            if (sizes[8])
                IntegralStream::createData(fex,index,streams[8],sizes[8]);

            if (sizes[9])
                ProcStream::createData(fex,index,streams[9],sizes[9]);
#endif
       }
    };
  };

Wave8::Wave8(Parameters* para, MemPool* pool) :
    Detector(para, pool),
    m_evtcount(0),
    m_evtNamesRaw(-1, -1), // placeholder
    m_evtNamesFex(-1, -1), // placeholder
    m_epics_name(para->kwargs["epics_prefix"]),
    m_paddr     (_getPaddr())
{
    para->rogueDet=true;
    para->virtChan=1;
    printf("*** found epics name %s\n",m_epics_name.c_str());
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

unsigned Wave8::_getPaddr()
{
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.w8_connect");
    check(pModule);

    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"w8_connect");
    check(pFunc);

    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"s",
                                              m_epics_name.c_str());

    check(mybytes);

    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json_str = (char*)PyBytes_AsString(json_bytes);

    Document *d = new Document();
    d->Parse(json_str);
    if (d->HasParseError()) {
        printf("Parse error: %s, location %zu\n",
               GetParseError_En(d->GetParseError()), d->GetErrorOffset());
        abort();
    }
    const Value& a = (*d)["paddr"];

    unsigned reg = a.GetInt();
    if (!reg) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
    }

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    return reg;
}

json Wave8::connectionInfo()
{
    // Exclude connection info until Wave8 timingTxLink is fixed
    return json({});

    unsigned reg = m_paddr;
    int xpm  = (reg >> 20) & 0xF;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

unsigned Wave8::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

  timespec tv_b; clock_gettime(CLOCK_REALTIME,&tv_b);

#define CHECK_TIME(s) {                                                 \
    timespec tv; clock_gettime(CLOCK_REALTIME,&tv);                     \
    printf("%s %f seconds\n",#s,                                        \
           double(tv.tv_sec-tv_b.tv_sec)+1.e-9*(double(tv.tv_nsec)-double(tv_b.tv_nsec))); }


    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.w8_config");
    check(pModule);

    CHECK_TIME(PyImport);

    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"w8_config");
    check(pFunc);

    CHECK_TIME(PyDict_Get);

    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssi",
                                              m_connect_json.c_str(),
                                              m_epics_name.c_str(),
                                              config_alias.c_str(),
                                              m_para->detName.c_str(),
                                              m_readoutGroup);

    CHECK_TIME(PyObj_Call);

    //check(mybytes);
    if (!mybytes) {
        PyErr_Print();
        Py_DECREF(pModule);
        return 0;
    }

    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    //    printf("json: %s\n",json);

    // convert to json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    unsigned len = Pds::translateJson2Xtc(json, buffer, configNamesId, m_para->detSegment);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    CHECK_TIME(translateJson);

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)buffer;
    memcpy(xtc.next(),jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    CHECK_TIME(Done);

    return 1;
}

void Wave8::connect(const json& connect_json, const std::string& collectionId)
{
  m_connect_json = connect_json.dump();
  m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
}

unsigned Wave8::configure(const std::string& config_alias, Xtc& xtc)
{
    //  Reset the PGP links
    // int fd = open(m_para->device.c_str(), O_RDWR);
    // //  user reset
    // dmaWriteRegister(fd, 0x00800000, (1<<31));
    // usleep(10);
    // dmaWriteRegister(fd, 0x00800000, 0);
    // //  QPLL reset
    // dmaWriteRegister(fd, 0x00a40024, 1);
    // usleep(10);
    // dmaWriteRegister(fd, 0x00a40024, 0);
    // usleep(10);
    // //  Reset the Tx and Rx
    // dmaWriteRegister(fd, 0x00a40024, 6);
    // usleep(10);
    // dmaWriteRegister(fd, 0x00a40024, 0);
    // close(fd);

    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    if ( !_addJson(xtc, configNamesId, config_alias ) )
        return 1;

    m_evtNamesRaw = NamesId(nodeId, EventNamesIndex+0);
    m_evtNamesFex = NamesId(nodeId, EventNamesIndex+1);
    W8::Streams::defineData(xtc,m_para->detName.c_str(),"detnum1235",
                            m_namesLookup,m_evtNamesRaw,m_evtNamesFex);
    printf("--Configure complete--\n");
    return 0;
}

void Wave8::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    m_evtcount+=1;

    int lane = __builtin_ffs(event->mask) - 1;
    uint32_t dmaIndex = event->buffers[lane].index;
    unsigned data_size = event->buffers[lane].size;
    EvtBatcherIterator ebit = EvtBatcherIterator((EvtBatcherHeader*)m_pool->dmaBuffers[dmaIndex], data_size);
    EvtBatcherSubFrameTail* ebsft;
    unsigned segsize[12];  memset(segsize, 0, sizeof(segsize));
    void*    segptr [12];

    while ((ebsft=ebit.next())) {
        segsize[ebsft->tdest()] = ebsft->size();
        segptr [ebsft->tdest()] = ebsft->data();
    }

    W8::Streams::createData(dgram.xtc, m_namesLookup, m_evtNamesRaw, m_evtNamesFex, 
                            &segptr[2], &segsize[2]);  // stream 0 is event header
}

void Wave8::shutdown()
{
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.w8_config");
    check(pModule);

    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"w8_unconfig");
    check(pFunc);

    // returns new reference
    PyObject_CallFunction(pFunc,"s",
                          m_epics_name.c_str());
    Py_DECREF(pModule);
}  

}
