#include "Digitizer.hh"
#include "TimingHeader.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/digitizer/Stream.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using namespace rapidjson;

class HsdDef : public VarDef
{
public:
    HsdDef(unsigned lane_mask)
    {
        Alg alg("fpga", 1, 2, 3);
        const unsigned nameLen = 7;
        char chanName[nameLen];
        NameVec.push_back({"env", Name::UINT32, 1});
        for (unsigned i = 0; i < sizeof(lane_mask)*sizeof(uint8_t); i++){
            if (!((1<<i)&lane_mask)) continue;
            snprintf(chanName, nameLen, "chan%2.2d", i);
            NameVec.push_back({chanName, alg});
        }
    }
};

Digitizer::Digitizer(Parameters* para) :
    Detector(para),
    m_evtcount(0),
    m_evtNamesId(para->tPrms.id, EventNamesIndex)
{
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

unsigned addJson(Xtc& xtc, NamesId& configNamesId) {

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.hsd_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"hsd_config");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssss","DAQ:LAB2:HSD:DEV02",
                                              "mcbrowne:psana@psdb-dev:9306",
                                              "configDB", "TMO", "BEAM",
                                              "xpphsd");
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    printf("json: %s\n",json);

    // convert to json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    unsigned len = translateJson2Xtc(json, buffer, configNamesId);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)buffer;
    memcpy(xtc.next(),jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    // get the lane mask from the json
    Document top;
    if (top.Parse(json).HasParseError())
        fprintf(stderr,"*** json parse error\n");
    const Value& enable = top["enable"];
    std::string enable_type = top[":types:"]["enable"][0].GetString();
    unsigned length = top[":types:"]["enable"][1].GetInt();

    unsigned lane_mask = 0;
    for (unsigned i=0; i<length; i++) if (enable[i].GetInt()) lane_mask |= 1<< i;
    printf("hsd lane_mask is 0x%x\n",lane_mask);

    return lane_mask;
}

void Digitizer::configure(Dgram& dgram, PGPData* pgp_data)
{
    // copy Event header into beginning of Datagram
    // cpo: feels like this should not be in the detector-specific code.
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Pds::TimingHeader* timing_header = reinterpret_cast<Pds::TimingHeader*>(pgp_data->buffers[index].data);
    dgram.seq = timing_header->seq;
    dgram.env = timing_header->env;

    unsigned lane_mask;
    // set up the names for the configuration data
    NamesId configNamesId(m_nodeId,ConfigNamesIndex);
    lane_mask = addJson(dgram.xtc, configNamesId);

    // set up the names for L1Accept data
    Alg hsdAlg("hsd", 1, 2, 3); // TODO: should this be configured by hsdconfig.py?
    unsigned segment = 0;
    Names& eventNames = *new(dgram.xtc) Names("xpphsd", hsdAlg, "hsd", "detnum1235", m_evtNamesId, segment);
    HsdDef myHsdDef(lane_mask);
    eventNames.add(dgram.xtc, myHsdDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
}

void Digitizer::event(Dgram& dgram, PGPData* pgp_data)
{
    // copy Event header into beginning of Datagram
    // cpo: feels like this should not be in the detector-specific code.
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Pds::TimingHeader* timing_header = reinterpret_cast<Pds::TimingHeader*>(pgp_data->buffers[index].data);
    dgram.seq = timing_header->seq;
    dgram.env = timing_header->env;

    CreateData hsd(dgram.xtc, m_namesLookup, m_evtNamesId);

    // HSD data includes two uint32_t "event header" words
    unsigned data_size;
    unsigned shape[MaxRank];
    shape[0] = 2;
    Array<uint32_t> arrayH = hsd.allocate<uint32_t>(0, shape);
    // FIXME: check that Matt is sending this extra HSD info in the
    // timing header
    arrayH(0) = timing_header->_opaque[0];
    arrayH(1) = timing_header->_opaque[1];
    for (int l=0; l<8; l++) { // TODO: print npeaks using psalg/Hsd.hh
        if (pgp_data->buffer_mask & (1 << l)) {
            // size without Event header
            data_size = pgp_data->buffers[l].size - sizeof(Pds::TimingHeader);
            shape[0] = data_size;
            Array<uint8_t> arrayT = hsd.allocate<uint8_t>(l+1, shape);
            memcpy(arrayT.data(), (uint8_t*)pgp_data->buffers[l].data + sizeof(Pds::TimingHeader), data_size);
         }
    }
}
