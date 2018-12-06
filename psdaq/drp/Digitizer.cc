#include "Digitizer.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"

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

// TODO: this is not ideal. Should be done dynamically using information returned in the json
class HsdConfigDef:public VarDef
{
public:
  enum index
    {
      enable,
      raw_prescale,
      lane_mask,
    };

  HsdConfigDef()
   {
       NameVec.push_back({"enable",Name::UINT64,1});
       NameVec.push_back({"raw_prescale",Name::UINT64,1});
       NameVec.push_back({"lane_mask",Name::UINT64,1});
   }
};
static HsdConfigDef myHsdConfigDef;

class HsdIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    HsdIter(Xtc* xtc, std::vector<NameIndex>& namesVec) : XtcIterator(xtc), _namesVec(namesVec)
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {

        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            unsigned namesId = shapesdata.shapes().namesId();
            DescData descdata(shapesdata, _namesVec[namesId]);
            Names& names = descdata.nameindex().names();

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                // Here we check algorithm and version can be analyzed
                if (strcmp(name.alg().name(), "fpga") == 0) {
                    if (name.alg().version() == 0x010203) {
                        auto array = descdata.get_array<uint8_t>(i);
                        chans[std::string(name.name())] = array.data();
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    std::vector<NameIndex>& _namesVec;


public:
    std::map <std::string, uint8_t*> chans;
};

Digitizer::Digitizer(unsigned src) : Detector(src), m_evtcount(0) {
}

unsigned addJson(Xtc& xtc, std::vector<NameIndex>& namesVec, Src& src) {

    FILE* file;
    Py_Initialize();
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* main_dict = PyModule_GetDict(main_module);
    file = fopen("hsdconfig.py","r");
    PyObject* pyrunfileptr = PyRun_File(file, "hsdconfig.py", Py_file_input, main_dict, main_dict);
    assert(pyrunfileptr!=NULL);
    PyObject* mybytes = PyDict_GetItemString(main_dict,"config");
    PyObject * temp_bytes = PyUnicode_AsEncodedString(mybytes, "UTF-8", "strict");
    char* json = (char*)PyBytes_AsString(temp_bytes);
    Py_Finalize();

    Document d;
    d.Parse(json);

    Value& software = d["alg"]["software"];
    Value& version = d["alg"]["version"];
    for (SizeType i = 0; i < version.Size(); i++) // Uses SizeType instead of size_t
        printf("version[%d] = %d\n", i, version[i].GetInt());

    // High level algorithm information
    Value& ninfo_software = d["ninfo"]["software"];
    Value& ninfo_detector = d["ninfo"]["detector"];
    Value& ninfo_serialNum = d["ninfo"]["serialNum"];
    Value& ninfo_seg = d["ninfo"]["seg"];

    Alg hsdConfigAlg(software.GetString(),version[0].GetInt(),version[1].GetInt(),version[2].GetInt());
    Names& configNames = *new(xtc) Names(ninfo_software.GetString(), hsdConfigAlg,
                                         ninfo_detector.GetString(), ninfo_serialNum.GetString(), src);
    configNames.add(xtc, myHsdConfigDef);
    namesVec.push_back(NameIndex(configNames));

    unsigned namesId = 0;
    CreateData fex(xtc, namesVec, namesId, src); //FIXME: avoid hardwiring nameId

    // TODO: dynamically discover

    Value& enable = d["xtc"]["ENABLE"];
    unsigned shape[MaxRank] = {enable.Size()};
    Array<uint64_t> arrayT = fex.allocate<uint64_t>(HsdConfigDef::enable,shape); //FIXME: figure out avoiding hardwired zero
    unsigned lane_mask = 0;
    for(unsigned i=0; i<shape[0]; i++){
        arrayT(i) = (uint64_t) enable[i].GetInt();
        lane_mask += enable[i].GetInt() << (shape[0]-(i+1)); // convert enable to unsigned using bitshift
    };

    Value& raw_prescale = d["xtc"]["RAW_PS"];
    shape[MaxRank] = {raw_prescale.Size()};
    Array<uint64_t> arrayT1 = fex.allocate<uint64_t>(HsdConfigDef::raw_prescale,shape); //FIXME: figure out avoiding hardwired zero
    for(unsigned i=0; i<shape[0]; i++){
        arrayT1(i) = (uint64_t) raw_prescale[i].GetInt();
    };

    return lane_mask;
}

void Digitizer::configure(Dgram& dgram, PGPData* pgp_data)
{
    // copy Event header into beginning of Datagram
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);
    memcpy(&dgram, event_header, sizeof(Transition));

    unsigned lane_mask;
    lane_mask = addJson(dgram.xtc, m_namesVec, _src);

    Alg hsdAlg("hsd", 1, 2, 3); // TODO: shouldn't this be configured by hsdconfig.py?
    unsigned segment = 0;
    Names& dataNames = *new(dgram.xtc) Names("xpphsd", hsdAlg, "hsd", "detnum1235", _src, segment);
    HsdDef myHsdDef(lane_mask);
    dataNames.add(dgram.xtc, myHsdDef);
    m_namesVec.push_back(NameIndex(dataNames));

    HsdIter hsdIter(&dgram.xtc, m_namesVec);
    hsdIter.iterate();
    unsigned nChan = hsdIter.chans.size();
}

void Digitizer::event(Dgram& dgram, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    unsigned namesId=1;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);

    memcpy(&dgram, event_header, sizeof(Transition));
    CreateData hsd(dgram.xtc, m_namesVec, namesId, _src);

    unsigned data_size;
    unsigned shape[MaxRank];
    shape[0] = 2;
    Array<uint32_t> arrayH = hsd.allocate<uint32_t>(0, shape);
    arrayH(0) = dgram.env[1];
    arrayH(1) = dgram.env[2];
    for (int l=0; l<8; l++) { // TODO: print npeaks using psalg/Hsd.hh
        if (pgp_data->buffer_mask & (1 << l)) {
            // size without Event header
            data_size = pgp_data->buffers[l]->size - sizeof(Transition);
            shape[0] = data_size;
            Array<uint8_t> arrayT = hsd.allocate<uint8_t>(l+1, shape);
            memcpy(arrayT.data(), (uint8_t*)pgp_data->buffers[l]->virt + sizeof(Transition), data_size);
            Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[l]->virt);
         }
    }
}
