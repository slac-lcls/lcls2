#include "Digitizer.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "rapidjson/document.h"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using namespace rapidjson;

class HsdDef : public VarDef
{
public:
    enum index
    {
        chan0,
        chan1,
        chan2,
        chan3,
    };

    HsdDef()
    {
        Alg alg("fpga", 1, 2, 3);
        NameVec.push_back({"chan0", alg});
        NameVec.push_back({"chan1", alg});
        NameVec.push_back({"chan2", alg});
        NameVec.push_back({"chan3", alg});
    }
};
static HsdDef myHsdDef;

class HsdConfigDef:public VarDef
{
public:
  enum index
    {
      enable,
      raw_prescale
    };

  HsdConfigDef()
   {
       NameVec.push_back({"enable",Name::UINT64,1});
       NameVec.push_back({"raw_prescale",Name::UINT64,1});
   }
};
static HsdConfigDef myHsdConfigDef;

Digitizer::Digitizer() : m_evtcount(0) {}

void addJson(Xtc& xtc, std::vector<NameIndex>& namesVec) {

    FILE* file;
    Py_Initialize();
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* main_dict = PyModule_GetDict(main_module);
    file = fopen("hsdConfig.py","r");
    PyObject* pyrunfileptr = PyRun_File(file, "hsdConfig.py",Py_file_input,main_dict,main_dict);
    assert(pyrunfileptr!=NULL);
    PyObject* mybytes = PyDict_GetItemString(main_dict,"dgram");
    printf("%p\n",mybytes);
    char* json = (char*)PyBytes_AsString(mybytes);
    printf("%p\n",mybytes);
    printf("%s\n",json);
    Py_Finalize();
    printf("Done\n");

    Document d;
    d.Parse(json);

    Value& software = d["alg"]["software"];
    Value& version = d["alg"]["version"];
    for (SizeType i = 0; i < version.Size(); i++) // Uses SizeType instead of size_t
        printf("version[%d] = %d\n", i, version[i].GetInt());

    Alg hsdConfigAlg(software.GetString(),version[0].GetInt(),version[1].GetInt(),version[2].GetInt());
    Names& configNames = *new(xtc) Names("xpphsd", hsdConfigAlg, "hsd", "detnum1235");
    configNames.add(xtc, myHsdConfigDef);
    namesVec.push_back(NameIndex(configNames));

    CreateData fex(xtc, namesVec, 3); //FIXME: avoid hardwiring nameId

    // TODO: dynamically discover

    Value& enable = d["xtc"]["ENABLE"];
    unsigned shape[MaxRank] = {enable.Size()};
    Array<uint64_t> arrayT = fex.allocate<uint64_t>(HsdConfigDef::enable,shape); //FIXME: figure out avoiding hardwired zero
    for(unsigned i=0; i<shape[0]; i++){
        arrayT(i) = (uint64_t) enable[i].GetInt();
    };

    Value& raw_prescale = d["xtc"]["RAW_PS"];
    shape[MaxRank] = {raw_prescale.Size()};
    Array<uint64_t> arrayT1 = fex.allocate<uint64_t>(HsdConfigDef::raw_prescale,shape); //FIXME: figure out avoiding hardwired zero
    for(unsigned i=0; i<shape[0]; i++){
        arrayT1(i) = (uint64_t) raw_prescale[i].GetInt();
    };
}

void Digitizer::configure(Dgram& dgram, PGPData* pgp_data)
{
    printf("Digitizer configure\n");

    // copy Event header into beginning of Datagram
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);
    memcpy(&dgram, event_header, sizeof(Transition));

    Alg hsdAlg("hsd", 1, 2, 3);
    unsigned segment = 0;
    Names& hsdNames = *new(dgram.xtc) Names("xpphsd", hsdAlg, "hsd", "detnum0", segment);
    hsdNames.add(dgram.xtc, myHsdDef);
    m_namesVec.push_back(NameIndex(hsdNames));

    addJson(dgram.xtc, m_namesVec);
}

void Digitizer::event(Dgram& dgram, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    unsigned nameId=0;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);
    memcpy(&dgram, event_header, sizeof(Transition));
    CreateData hsd(dgram.xtc, m_namesVec, nameId);
    printf("*** evt count %d isevt %d control %x\n",event_header->evtCounter,dgram.seq.isEvent(),dgram.seq.pulseId().control());
    
    unsigned data_size;
    unsigned shape[MaxRank];
    for (int l=0; l<8; l++) {
        if (pgp_data->buffer_mask & (1 << l)) {
            // size without Event header
            data_size = pgp_data->buffers[l]->size - sizeof(Transition);
            shape[0] = data_size;
            Array<uint8_t> arrayT = hsd.allocate<uint8_t>(l, shape);
            memcpy(arrayT.data(), (uint8_t*)pgp_data->buffers[l]->virt + sizeof(Transition), data_size);
         }
    }
}
