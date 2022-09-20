#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include <getopt.h>
#include "rapidjson/document.h"

#include "psdaq/service/Json2Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/TypeId.hh"

using namespace XtcData;
using namespace rapidjson;

#define BUFSIZE 1024*1024*32
// convert json to xtc
static char config_buf[BUFSIZE];
static char dgbuf[BUFSIZE];

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    int c;
    const char* detname=0;
    while((c = getopt(argc, argv, "d:")) != EOF) {
        switch(c) {
            case 'd':
                detname = optarg;
                break;
            default:
                exit(1);
        }
    }
    if (!detname) {
        printf("Must provide detector name with -d argument\n");
        exit(1);
    }
    Py_Initialize();
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psdaq.configdb.get_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"get_config_json_with_params");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"sssss","https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/", "TMO", "configDB", "BEAM", detname);
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);

    printf("json: %s\n",json);

    NamesId namesid(0,1);
    const void* config_end = config_buf + sizeof(config_buf);
    unsigned len = Pds::translateJson2Xtc(json, config_buf, config_end, namesid, detname);
    if (len <= 0) {
        fprintf(stderr, "Parse errors, exiting.\n");
        exit(1);
    }

    // example code demonstrating how to access JSON from C++
    // to be concise, this doesn't do the rapidjson type-checking
    // this is hsd-specific, so comment out by default
    Document top;
    if (top.Parse(json).HasParseError())
        fprintf(stderr,"*** json parse error\n");
    // array example
    // const Value& raw_start = top["raw"]["start"];
    // std::string start_type = top[":types:"]["raw"]["start"][0].GetString();
    // unsigned length = top[":types:"]["raw"]["start"][1].GetInt();
    // std::cout << "raw.start is " << raw_start[0].GetInt() << " with type " << start_type << " and length " << length << std::endl;
    // // non-array example
    // const Value& expert_fullthresh = top["expert"]["fullthresh"];
    // std::string fullthresh_type = top[":types:"]["expert"]["fullthresh"].GetString();
    // std::cout << "expert.fullthresh is " << expert_fullthresh.GetInt() << " with type " << fullthresh_type << std::endl;

    // const Value& enable = top["enable"];
    // std::string enable_type = top[":types:"]["enable"][0].GetString();
    // unsigned enable_length = top[":types:"]["enable"][1].GetInt();

    // unsigned lane_mask = 0;
    // for (unsigned i=0; i<enable_length; i++) if (enable[i].GetInt()) lane_mask |= 1<< i;
    // printf("hsd lane_mask is 0x%x\n",lane_mask);

    Xtc& xtcbuf = *(Xtc*)config_buf;

    // make a fake dgram
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    Transition tr(Dgram::Event, TransitionId::Configure, TimeStamp(tv.tv_sec, tv.tv_usec), env);
    Dgram& dg = *new(&dgbuf) Dgram(tr, Xtc(tid));
    const void* dgend = dgbuf + sizeof(dgbuf);

    // copy over the names/shapesdata xtc's (translateJson2Xtc puts
    // a Parent Xtc on the top level so we can copy over both
    // names/shapesdata at the same time)
    auto payload = dg.xtc.alloc(xtcbuf.sizeofPayload(), dgend);
    memcpy(payload,(const void*)xtcbuf.payload(),xtcbuf.sizeofPayload());

    FILE* fp = fopen("junk.xtc2", "w+");
    if (fwrite(&dg, 1, sizeof(Dgram), fp) != sizeof(Dgram)) {
        printf("Cannot write dgram header\n");
        exit(1);
    }
    unsigned payloadSize = dg.xtc.sizeofPayload();
    if (fwrite(dg.xtc.payload(), 1, payloadSize, fp) != payloadSize) {
        printf("Cannot write payload\n");
        exit(1);
    }
    fclose(fp);

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    Py_Finalize();
    return 0;
}
