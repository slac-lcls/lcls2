#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include "rapidjson/document.h"

#include "xtcdata/xtc/Json2Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/TypeId.hh"

using namespace XtcData;
using namespace rapidjson;

#define BUFSIZE 1024*1024

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        exit(1);
    }
}

int main() {
    Py_Initialize();
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
    PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssss","dummy_epics_prefix","mcbrowne:psana@psdb-dev:9306", "configDB", "TMO", "BEAM", "xpphsd");
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    printf("json: %s\n",json);

    // convert json to xtc
    char buffer[BUFSIZE];
    NamesId namesid(0,1);
    unsigned len = translateJson2Xtc(json, buffer, namesid);
    if (len <= 0) {
        fprintf(stderr, "Parse errors, exiting.\n");
        exit(1);
    }

    // example code demonstrating how to access JSON from C++
    // to be concise, this doesn't do the rapidjson type-checking
    Document top;
    if (top.Parse(json).HasParseError())
        fprintf(stderr,"*** json parse error\n");
    unsigned start = top["raw"]["start"].GetInt();
    std::string start_type = top[":types:"]["raw"]["start"].GetString();
    std::cout << "raw.start is " << start << " with type " << start_type << std::endl;

    Xtc& xtcbuf = *(Xtc*)buffer;

    // make a fake dgram
    TypeId tid(TypeId::Parent, 0);
    uint64_t pulseId = 0;
    uint32_t env = 0;
    struct timeval tv;
    char dgbuf[BUFSIZE];
    gettimeofday(&tv, NULL);
    Sequence seq(Sequence::Event, TransitionId::Configure, TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId,0));
    Dgram& dg = *new(&dgbuf) Dgram(Transition(seq, env), Xtc(tid));

    // copy over the names/shapesdata xtc's (translateJson2Xtc puts
    // a Parent Xtc on the top level so we can copy over both
    // names/shapesdata at the same time)
    memcpy(dg.xtc.next(),xtcbuf.payload(),xtcbuf.sizeofPayload());
    dg.xtc.alloc(xtcbuf.sizeofPayload());

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
