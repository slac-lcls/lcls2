#include <Python.h>
#include <stdio.h>
#include <assert.h>

#include "xtcdata/xtc/Json2Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

static char buffer[4*1024*1024];

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        exit(1);
    }
}

int main() {
    Py_Initialize();
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.get_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"get_config");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"sssss","cpo:psana@psdb-dev:9306", "cpotest", "AMO", "BEAM", "xpphsd1");
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    printf("json: %s\n",json);

    // convert to json to xtc
    XtcData::NamesId namesid(0,1);
    unsigned len = XtcData::translateJson2Xtc(json, buffer, namesid);
    if (len <= 0) {
        fprintf(stderr, "Parse errors, exiting.\n");
        exit(1);
    }
    FILE* fp = fopen("junk.xtc2", "w+");
    char dgram[20] = {0};   // Hack: sizeof(Dgram) - sizeof(Xtc) = 20
    if (fwrite(dgram, 1, 20, fp) != 20) {
        printf("Cannot write dgram header\n");
    }
    if (fwrite(buffer, 1, len, fp) != len) {
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
