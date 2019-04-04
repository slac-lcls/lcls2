#include <Python.h>
#include <stdio.h>
#include <assert.h>

#include "xtcdata/xtc/Json2Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

static char buffer[4*1024*1024];

// FIXME (cpo): need to add appropriate DECREF's

int main() {
    const char fname[] = "get_config.py";
    FILE* file;
    Py_Initialize();
    PyObject* main_module = PyImport_AddModule("__main__");
    static const unsigned argc = 5;
    const wchar_t* argv[argc] = {L"cpo:psana@psdb-dev:9306", L"cpotest", L"AMO", L"BEAM", L"xpphsd1"};
    PySys_SetArgv(argc,const_cast<wchar_t**>(argv));
    PyObject* main_dict = PyModule_GetDict(main_module);
    file = fopen(fname,"r");
    PyObject* pyrunfileptr = PyRun_File(file, fname, Py_file_input, main_dict, main_dict);
    if (pyrunfileptr==NULL) {
        PyErr_Print();
        exit(1);
    }
    PyObject* mybytes = PyDict_GetItemString(main_dict,"config_json");
    PyObject * temp_bytes = PyUnicode_AsASCIIString(mybytes);
    char* json = (char*)PyBytes_AsString(temp_bytes);
    printf("json: %s\n",json);
    Py_Finalize();

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

    return 0;
}
