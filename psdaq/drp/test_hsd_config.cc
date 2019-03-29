#include <Python.h>
#include <stdio.h>
#include <assert.h>

#include "xtcdata/xtc/Json2Xtc.hh"
#include "xtcdata/xtc/NamesId.hh"

static char buffer[4*1024*1024];

int main() {
    const char fname[] = "test_hsd_config.py";
    FILE* file;
    Py_Initialize();
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* main_dict = PyModule_GetDict(main_module);
    file = fopen(fname,"r");
    PyObject* pyrunfileptr = PyRun_File(file, fname, Py_file_input, main_dict, main_dict);
    assert(pyrunfileptr!=NULL);
    PyObject* mybytes = PyDict_GetItemString(main_dict,"config_json");
    PyObject * temp_bytes = PyUnicode_AsASCIIString(mybytes);
    char* json = (char*)PyBytes_AsString(temp_bytes);
    printf("json: %s\n",json);
    Py_Finalize();

    // convert to json to xtc
    XtcData::NamesId namesid(0,1);
    int len = XtcData::translateJson2Xtc(json, buffer, namesid);
    if (len <= 0) {
        fprintf(stderr, "Parse errors, exiting.\n");
        exit(0);
    }
    FILE* fp = fopen("junk.xtc2", "w+");
    char dgram[20] = {0};   // Hack: sizeof(Dgram) - sizeof(Xtc) = 20
    if (fwrite(dgram, 1, 20, fp) != 20) {
        printf("Cannot write dgram header\n");
    }
    if (fwrite(buffer, 1, len, fp) != len) {
        printf("Cannot write payload\n");
        exit(0);
    }
    fclose(fp);

    return 0;
}
