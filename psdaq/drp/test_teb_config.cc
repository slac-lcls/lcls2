#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include "rapidjson/document.h"

using namespace rapidjson;

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
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"get_config_json");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"sssss","https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/", "configDB", "TMO", "BEAM", "tmoteb");
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);

    // example code demonstrating how to access JSON from C++
    // to be concise, this doesn't do the rapidjson type-checking
    Document top;
    if (top.Parse(json).HasParseError())
        fprintf(stderr,"*** json parse error\n");
    std::string libname = top["soname"].GetString();
    std::cout << "libname is " << libname << std::endl;

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    Py_Finalize();
    return 0;
}
