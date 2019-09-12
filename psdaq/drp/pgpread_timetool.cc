#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include "TimingHeader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <Python.h>

#define MAX_RET_CNT_C 1000
static int fd;
std::atomic<bool> terminate;

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}


unsigned dmaDest(unsigned lane, unsigned vc)
{
    return (lane<<8) | vc;
}

void int_handler(int dummy)
{
    terminate.store(true, std::memory_order_release);
    // dmaUnMapDma();
}

int toggle_acquisition(int x)
{
    printf("starting prescaler config testing  \n");

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    Py_Initialize();
    PyObject* sysPath = PySys_GetObject((char*)"path");


    //PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/software/TimeTool/scripts"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/submodules/surf/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/submodules/axi-pcie-core/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/submodules/lcls2-pgp-fw-lib/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/submodules/lcls-timing-core/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/submodules/lcls2-pgp-fw-lib/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/applications/TimeTool/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/firmware/submodules/clink-gateway-fw-lib/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/software/TimeTool/python"));
    PyList_Append(sysPath, PyUnicode_FromString("/u1/sioan/slac-lcls/lcls2/lcls2-pcie-apps/software/TimeTool/scripts"));


    pName = PyUnicode_DecodeFSDefault("toggle_prescaling");


    pModule = PyImport_Import(pName);

    check(pModule);

    if (!pModule){
    printf("can't find module \n");
    return 0;
    }

    pFunc = PyObject_GetAttrString(pModule, "toggle_prescaling");
    check(pFunc);

    PyObject_CallFunction(pFunc, NULL);

    Py_XDECREF(pFunc);


     if (PyErr_Occurred()){
                PyErr_Print();
        }

    //Py_XDECREF(pArgs);
    //Py_XDECREF(pModule);
    //Py_XDECREF(sysPath);
    //Py_XDECREF(pValue);
    //Py_XDECREF(pName);

    printf("ending prescaler config testing \n ");

}

int main_b(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return 1;
    }

    Py_Initialize();

    PyObject* sysPath = PySys_GetObject((char*)"path");
    PyList_Append(sysPath, PyUnicode_FromString("."));

    pName = PyUnicode_DecodeFSDefault(argv[1]);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(argc - 3);
            for (i = 0; i < argc - 3; ++i) {
                pValue = PyLong_FromLong(atoi(argv[i + 3]));
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    printf("starting main \n");
    toggle_acquisition(0);

    int c, channel;

    timespec ts; 

    channel = 0;
    std::string device;
    while((c = getopt(argc, argv, "c:d:")) != EOF) {
        switch(c) {
            case 'd':
                device = optarg;
                break;
            case 'c':
                channel = atoi(optarg);
                break;
        }
    }

    terminate.store(false, std::memory_order_release);
    signal(SIGINT, int_handler);

    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (unsigned i=0; i<4; i++) {
        dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, channel));
    }

    std::cout<<"device  "<<device<<'\n';
    fd = open(device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<<device<<'\n';
        return -1;
    }

    uint32_t dmaCount, dmaSize;
    void** dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (dmaBuffers == NULL ) {
        printf("Failed to map dma buffers!\n");
        return -1;
    }
    printf("dmaCount %u  dmaSize %u\n", dmaCount, dmaSize);

    dmaSetMaskBytes(fd, mask);


    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dmaDest[MAX_RET_CNT_C];

    uint8_t *raw_data;

    uint8_t expected_next_count = 0;

    uint32_t raw_counter = 0;

    while (1) {
        if (terminate.load(std::memory_order_acquire) == true) {
            close(fd);
            printf("closed\n");
            break;
        }

        clock_gettime(CLOCK_REALTIME, &ts);

        int32_t ret = dmaReadBulkIndex(fd, MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dmaDest);
        for (int b=0; b < ret; b++) {
            uint32_t index = dmaIndex[b];
            uint32_t size = dmaRet[b];
            //uint32_t dest = dmaDest[b] >> 8;
            raw_data = reinterpret_cast<uint8_t *>(dmaBuffers[index]);

            //if(size !=2112){
            //    printf("corrupted frame. size = %d",size);
            //}
            

            if(raw_counter%100000 == 0){
                printf("%x %x %x %x %d %d %d %d",raw_data[1],expected_next_count,raw_data[32],raw_data[32],ts.tv_sec,ts.tv_nsec,raw_counter,size);
                printf("\n");
            }


            raw_counter = raw_counter + 1;

            if(expected_next_count != raw_data[1]){
                printf("Dropped shot");
                printf("\n");
            }

            expected_next_count = (raw_data[1]+1)%256;

            //Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(dmaBuffers[index]);
            //XtcData::TransitionId::Value transition_id = event_header->seq.service();

            //printf("Size %u B | Dest %u | Transition id %d | pulse id %lu | event counter %u | index %u\n",
            //       size, dest, transition_id, event_header->seq.pulseId().value(), event_header->evtCounter, index);
            //printf("env %08x\n", event_header->env);
        }
	    if ( ret > 0 ) dmaRetIndexes(fd, ret, dmaIndex);
	    //sleep(0.1)
    }
    printf("finished\n");
}
