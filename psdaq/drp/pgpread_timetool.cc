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
#include <ctime>
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

int tt_config(int x)
{
    printf("Initializing python \n");    
    Py_Initialize();
    // returns new reference
    printf("importing module \n");
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.tt_config");
    
    printf("checking module \n");
    check(pModule);
    
    // returns borrowed reference
    printf("getting dict \n");
    PyObject* pDict = PyModule_GetDict(pModule);
    printf("checking dict \n");
    // returns borrowed reference
    printf("loading function \n");
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"tt_config");
    check(pDict);
    printf("checking function \n");
    check(pFunc);


    /*PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssi",
                                              m_connect_json.c_str(),
                                              m_epics_name.c_str(),
                                              "BEAM", 
                                              m_para->detName.c_str(),
                                              m_readoutGroup);*/

    /*PyObject* mybytes = PyObject_CallFunction(pFunc,
                                              m_connect_json.c_str(),
                                              m_epics_name.c_str(),
                                              "BEAM", 
                                              m_para->detName.c_str(),
                                              m_readoutGroup);*/

    //char* m_connect_json_str = "{'body': {'control': {'0': {'control_info': {'instrument': 'TMO', 'cfg_dbase': 'mcbrowne:psana@psdb-dev:9306/configDB'}}}}}" ; 
    char* m_connect_json_str = "\{\"body\": \{\"control\": \{\"0\": \{\"control_info\": \{\"instrument\": \"TMO\", \"cfg_dbase\": \"mcbrowne:psana@psdb-dev:9306/configDB\"\}\}\}\}\}" ;

    PyObject* mybytes = PyObject_CallFunction(pFunc,"sssi",m_connect_json_str,"BEAM", "tmotimetool",0);

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


int main(int argc, char* argv[])
{
    printf("starting main \n");


    int c, channel;

    timespec ts; 

    channel = 0;
    std::string device;
    while((c = getopt(argc, argv, "c:d:t")) != EOF) {
        switch(c) {
            case 'd':
                device = optarg;
                break;
            case 'c':
                channel = atoi(optarg);
                break;
            case 't':
                   printf("entering tt config \n");
                   tt_config(0);
                   //toggle_acquisition(0);  
        }
    }

    printf("finished with tt config \n");
    usleep(1e6);

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


    int32_t      dmaRet[MAX_RET_CNT_C];
    uint32_t     dmaIndex[MAX_RET_CNT_C];
    uint32_t     dmaDest[MAX_RET_CNT_C];

    uint8_t     *raw_data;

    uint8_t      expected_next_count  = 0;

    uint32_t     raw_counter          = 0;
    uint32_t     last_raw_counter     = 0;
    uint32_t     t_counter            = 0;
    std::time_t  last_time;    

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
            

            if(last_time != ts.tv_sec){
                printf("%x %x %x %x %d %d %d %d",raw_data[1],expected_next_count,raw_data[32],raw_data[32],ts.tv_sec,ts.tv_nsec,raw_counter-last_raw_counter,size);
                last_raw_counter = raw_counter;
                printf("\n");
            }

            last_time = ts.tv_sec;
        
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
