#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include <stdio.h>
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "AxiBatcherParser.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <ctime>
#include <Python.h>

#define MAX_RET_CNT_C 1000

using namespace XtcData;

static int        fd;
std::atomic<bool> terminate;

const  unsigned   BUFSIZE                = 1024*1024*32;
static char       config_buf[BUFSIZE];
static char       dgram_buf[BUFSIZE];

enum {ConfigNamesIndex, EventNamesIndex};

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

class TTDef : public VarDef
{
public:
    enum index {
        data
    };
    TTDef()
    {
        Alg alg("timetool", 0, 0, 1);
        NameVec.push_back({"data", Name::UINT8, 1});
    }
} TTDef;

Dgram& createTransition(TransitionId::Value transId,
                        unsigned& timestamp_val) {
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    struct timeval tv;

    tv.tv_sec = 0;
    tv.tv_usec = timestamp_val;
    timestamp_val++;

    Transition tr(Dgram::Event, transId, TimeStamp(tv.tv_sec, tv.tv_usec), env);
    return *new(dgram_buf) Dgram(tr, Xtc(tid));
}

unsigned dmaDest(unsigned lane, unsigned vc)
{
    return (lane<<8) | vc;
}

void int_handler(int dummy)
{
    terminate.store(true, std::memory_order_release);
}

int toggle_acquisition(int x)
{
    printf("starting prescaler config testing  \n");

    PyObject *pName, *pModule, *pFunc;

    Py_Initialize();

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

    Py_XDECREF(pModule);
    Py_XDECREF(pName);

    printf("ending prescaler config testing \n ");

    return 0;

}

int tt_config(int x,NamesLookup &namesLookup,FILE *xtcFile)
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

    char const* m_connect_json_str = "\{\"body\": \{\"control\": \{\"0\": \{\"control_info\": \{\"instrument\": \"TST\", \"cfg_dbase\": \"mcbrowne:psana@psdb-dev:9306/sioanDB\" } } } } }" ;

    PyObject* mybytes = PyObject_CallFunction(pFunc,"sssi",m_connect_json_str,"BEAM", "tmotimetool",0);
    check(mybytes);

    Py_XDECREF(pFunc);


     if (PyErr_Occurred()){
                PyErr_Print();
        }

    //***********************************
    //***** converting json to xtc ******
    //***********************************
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);

    // convert to json to xtc
    unsigned nodeId = 0; //Fix me for real drp
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    unsigned len = Pds::translateJson2Xtc(json, config_buf, configNamesId);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    //***********************************
    //***** writing xtc to buffer  ******
    //***********************************
    unsigned timestamp_val = 0;
    Dgram& config = createTransition(TransitionId::Configure,timestamp_val);    //what are the arguments here?

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)config_buf;                                           //config buf is global 
    Xtc& xtc     = config.xtc;                                                  //
    memcpy(xtc.next(),jsonxtc.payload(),jsonxtc.sizeofPayload());               //this line copies jsonxtc to the xtc object.
    xtc.alloc(jsonxtc.sizeofPayload());

    // append the metadata; which algorithm is needed to interpret bytes, the detector type, etc...
    Alg ttAlg("ttalg", 0, 0, 1);

    NamesId eventNamesId(nodeId,EventNamesIndex);


    unsigned segment = 0;
    Names& eventNames = *new(xtc) Names("tmott", ttAlg, "ttdet", "tt_detector_identification_placeholder", eventNamesId, segment);
    eventNames.add(xtc, TTDef);
    namesLookup[eventNamesId] = NameIndex(eventNames);

    //***********************************
    //***** writing xtc to file    ******
    //***********************************


    //xtc file will be corrupted if this is not written.
    if (fwrite(&config, sizeof(config) + config.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }



    Py_XDECREF(pModule);    

    printf("ending prescaler config testing \n ");

    return 0;

}


int main(int argc, char* argv[])
{

    
    printf("starting main \n");

    FILE* xtcFile = fopen("timetoolconfig.xtc2", "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
    }


    NamesLookup namesLookup;
    
    int c, channel;
    int write_interval = 1;

    timespec ts; 

    channel = 0;
    std::string device;
    while((c = getopt(argc, argv, "c:d:tf")) != EOF) {
        switch(c) {
            case 'd':
                device = optarg;
                break;
            case 'c':
                channel = atoi(optarg);
                break;
            case 't':
                   printf("entering tt config \n");
                   tt_config(0,namesLookup,xtcFile);
                   break;
                   //toggle_acquisition(0);                     
            case 'f':
                   printf("doing fast write \n");
                   write_interval = 0;
                   break;
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


    int32_t                 dmaRet[MAX_RET_CNT_C];
    uint32_t                dmaIndex[MAX_RET_CNT_C];
    uint32_t                dmaDest[MAX_RET_CNT_C];

    uint8_t                *raw_data;

    uint8_t                 expected_next_count          = 0;

    uint32_t                raw_counter                  = 0;
    uint32_t                last_raw_counter             = 0;

    std::time_t             last_time;    
    std::vector<uint8_t>    raw_vector;

    eventBuilderParser      my_frame;

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
            


            raw_vector = std::vector<uint8_t> (raw_data,raw_data+size);
            my_frame.load_frame( raw_vector );
            my_frame.parse_array();
            
            if(ts.tv_sec - last_time  >= write_interval){
                
                

                my_frame.print_frame();


                printf("%x %x %x %x %ld elapsed time = %ld number of shots = %d %d \n",raw_data[1],expected_next_count,raw_data[32],raw_data[32],ts.tv_sec,ts.tv_sec-last_time,raw_counter-last_raw_counter,size);
                last_raw_counter = raw_counter;

                //*****************************
                //**** writing xtc to disk ****
                //*****************************
                //refer to this link https://docs.google.com/presentation/d/1KsTXafudKoDTHlqAQXOHSZRLQ1x734kTlrVniV3K-NI/edit#slide=id.g64c80c79d8_0_69 
                //for a visual layout of how the data structures below create the object that gets stored to xtc.
                unsigned nodeId = 0;
                unsigned env    = 0;


                Transition tr(Dgram::Event, TransitionId::L1Accept, TimeStamp(ts.tv_sec, ts.tv_nsec), env);
                TypeId     tid(TypeId::Parent, 0);
                Dgram&     dgram = *new(dgram_buf) Dgram(tr, Xtc(tid));
                NamesId    eventNamesId(nodeId,EventNamesIndex);
                //this instantiates the dgram.xtc component.  Here's the path dgram takes before it gets written.  dgram is now contained within fex
                CreateData fex(dgram.xtc, namesLookup, eventNamesId);

                unsigned shape[MaxRank];
                shape[0] = size;
                Array<uint8_t> arrayT = fex.allocate<uint8_t>(TTDef::data,shape);   //arrayT is now pointing at dgram.xtc
                for(unsigned i=0; i<shape[0]; i++){
                    arrayT(i) = raw_data[i];                                        //this copies the data from raw_data to arrayT (where arrayT is really dgram.xtc)
                };


                //here finally dgram is written to disk
                if (fwrite(&dgram, sizeof(dgram) + dgram.xtc.sizeofPayload(), 1, xtcFile) != 1) {
                    printf("Error writing to output xtc file.\n");
                }

                printf("wrote to disk \n");



            }

            my_frame.clear();



            last_time = ts.tv_sec;
        
            raw_counter = raw_counter + 1;

            if(expected_next_count != raw_data[1]){
                printf("Dropped shot. raw_counter = %d, expected_next_counter = %d ",raw_data[1],expected_next_count);
                printf("\n");

                //expected_next_count = (raw_data[1]+1)%256;
            }

            expected_next_count = (raw_data[1]+1)%256;

            //Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(dmaBuffers[index]);
            //XtcData::TransitionId::Value transition_id = event_header->service();

        }
	    if ( ret > 0 ) dmaRetIndexes(fd, ret, dmaIndex);
	    //sleep(0.1)
    }
    fclose(xtcFile);
    printf("finished\n");
}
