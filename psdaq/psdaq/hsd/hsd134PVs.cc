#include <string>
#include <sstream>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <Python.h>

#include "psdaq/aes-stream-drivers/AxiVersion.h"
#include "Module134.hh"
#include "ChipAdcCore.hh"
#include "Pgp3.hh"
#include "PV134Stats.hh"
#include "PV134Ctrls.hh"
#include "I2c134.hh"
#include "Fmc134Cpld.hh"

#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/Reg.hh"
//#include "psdaq/mmhw/Xvc.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Timer.hh"
#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

#include "psdaq/epicstools/EpicsPVA.hh"

#include "psdaq/app/AppUtils.hh"

#include <rogue/hardware/axi/AxiStreamDma.h>
#include <rogue/protocols/xilinx/Xvc.h>
#include <rogue/Helpers.h>

extern int optind;

using Pds::Task;

namespace Pds {

    namespace HSD {

        class PvAllocate : public Routine {
        public:
            PvAllocate(PV134Stats& pvs,
                       PV134Ctrls& pvc,
                       const char* prefix) :
                _pvs(pvs), _pvc(pvc), _prefix(prefix)
            { printf("PvAllocate &_pvc %p\n",&_pvc); }
        public:
            void routine() {
                std::ostringstream o;
                o << _prefix;
                std::string pvbase = o.str();
                _pvs.allocate(pvbase);
                _pvc.allocate(pvbase);
                delete this;
            }
        private:
            PV134Stats&    _pvs;
            PV134Ctrls&    _pvc;
            std::string   _prefix;
        };

        class StatsTimer : public Timer {
        public:
            StatsTimer(Module134& dev, unsigned inputchan);
            ~StatsTimer() { _task->destroy(); }
        public:
            void allocate(const char* prefix);
            void start   ();
            void cancel  ();
            void expired ();
            Task* task() { return _task; }
            unsigned duration  () const { return 1000; }
            //      unsigned duration  () const { return 1010; }  // 1% error on timer
            unsigned repetitive() const { return 1; }
        private:
            Module134&  _dev;
            Task*       _task;  // Serialize all register access through this task
            PV134Stats  _pvs;
            PV134Ctrls  _pvc;
        };
    };
};

using namespace Pds::HSD;

static Module134* reg = NULL;

static void sigHandler( int signal )
{
    if (reg) {
        reg->chip(0).reg.stop();
        reg->chip(1).reg.stop();
    }
    ::exit(signal);
}

StatsTimer::StatsTimer(Module134& dev, unsigned inputchan) :
    _dev      (dev),
    _task     (new Task(TaskObject("PtnS"))),
    _pvs      (dev),
    _pvc      (dev, *_task, inputchan)
{
}

void StatsTimer::allocate(const char* prefix)
{ _task->call(new PvAllocate(_pvs, _pvc, prefix)); }

void StatsTimer::start()
{ Timer::start(); }

void StatsTimer::cancel()
{
    Timer::cancel();
    expired();
}

//
//  Update EPICS PVs
//
void StatsTimer::expired()
{ _pvs.update(); }

void usage(const char* p) {
    printf("Usage: %s -d <device> [options]\n",p);
    printf("Options: -P <prefix>  (default: DAQ:LAB2:HSD)\n");
    printf("         -D <db_url,instrument,obj0,obj1[,L]> : 'L' uploads to database\n");
    printf("         -E           (abort on error)\n");
    printf("         -X <port>    (connect XVC)\n");
}

static PyObject* _check(PyObject* obj)
{
    if (!obj) {
        PyErr_Print();
        throw "**** python error";
    }
    return obj;
}


int main(int argc, char** argv)
{
    extern char* optarg;

    int c;
    bool lUsage = false;

    unsigned xvcPort   = 0;
    const char* dev    = 0;
    const char* prefix = "DAQ:LAB2:HSD";
    bool lInternalTiming = false;
    bool lLoopback = false;
    bool lAbortOnErr = false;
    bool lverbose    = false;
    unsigned    busId  = 0;
    const char* db_args[5] = {0,0,0,0,0};
    std::shared_ptr<rogue::hardware::axi::AxiStreamDma> _dma;
    std::shared_ptr<rogue::protocols::xilinx::Xvc>      _xvc;

    while ( (c=getopt( argc, argv, "d:D:ELP:IX:vh")) != EOF ) {
        switch(c) {
        case 'd':
            dev    = optarg;      break;
        case 'D':
            {
                char* p = optarg;
                unsigned i=0;
                while(p) {
                    db_args[i++] = p;
                    if ((p = strchr(p,','))==NULL)
                        break;
                    *p++ = 0;
                }
            } break;
        case 'E':
            lAbortOnErr = true;   break;
        case 'I':
            lInternalTiming = true;  break;
        case 'L':
            lLoopback = true; break;
        case 'P':
            prefix = optarg;      break;
        case 'X':
            xvcPort = strtoul(optarg,NULL,0); break;
        case 'v':
            lverbose = true;      break;
        case '?':
        default:
            lUsage = true;      break;
        }
    }

    logging::init(argv[0], lverbose ? LOG_DEBUG : LOG_INFO);
    logging::info("logging configured");

    if (!dev) {
        printf("No device specified\n");
        lUsage = true;
    }

    if (optind < argc) {
        printf("%s: invalid argument -- %s\n",argv[0], argv[optind]);
        lUsage = true;
    }

    if (lUsage) {
        usage(argv[0]);
        exit(1);
    }

    if (xvcPort) {
        _dma = rogue::hardware::axi::AxiStreamDma::create(dev,0,1);
        _xvc = rogue::protocols::xilinx::Xvc::create(xvcPort);
        //(*_dma) == _xvc;
        rogueStreamConnectBiDir(_dma, _xvc);
    }

    int fd = open(dev, O_RDWR);
    if (fd<0) {
        perror("Open device failed");
        return -1;
    }

    Pds::Mmhw::Reg::verbose(lverbose);

    Module134* m = Module134::create(fd);
    m->dumpMap();

    std::string buildStamp = m->version().buildStamp();
    printf("BuildStamp: %s\n",buildStamp.c_str());
    unsigned buildVersion = m->version().FpgaVersion;

    std::vector< Pds_Epics::EpicsPVA* > pvaa;

    for(unsigned i=0; i<2; i++) {
        std::string sprefix(prefix);
        sprefix += (i==0) ? ":A:FWBUILD" : ":B:FWBUILD";
        Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
        pvaa.push_back(pv);
        Pds_Epics::EpicsPVA& pvBuild = *pv;
        while(!pvBuild.connected())
            usleep(1000);
        pvBuild.putFrom(buildStamp);
    }

    for(unsigned i=0; i<2; i++) {
        std::string sprefix(prefix);
        sprefix += (i==0) ? ":A:FWVERSION" : ":B:FWVERSION";
        Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
        pvaa.push_back(pv);
        Pds_Epics::EpicsPVA& pvBuild = *pv;
        while(!pvBuild.connected())
            usleep(1000);
        pvBuild.putFrom(buildVersion);
    }

    Py_Initialize();
    PyObject* module  = _check(PyImport_ImportModule("psdaq.hsd.calib")); // returns new reference
    PyObject* pDict   = _check(PyModule_GetDict(module));
    std::string adc_calib[2];
    if (db_args[0]) {    // Fetch calibration
        PyObject* pFunc   = _check(PyDict_GetItemString(pDict, "get_calib"));
        for(unsigned i=0; i<2; i++) {
            PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "ssss",
                                                             db_args[0],
                                                             db_args[1],
                                                             db_args[2+i],
                                                             "CALIB"));
            PyObject* json_bytes = _check(PyUnicode_AsASCIIString(mybytes)); // returns new reference
            adc_calib[i] = (const char*)PyBytes_AsString(json_bytes);
            Py_DECREF(json_bytes);
            Py_DECREF(mybytes);
        }
    }

#if 1
    m->setup_timing(lLoopback);

    //  Get the ADC INPUT selection from the jesdsetup PV and configure now
    unsigned inputchan;
    {   std::string sprefix(prefix);
        sprefix += ":A:RESET";
        Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
        inputchan = pv->getScalarAs<unsigned>("jesdsetup");
        m->setup_jesd(lAbortOnErr,
                      adc_calib[0],
                      adc_calib[1],
                      inputchan,
                      lInternalTiming);
    }
#endif

    if (db_args[4] && db_args[4][0]=='L') {    // Write calibration
        PyObject* pFunc   = _check(PyDict_GetItemString(pDict, "set_calib"));
        for(unsigned i=0; i<2; i++) {
            PyObject_CallFunction(pFunc, "sssss",
                                  db_args[0],
                                  db_args[1],
                                  db_args[2+i],
                                  "CALIB",
                                  adc_calib[i].c_str());
        }
    }
    Py_DECREF(module);
    Py_Finalize();

    busId = strtoul(dev+strlen(dev)-2,NULL,16);
    m->set_local_id(busId);

    //  Name the remote partner on the timing link
    { unsigned upaddr = m->remote_id();
        std::string paddr = Psdaq::AppUtils::parse_paddr(upaddr);
        for(unsigned i=0; i<2; i++) {
            std::string sprefix(prefix);
            sprefix += ":"+std::string(1,'A'+i)+":PADDR";
            Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
            pvaa.push_back(pv);
            { Pds_Epics::EpicsPVA& pvPaddr = *pv;
                while(!pvPaddr.connected())
                    usleep(1000);
                pvPaddr.putFrom(paddr); }
            /*  Do this in PVStats::update
            sprefix += "_U";
            Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
            pvaa.push_back(pv);
            { Pds_Epics::EpicsPVA& pvPaddr = *pv;
                while(!pvPaddr.connected())
                    usleep(1000);
                pvPaddr.putFrom(upaddr); }
            */
        }
        printf("paddr [0x%x] [%s]\n", upaddr, paddr.c_str());
    }

    //  Name the remote partner on the PGP link
    for(unsigned i=0; i<2; i++) {
        unsigned uplink = m->pgp()[i*4]->remoteLinkId();
        std::string sprefix(prefix);
        sprefix += ":"+std::string(1,'A'+i)+":PLINK";
        Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
        pvaa.push_back(pv);
        Pds_Epics::EpicsPVA& pvPaddr = *pv;
        while(!pvPaddr.connected())
            usleep(1000);
        pvPaddr.putFrom(uplink);
        printf("plink [0x%x]\n", uplink);
    }

    unsigned keepRows = m->chip(0).fex._stream[1].info[0] & 0xff;
    for(unsigned i=0; i<2; i++) {
        std::string sprefix(prefix);
        sprefix += (i==0) ? ":A:KEEPROWS" : ":B:KEEPROWS";
        Pds_Epics::EpicsPVA* pv = new Pds_Epics::EpicsPVA(sprefix.c_str());
        pvaa.push_back(pv);
        Pds_Epics::EpicsPVA& pvBuild = *pv;
        while(!pvBuild.connected())
            usleep(1000);
        pvBuild.putFrom(keepRows);
    }

    StatsTimer* timer = new StatsTimer(*m, inputchan);

    ::signal( SIGINT, sigHandler );

    timer->allocate(prefix);
    timer->start();

    //  Cleanup PV references
    usleep(100000);
    for(unsigned i=0; i<pvaa.size(); i++) {
        delete pvaa[i];
    }

    //    Pds::Mmhw::Xvc::launch( &m->xvc(), 11000+busId, false );
    while(1)
        sleep(1);                    // Seems to help prevent a crash in cpsw on exit

    return 0;
}
