#include <string>
#include <sstream>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include "Module134.hh"
#include "ChipAdcCore.hh"
#include "PV134Stats.hh"
#include "PV134Ctrls.hh"

#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/Xvc.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Timer.hh"

#include "psdaq/epicstools/EpicsPVA.hh"

#include "psdaq/app/AppUtils.hh"

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
      StatsTimer(Module134& dev);
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

StatsTimer::StatsTimer(Module134& dev) :
  _dev      (dev),
  _task     (new Task(TaskObject("PtnS"))),
  _pvs      (dev),
  _pvc      (dev, *_task)
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
  printf("         -E           (tolerate errors)\n");
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  bool lUsage = false;

  const char* dev    = 0;
  const char* prefix = "DAQ:LAB2:HSD";
  bool lAbortOnErr = true;

  while ( (c=getopt( argc, argv, "d:EP:Ih")) != EOF ) {
    switch(c) {
    case 'd':
      dev    = optarg;      break;
    case 'E':
      lAbortOnErr = false;  break;
    case 'P':
      prefix = optarg;      break;
    case '?':
    default:
      lUsage = true;      break;
    }
  }

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

  int fd = open(dev, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module134* m = Module134::create(fd);
  m->dumpMap();

  std::string buildStamp = m->version().buildStamp();
  printf("BuildStamp: %s\n",buildStamp.c_str());
  
  for(unsigned i=0; i<2; i++) {
    std::string sprefix(prefix);
    sprefix += (i==0) ? ":A:FWBUILD" : ":B:FWBUILD";
    Pds_Epics::EpicsPVA pvBuild(sprefix.c_str());
    while(!pvBuild.connected())
      usleep(1000);
    pvBuild.putFrom(buildStamp); 
  }

  m->setup_timing();
  m->setup_jesd(lAbortOnErr);

  m->set_local_id(strtoul(dev+strlen(dev)-2,NULL,16));

  { unsigned upaddr = m->remote_id();
    std::string paddr = Psdaq::AppUtils::parse_paddr(upaddr);
    for(unsigned i=0; i<2; i++) {
      std::string sprefix(prefix);
      sprefix += ":"+std::string(1,'A'+i)+":PADDR";
      { Pds_Epics::EpicsPVA pvPaddr(sprefix.c_str());
        while(!pvPaddr.connected())
          usleep(1000);
        pvPaddr.putFrom(paddr); }
      sprefix += "_U";
      { Pds_Epics::EpicsPVA pvPaddr(sprefix.c_str());
        while(!pvPaddr.connected())
          usleep(1000);
        pvPaddr.putFrom(upaddr); }
    }
    printf("paddr [0x%x] [%s]\n", upaddr, paddr.c_str());
  }

  StatsTimer* timer = new StatsTimer(*m);

  ::signal( SIGINT, sigHandler );

  timer->allocate(prefix);
  timer->start();

  //  Pds::Mmhw::Xvc::launch( m->xvc(), false );
  while(1)
    sleep(1);                    // Seems to help prevent a crash in cpsw on exit

  return 0;
}
