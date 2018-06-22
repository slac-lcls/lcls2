#include <string>
#include <sstream>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/PVStats.hh"
#include "psdaq/hsd/PVCtrls.hh"
#include "psdaq/mmhw/AxiVersion.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Timer.hh"


extern int optind;

using Pds::Task;

namespace Pds {

  namespace HSD {

    static Module* reg = NULL;

    static void sigHandler( int signal )
    {
      if (reg) {
        reg->stop();
        reinterpret_cast<QABase*>((char*)reg->reg()+0x80000)->dump();
      }

      ::exit(signal);
    }

    class PvAllocate : public Routine {
    public:
      PvAllocate(PVStats& pvs,
                 PVCtrls& pvc,
                 const char* prefix) :
        _pvs(pvs), _pvc(pvc), _prefix(prefix) {}
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
      PVStats&    _pvs;
      PVCtrls&    _pvc;
      std::string _prefix;
    };

    class StatsTimer : public Timer {
    public:
      StatsTimer(Module& dev);
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
      Module&  _dev;
      Task*    _task;  // Serialize all register access through this task
      PVStats  _pvs;
      PVCtrls  _pvc;
    };
  };
};

using namespace Pds::HSD;

StatsTimer::StatsTimer(Module& dev) :
  _dev      (dev),
  _task     (new Task(TaskObject("PtnS"))),
  _pvs      (dev),
  _pvc      (dev, *_task)
{
}

void StatsTimer::allocate(const char* prefix)
{
  _task->call(new PvAllocate(_pvs, _pvc, prefix));
}

void StatsTimer::start()
{
  //theTimer = this;

  Timer::start();
}

void StatsTimer::cancel()
{
  Timer::cancel();
  expired();
}

//
//  Update EPICS PVs
//
void StatsTimer::expired()
{
  _pvs.update();
}

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <device>  (default: /dev/qadca)\n"
         "         -P <prefix>  (default: DAQ:LAB2:HSD)\n");
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  bool lUsage = false;

  const char* dev    = "/dev/qadca";
  const char* prefix = "DAQ:LAB2:HSD";

  while ( (c=getopt( argc, argv, "d:P:h")) != EOF ) {
    switch(c) {
    case 'd':
      dev    = optarg;      break;
    case 'P':
      prefix = optarg;      break;
    case '?':
    default:
      lUsage = true;      break;
    }
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

  Module* m = Module::create(fd, M3_7);

  std::string buildStamp = m->version().buildStamp();
  printf("BuildStamp: %s\n",buildStamp.c_str());

  StatsTimer* timer = new StatsTimer(*m);

  Task* task = new Task(Task::MakeThisATask);

  ::signal( SIGINT, sigHandler );

  timer->allocate(prefix);
  timer->start();

  task->mainLoop();

  sleep(1);                    // Seems to help prevent a crash in cpsw on exit

  return 0;
}
