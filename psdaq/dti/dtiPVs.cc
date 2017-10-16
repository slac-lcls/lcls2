#include <string>
#include <sstream>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include "psdaq/cphw/Reg.hh"

#include "psdaq/dti/Module.hh"
#include "psdaq/dti/PVStats.hh"
#include "psdaq/dti/PVCtrls.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Timer.hh"


extern int optind;

namespace Pds {
  namespace Dti {

    //class StatsTimer;

    //static StatsTimer* theTimer = NULL;

    static void sigHandler( int signal )
    {
      //if (theTimer)  theTimer->cancel();

      ::exit(signal);
    }

    class PvAllocate : public Routine {
    public:
      PvAllocate(PVStats& pvs,
                 PVCtrls& pvc,
                 char* partition,
                 Module& m) :
        _pvs(pvs), _pvc(pvc), _partition(partition), _m(m) {}
    public:
      void routine() {
        std::ostringstream o;
        o << "DAQ:" << _partition;
        std::string pvbase = o.str();
        _pvs.allocate(pvbase);
        _pvc.allocate(pvbase);
        delete this;
      }
    private:
      PVStats&    _pvs;
      PVCtrls&    _pvc;
      std::string _partition;
      Module&     _m;
    };

    class StatsTimer : public Timer {
    public:
      StatsTimer(Module& dev);
      ~StatsTimer() { _task->destroy(); }
    public:
      void allocate(char* partition);
      void start   ();
      void cancel  ();
      void expired ();
      Task* task() { return _task; }
      //      unsigned duration  () const { return 1000; }
      unsigned duration  () const { return 1010; }  // 1% error on timer
      unsigned repetitive() const { return 1; }
    private:
      Module&  _dev;
      Task*    _task;
      Stats    _s;
      timespec _t;
      PVStats  _pvs;
      PVCtrls  _pvc;
    };
  };
};

using namespace Pds;
using namespace Pds::Dti;

StatsTimer::StatsTimer(Module& dev) :
  _dev      (dev),
  _task     (new Task(TaskObject("PtnS"))),
  _pvc      (dev)
{
}

void StatsTimer::allocate(char* partition)
{
  _task->call(new PvAllocate(_pvs, _pvc, partition, _dev));
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
  timespec t; clock_gettime(CLOCK_REALTIME,&t);
  Stats    s = _dev.stats();
  double dt = double(t.tv_sec-_t.tv_sec)+1.e-9*(double(t.tv_nsec)-double(_t.tv_nsec));
  _pvs.update(s,_s,dt);
  _s=s;
  _t=t;
}


void usage(const char* p) {
  printf("Usage: %s [-a <IP addr (dotted notation)>] [-p <port>] addr <value>\n",p);
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  bool lUsage = false;

  const char* ip = "10.0.2.103";
  unsigned short port = 8192;
  char* partition = NULL;

  while ( (c=getopt( argc, argv, "a:p:P:h")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg;
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case 'P':
      partition = optarg;
      break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (partition==NULL) {
    printf("%s: partition required\n",argv[0]);
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

  Pds::Cphw::Reg::set(ip, port, 0);

  Module* m = Module::locate();

  StatsTimer* timer = new StatsTimer(*m);

  Task* task = new Task(Task::MakeThisATask);

  ::signal( SIGINT, sigHandler );

  timer->allocate(partition);
  timer->start();

  task->mainLoop();

  sleep(1);                    // Seems to help prevent a crash in cpsw on exit

  return 0;
}
