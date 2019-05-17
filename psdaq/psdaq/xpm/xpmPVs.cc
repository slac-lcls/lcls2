#include <string>
#include <sstream>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include <cpsw_error.h>

#include "psdaq/cphw/Reg.hh"

#include "psdaq/xpm/Module.hh"
#include "psdaq/xpm/PVStats.hh"
#include "psdaq/xpm/PVCtrls.hh"
#include "psdaq/xpm/PVPStats.hh"
#include "psdaq/xpm/PVPCtrls.hh"
#include "psdaq/xpm/XpmSequenceEngine.hh"

#include "psdaq/epicstools/EpicsPVA.hh"
#include "psdaq/epicstools/PVMonitorCb.hh"

#include "psdaq/service/Routine.hh"
#include "psdaq/service/Semaphore.hh"
#include "psdaq/service/Task.hh"
#include "psdaq/service/Timer.hh"

#include <cpsw_error.h>  // To catch a CPSW exception and continue

using Pds_Epics::EpicsPVA;
using Pds_Epics::PVMonitorCb;
using Pds::Xpm::CoreCounts;
using Pds::Xpm::L0Stats;
using Pds::Xpm::MmcmPhaseLock;
using std::string;

extern int optind;

namespace Pds {
  namespace Xpm {

    //class StatsTimer;

    //static StatsTimer* theTimer = NULL;

    static void sigHandler( int signal )
    {
      //if (theTimer)  theTimer->cancel();

      ::exit(signal);
    }

    class StatsTimer : public Timer {
    public:
      StatsTimer(Module& dev);
      ~StatsTimer() { _task->destroy(); }
    public:
      void allocate(const char* module_prefix,
                    const char* partition_prefix,
                    unsigned    shelf);
      void start   ();
      void cancel  ();
      void expired ();
      Task* task() { return _task; }
      //      unsigned duration  () const { return 1000; }
      unsigned duration  () const { return 1010; }  // 1% error on timer
      unsigned repetitive() const { return 1; }
    public:  // PVMonitorCb interface
      void     updated   ();
    private:
      void     _allocate();
    private:
      Module&    _dev;
      Semaphore  _sem;
      Task*      _task;
      string     _module_prefix;
      string     _partition_prefix;
      unsigned   _shelf;
      EpicsPVA*  _partPV;
      EpicsPVA*  _paddrPV;
      EpicsPVA*  _fwBuildPV;
      EpicsPVA*  _mmcmPV[4];
      EpicsPVA*  _rxAlignPV[2];
      L0Stats    _s    [Pds::Xpm::Module::NPartitions];
      PVPStats*  _pvps [Pds::Xpm::Module::NPartitions];
      PVStats    _pvs;
      PVCtrls    _pvc;
      PVPCtrls*  _pvpc [Pds::Xpm::Module::NPartitions];
      friend class PvAllocate;
      unsigned   _nmmcm;
    };

    class PvAllocate : public Routine {
    public:
      PvAllocate(StatsTimer& parent) : _parent(parent) {}
    public:
      void routine() { _parent._allocate(); delete this; }
    private:
      StatsTimer& _parent;
    };

  };
};

static void fillMmcm( EpicsPVA*& pv, MmcmPhaseLock& mmcm ) {
  if (pv && pv->connected()) {
    unsigned w = 2048;
    pvd::shared_vector<int> vec(w+1);
    vec[0] = mmcm.delayValue;
    for(unsigned j=1; j<=w; j++) {
      mmcm.ramAddr = j;
      unsigned q = mmcm.ramData;
      vec[j] = q;
    }
    pv->putFromVector<int>(freeze(vec));
    pv = 0;
  }
}

static void fillRxAlignPV( EpicsPVA*& pv, Pds::Cphw::GthRxAlign& align ) {
  if (pv && pv->connected()) {
    pvd::shared_vector<int> vec(65);
    unsigned j=0,k=0;
    vec[j++] = align.gthAlignLast &0x7f;
    while(j < 65) {
      unsigned v = align.gthAlign[k++];
      vec[j++] = v&0xff; v>>=8;
      vec[j++] = v&0xff; v>>=8;
      vec[j++] = v&0xff; v>>=8;
      vec[j++] = v&0xff; v>>=8;
    }
    pv->putFromVector<int>(freeze(vec));
    pv = 0;
  }
}

static inline double tdiff(timespec& tm0, timespec& tm1)
{
  return double(tm1.tv_sec - tm0.tv_sec) + 1.e-9*(double(tm1.tv_nsec)-double(tm0.tv_nsec));
}

using namespace Pds;
using namespace Pds::Xpm;

StatsTimer::StatsTimer(Module& dev) :
  _dev       (dev),
  _sem       (Semaphore::FULL),
  _task      (new Task(TaskObject("PtnS"))),
  _pvs       (dev,_sem),
  _pvc       (dev,_sem)
{
}

void StatsTimer::allocate(const char* module_prefix,
                          const char* partition_prefix,
                          unsigned    shelf)
{
  _module_prefix    = module_prefix;
  _partition_prefix = partition_prefix;
  _shelf            = shelf;

  _task->call(new PvAllocate(*this));
}

void StatsTimer::_allocate()
{
  //  Wait for module to become ready
  { 
    _nmmcm = 0;
    if (Module::feature_rev()>0) {
      _nmmcm = 4;
      while(!_dev._mmcm_amc.ready()) {
        printf("Waiting for XTPG phase lock: ready=[%c/%c]%c%c%c%c\n",
               _dev._usTiming.RxRstDone==1 ? 'T':'F',
               _dev._cuTiming.RxRstDone==1 ? 'T':'F',
               _dev._mmcm[0].ready() ? 'T':'F',
               _dev._mmcm[1].ready() ? 'T':'F',
               _dev._mmcm[2].ready() ? 'T':'F',
               _dev._mmcm_amc.ready()? 'T':'F');
        sleep(1);
      }
    }
  }

  _pvs.allocate(_module_prefix);
  _pvc.allocate(_module_prefix);

  for(unsigned i=0; i<Pds::Xpm::Module::NPartitions; i++) {
    std::stringstream ostr,dtstr;
    ostr  << _partition_prefix << ":" << i;
    dtstr << _module_prefix << ":PART:" << i;
    _pvps[i] = new PVPStats(_dev,i);
    _pvps[i]->allocate(ostr.str(),dtstr.str());
    _pvpc[i] = new PVPCtrls(_dev,_sem,*_pvps[i],_shelf,i);
    _pvpc[i]->allocate(ostr.str());
  }

  { std::stringstream ostr;
    ostr << _module_prefix << ":PAddr";
    _paddrPV = new EpicsPVA(ostr.str().c_str()); }

  { std::stringstream ostr;
    ostr << _module_prefix << ":FwBuild";
    printf("fwbuildpv: %s\n", ostr.str().c_str());
    _fwBuildPV = new EpicsPVA(ostr.str().c_str(),256);  }

  if (Module::feature_rev()>0) {
    for(unsigned i=0; i<4; i++) {
      std::stringstream ostr;
      ostr << _module_prefix << ":XTPG:MMCM" << i;
      printf("mmcmpv[%d]: %s\n", i, ostr.str().c_str());
      _mmcmPV[i] = new EpicsPVA(ostr.str().c_str());  
    }

    for(unsigned i=0; i<2; i++) {
      std::stringstream ostr;
      ostr << _module_prefix << (i==0 ? ":Us":":Cu") << ":RxAlign";
      printf("rxalignpv[%d]: %s\n", i, ostr.str().c_str());
      _rxAlignPV[i] = new EpicsPVA(ostr.str().c_str());  
    }
  }
  else {
    for(unsigned i=0; i<2; i++)
      _rxAlignPV[i]=0;
  }
}

void StatsTimer::start()
{
  //theTimer = this;

  _dev._usTiming.resetStats();
  _dev._cuTiming.resetStats();

  //  _pvs.begin();
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
  try {
    _pvs.update();

    for(unsigned i=0; i<Pds::Xpm::Module::NPartitions; i++) {
      if (_pvpc[i]->enabled()) {
        _sem.take();
        try {
          _dev.setPartition(i);
          _pvps[i]->update();
        } catch (CPSWError& e) {
          printf("Caught exception %s\n", e.what());
        }
        _sem.give();
      }
    }
  } catch (CPSWError& e) { printf("cpsw exception %s\n",e.what()); }

  //  _pvc.dump();

  if (_paddrPV && _paddrPV->connected()) {
    _paddrPV->putFrom<unsigned>(_dev._paddr);
    _paddrPV = 0;
  }

  if (_fwBuildPV && _fwBuildPV->connected()) {
    std::string bld = _dev._version.buildStamp();
    printf("fwBuild: %s\n",bld.c_str());
    _fwBuildPV->putFrom<std::string>(bld.c_str());
    _fwBuildPV = 0;
  }

  for(unsigned i=0; i<_nmmcm; i++) 
    fillMmcm(_mmcmPV[i], i<3 ? _dev._mmcm[i] : _dev._mmcm_amc);

  for(unsigned i=0; i<2; i++)
    fillRxAlignPV(_rxAlignPV[i], i==0 ? _dev._usGthAlign : _dev._cuGthAlign);
}


void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr> (default: 10.0.2.102)\n"
         "         -p <port>    (default: 8193)\n"
         "         -P <prefix>  (default: DAQ:LAB2\n");
}

int main(int argc, char** argv)
{
  extern char* optarg;

  int c;
  bool lUsage = false;

  const char* ip = "10.0.2.102";
  unsigned short port = 8193;
  const char* prefix = "DAQ:LAB2";
  char module_prefix[64];
  char partition_prefix[64];

  while ( (c=getopt( argc, argv, "a:p:P:h")) != EOF ) {
    switch(c) {
    case 'a':
      ip = optarg;
      break;
    case 'p':
      port = strtoul(optarg,NULL,0);
      break;
    case 'P':
      prefix = optarg;
      break;
    case '?':
    default:
      lUsage = true;
      break;
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

  unsigned shelf = strtoul(strchr(strchr(ip,'.')+1,'.')+1,NULL,10);
  //  Crate number is "c" in IP a.b.c.d
  sprintf(module_prefix   ,"%s:XPM:%u" ,
          prefix,
          shelf);
  sprintf(partition_prefix,"%s:PART",prefix);

  Pds::Cphw::Reg::set(ip, port, 0);

  Module* m = Module::locate();
  m->init();
  //  Assign transmit link ID
  { unsigned v = (0xff0000 | (shelf<<8) | strtoul(strrchr(ip,'.')+1,NULL,10)) << 4;
    m->_paddr = v;
    printf("Set PADDR to 0x%x\n",v); }

  StatsTimer* timer = new StatsTimer(*m);

  Task* task = new Task(Task::MakeThisATask);

  ::signal( SIGINT, sigHandler );

  timer->allocate(module_prefix,
                  partition_prefix,
                  shelf);
  timer->start();

  task->mainLoop();

  sleep(1);                    // Seems to help prevent a crash in cpsw on exit

  return 0;
}
