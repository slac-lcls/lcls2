//
//  Merge the multicast BLD with the timing stream from a TPR
//
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <string>
#include <signal.h>

#include "psdaq/bld/Client.hh"

#include "AppUtils.hh"

#include "psdaq/epicstools/PVBase.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
using namespace XtcData;

static const Name::DataType xtype[] =
  { Name::UINT8 , // pvBoolean
    Name::INT8  , // pvByte
    Name::UINT16, // pvShort
    Name::INT32 , // pvInt
    Name::INT64 , // pvLong
    Name::UINT8 , // pvUByte
    Name::UINT16, // pvUShort
    Name::UINT32, // pvUInt
    Name::UINT64, // pvULong
    Name::FLOAT , // pvFloat
    Name::DOUBLE, // pvDouble
    Name::CHARSTR, // pvString
  };

namespace Pds_Epics {
  class BldPV : public PVBase {
  public:
    BldPV(const char* channelName) :
      PVBase(channelName) {}
  public:
    unsigned getUID() const { return strtoul(_strct->getStructure()->getID().c_str(),NULL,10); }
    const pvd::StructureConstPtr structure() const { return _strct->getStructure(); }
    XtcData::VarDef getVarDef(size_t& sz) const {
      sz = 0;
      XtcData::VarDef vd;
      const pvd::FieldConstPtrArray& fields = structure()->getFields();
      const pvd::StringArray&        names  = structure()->getFieldNames();
      for(unsigned i=0; i<fields.size(); i++) {
        switch (fields[i]->getType()) {
        case pvd::scalar:
          { const pvd::Scalar* s = static_cast<const pvd::Scalar*>(fields[i].get());
            Name::DataType xt = xtype[s->getScalarType()];
            vd.NameVec.push_back(Name(names[i].c_str(), xt));
            sz += Name::get_element_size(xt);
            break; }
          //        case pvd::scalarArray:
        default:
          throw std::string("PV type ")+pvd::TypeFunc::name(fields[i]->getType())+
            " for field "+names[i]+" not supported";
          break;
        }
      }
      return vd;
    }
    void updated() {}
  };
};

using Pds_Epics::PVBase;
using Pds_Epics::BldPV;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -i <ip interface, name or dotted notation>\n");
  printf("         -N <bld service name>\n");
  printf("         -P <partition>\n");
  printf("         -f <filename>\n");
  printf("         -# <nprint>\n");
}

static uint64_t  event = 0;
static uint64_t  bytes = 0;
static uint64_t  misses = 0;

// METHOD dump IS COMMENTED IN ORDER TO GET RID OF COMPILER WARNING:
//  warning: void dump(XtcData::Xtc*, const char*) defined but not used [-Wunused-function]
//  static void dump(XtcData::Xtc* xtc, const char* title)

//static void dump(XtcData::Xtc* xtc, const char* title)
//{  // dump the xtc
//  printf("-- %s --\n",title);
//  const uint32_t* p = reinterpret_cast<const uint32_t*>(xtc);
//  for(unsigned i=0; i<xtc->extent>>2; i++)
//    printf("%08x%c", p[i], (i&7)==7 ? '\n':' ');
//  if ((xtc->extent&31)!=0)
//    printf("\n");
//}


static void sigHandler(int signal)
{
  psignal(signal, "bld_client received signal");
  ::exit(signal);
}

static Dgram* write_config( NameIndex&       nameIndex,
                            NamesId&         namesId,
                            VarDef&          bldDef,
                            char*            buff,
                            const void*      bufEnd)
{
  timespec tv; clock_gettime(CLOCK_REALTIME,&tv);
  Dgram& dg = *new (buff) Dgram( Transition( Dgram::Event,
                                             TransitionId::Configure,
                                             TimeStamp(tv.tv_sec,tv.tv_nsec),
                                             0 ),
                                 Xtc( TypeId(TypeId::Parent, 0) ) );

  Alg     bldAlg    ("bldAlg", 1, 2, 3);
  Names&  bldNames = *new(dg.xtc, bufEnd) Names(bufEnd,
                                                "mybld", bldAlg, "bld",
                                                "bld1234", namesId, 0);
  bldNames.add(dg.xtc, bufEnd, bldDef);

  nameIndex = NameIndex(bldNames);

  return &dg;
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  const char* filename = 0;
  unsigned intf = 0;
  unsigned partn = 0;
  const char* payName = 0;
  const char* addName = 0;
  const char* prtName = 0;
  int payloadSize = -1;
  bool lverbose = false;
  unsigned nprint = 10;

  while ( (c=getopt( argc, argv, "f:i:N:p:P:v#:")) != EOF ) {
    switch(c) {
    case 'f':
      filename = optarg;
      break;
    case 'i':
      intf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'N':
      payName = strtok(optarg,",");
      addName = strtok(NULL  ,",");
      prtName = strtok(NULL  ,",");
      break;
    case 'p':
      payloadSize = strtoul(optarg,NULL,0);
      break;
    case 'P':
      partn = strtoul(optarg,NULL,0);
      break;
    case 'v':
      lverbose = true;
      break;
    case '#':
      nprint = strtoul(optarg,NULL,0);
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if (!prtName || !intf) {
    usage(argv[0]);
    return -1;
  }

  //
  //  Fetch channel field names from PVA
  //
  BldPV* pvaPayload    = new BldPV (payName);
  PVBase* pvaAddr      = new PVBase(addName);
  PVBase* pvaPort      = new PVBase(prtName);
  // PVBase* pvaAddr      = new PVBase((std::string(bldName)+":ADDR"   ).c_str());
  // PVBase* pvaPort      = new PVBase((std::string(bldName)+":PORT"   ).c_str());

  while(1) {
      if (//pvaPayload   ->ready() &&
        pvaAddr      ->ready() &&
        pvaPort      ->ready())
      break;
    usleep(100000);
  }

  unsigned mcaddr = Psdaq::AppUtils::parse_ip(pvaAddr->getScalarAs<std::string>().c_str());

  printf("Intf/Addr/Port 0x%x/0x%x/0x%x\n",
         intf,
         mcaddr,
         pvaPort->getScalarAs<unsigned>());
  //
  //  Open the bld receiver
  //

  Pds::Bld::Client  bld(intf, 
                        mcaddr,
                        pvaPort->getScalarAs<unsigned>());

  Psdaq::MonitorArgs monitor_args;
  monitor_args.add("Events","Hz" ,event);
  monitor_args.add("Size"  ,"Bps",bytes);
  monitor_args.add("Misses","Hz" ,misses);
  pthread_t thr = Psdaq::AppUtils::monitor(monitor_args);

  struct sigaction sa;
  sa.sa_handler = sigHandler;
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGINT ,&sa,NULL);
  sigaction(SIGABRT,&sa,NULL);
  sigaction(SIGKILL,&sa,NULL);
  sigaction(SIGSEGV,&sa,NULL);

  char* eventb = new char[ 8*1024 ];
  const void* eventEnd = eventb + 8*1024;

  unsigned _id = 0;
  uint64_t ppulseId=0;

  while(1) {

    size_t         payloadSz = payloadSize;
    unsigned       id = 1;

    if (payloadSize < 0) {
        //
        //  Wait for payload description
        //
        id = pvaPayload->getUID();
        while(id == _id) {
            usleep(100);
            id = pvaPayload->getUID();
        }
          
        //
        //  Prepare configure transition payload
        //
        NameIndex      nameIndex;
        NamesId        namesId(0,0);
        VarDef         bldDef = pvaPayload->getVarDef(payloadSz);
    }

    bld.setID(_id = id);

    while(1) {

      //  First, fetch BLD component
      //    uint64_t pulseId = 0;
      uint64_t pulseId = bld.fetch(eventb,payloadSz);
      //  Check if payload ID has changed
      //  Do we need to handle dynamic changes or just stop receiving
      //    data until a reconfigure?
      if (!pulseId) {
          printf("  break\n");
          break;
      }

      event++;
      bytes += payloadSz;
      if (lverbose)
          printf(" 0x%016llx  %lld\n", pulseId, pulseId-ppulseId);

      ppulseId = pulseId;

    }
  }

  pthread_join(thr,NULL);

  return 0;
}

