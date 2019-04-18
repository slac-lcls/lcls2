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

#include "psdaq/pgp/kcu1500/app/Client.hh"
#include "psdaq/bld/Client.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/Xtc.hh"
using namespace XtcData;

//  DescribedData must have a variable-length array!
//#define PAD_DESCDATA

#include "AppUtils.hh"

#include "psdaq/epicstools/PVBase.hh"

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
}

static uint64_t  event = 0;
static uint64_t  bytes = 0;
static uint64_t  misses = 0;

static Pds::Kcu::Client* tpr = 0;

static void dump(XtcData::Xtc* xtc, const char* title)
{  // dump the xtc
  printf("-- %s --\n",title);
  const uint32_t* p = reinterpret_cast<const uint32_t*>(xtc);
  for(unsigned i=0; i<xtc->extent>>2; i++)
    printf("%08x%c", p[i], (i&7)==7 ? '\n':' ');
  if ((xtc->extent&31)!=0)
    printf("\n");
}

static void sigHandler(int signal)
{
  psignal(signal, "bld_client received signal");
  tpr->stop();
  ::exit(signal);
}

static void write_config( NameIndex&       nameIndex,
                          NamesId&         namesId,
                          VarDef&          bldDef,
                          FILE*            fout )
{
  char* configBuff = new char[1024*1024];
  memset(configBuff, 0, 1024*1024);
  timespec tv; clock_gettime(CLOCK_REALTIME,&tv);
  Dgram& dg = *new (configBuff) Dgram( Transition( Sequence( Sequence::Event,
                                                             TransitionId::Configure,
                                                             TimeStamp(tv.tv_sec,tv.tv_nsec),
                                                             PulseId(0,0)), 0 ),
                                       Xtc( TypeId(TypeId::Parent, 0) ) );
    
  Alg     bldAlg    ("bldAlg", 1, 2, 3);
  Names&  bldNames = *new(dg.xtc) Names("mybld", bldAlg, "bld",
                                        "bld1234", namesId, 0);
  bldNames.add(dg.xtc, bldDef);

  nameIndex = NameIndex(bldNames);

  if (fout) {
    fwrite(&dg,sizeof(dg)+dg.xtc.sizeofPayload(),1,fout);
  }

  delete[] configBuff;
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  const char* filename = 0;
  unsigned intf = 0;
  unsigned partn = 0;
  const char* bldName = 0;
  bool lverbose = false;
  unsigned nprint = 10;
  
  while ( (c=getopt( argc, argv, "f:i:N:P:v#:")) != EOF ) {
    switch(c) {
    case 'f':
      filename = optarg;
      break;
    case 'i':
      intf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'N':
      bldName = optarg;
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

  if (!bldName || !intf) {
    usage(argv[0]);
    return -1;
  }

  //
  //  Open the timing receiver
  //
  tpr = new Pds::Kcu::Client("/dev/datadev_0");

  //
  //  Fetch channel field names from PVA
  //
  BldPV* pvaPayload    = new BldPV ((std::string(bldName)+":PAYLOAD").c_str());
  PVBase* pvaAddr      = new PVBase((std::string(bldName)+":ADDR"   ).c_str());
  PVBase* pvaPort      = new PVBase((std::string(bldName)+":PORT"   ).c_str());

  while(1) {
    if (pvaPayload   ->connected() &&
        pvaAddr      ->connected() &&
        pvaPort      ->connected())
      break;
    usleep(100000);
  }

  printf("Intf/Addr/Port 0x%x/0x%x/0x%x\n", 
         intf,
         pvaAddr->getScalarAs<unsigned>(), 
         pvaPort->getScalarAs<unsigned>());
  //
  //  Open the bld receiver
  //
  Pds::Bld::Client  bld(intf, pvaAddr->getScalarAs<unsigned>(), pvaPort->getScalarAs<unsigned>());

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

  //
  //  Configure : determine structure of data
  //
  FILE* fout = filename ? fopen(filename,"w") : 0;

  char* eventb = new char[ 8*1024 ];
  tpr->start(partn);

  unsigned _id = 0;

  while(1) {

    //
    //  Wait for payload description
    //
    unsigned id = pvaPayload->getUID();
    while(id == _id) {
      usleep(100);
      id = pvaPayload->getUID();
    }
    bld.setID(_id = id);
    
    NameIndex      nameIndex;
    NamesId        namesId(0,0);
    size_t         payloadSz;
    VarDef         bldDef = pvaPayload->getVarDef(payloadSz);

#ifdef PAD_DESCDATA
    unsigned paddingIndex = bldDef.NameVec.size();
    bldDef.NameVec.push_back(Name("padding", Name::UINT32,1));
    size_t xtcPayloadSz = payloadSz+sizeof(uint32_t);
#else
    size_t xtcPayloadSz = payloadSz;
#endif

    write_config(nameIndex, namesId, bldDef, fout);

    Dgram* dgram = new (eventb) Dgram( Transition( Sequence( Sequence::Event,
                                                             TransitionId::L1Accept,
                                                             TimeStamp(0,0),
                                                             PulseId(0,0)), 0 ),
                                       Xtc( TypeId(TypeId::Parent, 0) ) );
    
    Xtc&   xtc   = dgram->xtc;

    DescribedData desc(xtc, nameIndex, namesId);
    desc.set_data_length(xtcPayloadSz);
#ifdef PAD_DESCDATA
    unsigned shape[MaxRank] = {1,1};
    desc.set_array_shape(paddingIndex, shape);
#endif

    uint64_t ppulseId=0;

    while(1) {

      //  First, fetch BLD component
      //    uint64_t pulseId = 0;
      uint64_t pulseId = bld.fetch((char*)desc.data(),payloadSz);
      //  Check if payload ID has changed
      //  Do we need to handle dynamic changes or just stop receiving
      //    data until a reconfigure?
      if (!pulseId)
        break;

      if (lverbose && nprint) {
        printf("bld pid 0x%llx\n", pulseId);
        nprint--;
      }
    
      //  Second, fetch header (should already be waiting)
      const XtcData::Transition* tr = tpr->advance(pulseId);
      //const XtcData::Transition* tr = 0;

      if (tr) {
        ppulseId = pulseId;

        new (dgram) Transition(*tr);

        if (fout)
          fwrite(dgram, sizeof(*dgram)+dgram->xtc.sizeofPayload(), 1, fout);

        event++;
        bytes += sizeof(XtcData::Dgram)+payloadSz;
        if (lverbose)
          printf(" %9u.%09u %016lx extent 0x%x payload %08x %08x...\n",
                 dgram->seq.stamp().seconds(),
                 dgram->seq.stamp().nanoseconds(),
                 dgram->seq.pulseId().value(),
                 dgram->xtc.extent,
                 reinterpret_cast<uint32_t*>(dgram->xtc.payload())[0],
                 reinterpret_cast<uint32_t*>(dgram->xtc.payload())[1]);
      }
      else {
        //      if (misses++ > 100)
        //        exit(1);

        if (nprint) {
          printf("Miss: %016lx  prev %016lx\n",
                 pulseId, ppulseId);
          nprint--;
        }
      }
    }
  }

  pthread_join(thr,NULL);

  return 0;
}

