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
#include <arpa/inet.h>
#include <sys/socket.h>

#include "psdaq/bld/Client.hh"
#include "psdaq/bld/Server.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include "AppUtils.hh"

#include "psdaq/epicstools/PVBase.hh"

static int setup_mc(unsigned addr, unsigned port, unsigned interface);

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

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -i <ip interface, name or dotted notation>\n");
  printf("         -I <bld service name>\n");
  printf("         -O <bld service name>\n");
  printf("         -o <events delay>\n");
}

using Pds_Epics::PVBase;
using Pds_Epics::BldPV;

static uint64_t  event = 0;
static uint64_t  bytes = 0;

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  unsigned intf = 0;
  const char* bldInput = 0;
  const char* bldOutput = 0;
  bool lverbose = false;
  unsigned nprint = 10;
  unsigned delay = 0;
  
  while ( (c=getopt( argc, argv, "i:I:o:O:v#:")) != EOF ) {
    switch(c) {
    case 'i':
      intf = Psdaq::AppUtils::parse_interface(optarg);
      break;
    case 'I':
      bldInput = optarg;
      break;
    case 'o':
      delay = strtoul(optarg,NULL,0);
      break;
    case 'O':
      bldOutput = optarg;
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

  if (!bldInput || !bldOutput || !intf) {
    usage(argv[0]);
    return -1;
  }

  //
  //  Fetch channel field names from PVA
  //
  BldPV*  inpPayload   = new BldPV ((std::string(bldInput)+":PAYLOAD").c_str());
  PVBase* inpAddr      = new PVBase((std::string(bldInput)+":ADDR"   ).c_str());
  PVBase* inpPort      = new PVBase((std::string(bldInput)+":PORT"   ).c_str());

  while(1) {
    if (inpPayload   ->connected() &&
        inpAddr      ->connected() &&
        inpPort      ->connected())
      break;
    usleep(100000);
  }

  PVBase* outAddr      = new PVBase((std::string(bldOutput)+":ADDR"   ).c_str());
  PVBase* outPort      = new PVBase((std::string(bldOutput)+":PORT"   ).c_str());

  while(1) {
    if (outAddr      ->connected() &&
        outPort      ->connected())
      break;
    usleep(100000);
  }


  printf("Intf/Addr/Port 0x%x/0x%x/0x%x  : Addr/Port 0x%x/0x%x\n", 
         intf,
         inpAddr->getScalarAs<unsigned>(), 
         inpPort->getScalarAs<unsigned>(),
         outAddr->getScalarAs<unsigned>(), 
         outPort->getScalarAs<unsigned>());
  //
  //  Open the bld receiver
  //
  Pds::Bld::Client  input(intf, 
                          inpAddr->getScalarAs<unsigned>(), 
                          inpPort->getScalarAs<unsigned>());

  Psdaq::MonitorArgs monitor_args;
  monitor_args.add("Events","Hz" ,event);
  monitor_args.add("Size"  ,"Bps",bytes);
  pthread_t thr = Psdaq::AppUtils::monitor(monitor_args);

  unsigned _id = 0;

  int fd_mc = setup_mc(outAddr->getScalarAs<unsigned>(),
                       outPort->getScalarAs<unsigned>(),
                       intf);

  Pds::Bld::Server output(fd_mc);

  while(1) {
    //
    //  Wait for payload description
    //
    unsigned id = inpPayload->getUID();
    while(id == _id) {
      usleep(100);
      id = inpPayload->getUID();
    }
    input.setID(_id = id);

    size_t payloadSz;
    inpPayload->getVarDef(payloadSz);

    char* payload = new char[payloadSz];

    while(1) {

      //  First, fetch INPUT component
      //    uint64_t pulseId = 0;
      uint64_t pulseId = input.fetch(payload,payloadSz);

      if (lverbose && nprint) {
        printf("input pid 0x%lx\n", pulseId);
        nprint--;
      }

      if (delay)
        delay--;
      else {
        output.publish( pulseId, 0,
                        payload, payloadSz);
        event++;
        bytes += payloadSz;
      }
    }
  }

  pthread_join(thr,NULL);

  return 0;
}

int setup_mc(unsigned addr, unsigned port, unsigned interface)
{
  printf("setup_mc %x/%u %x\n",addr,port,interface);

  int fd_mc;

  fd_mc = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_mc < 0) {
    perror("Open mcast socket");
    return -1;
  }

  sockaddr_in saddr_mc;
  saddr_mc.sin_family      = PF_INET;
  saddr_mc.sin_addr.s_addr = htonl(addr);
  saddr_mc.sin_port        = htons(port);
    
  int y=1;
  if(setsockopt(fd_mc, SOL_SOCKET, SO_BROADCAST, (char*)&y, sizeof(y)) == -1) {
    perror("set broadcast");
    return -1;
  }

  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(interface);
  sa.sin_port = htons(0);
  printf("Binding to %x.%u\n", ntohl(sa.sin_addr.s_addr),ntohs(sa.sin_port));
  if (::bind(fd_mc, (sockaddr*)&sa, sizeof(sa)) < 0) {
    perror("bind");
    return -1;
  }

  if (connect(fd_mc, (sockaddr*)&saddr_mc, sizeof(saddr_mc)) < 0) {
    perror("Error connecting UDP mcast socket");
    return -1;
  }

  { in_addr addr;
    addr.s_addr = htonl(interface);
    if (setsockopt(fd_mc, IPPROTO_IP, IP_MULTICAST_IF, (char*)&addr,
                   sizeof(in_addr)) < 0) {
      perror("set ip_mc_if");
      return -1;
    }
  }

  return fd_mc;
}

