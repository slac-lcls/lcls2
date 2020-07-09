//
//  Merge the multicast BLD with the timing stream from a TPR
//
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <signal.h>

#include "psdaq/tpr/Client.hh"
#include "psdaq/tpr/Frame.hh"

//#include "psdaq/epicstools/EpicsPVA.hh"
#include "pva/client.h"
#include "pv/pvTimeStamp.h"

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -r <rate> (0:929kHz,1:71kHz,..)\n");
  printf("         -e <evcode>\n");
  printf("         -p <pvname>\n");
  printf("         -v (verbose)\n");
  printf("Either -r or -e is required\n");
}

namespace pvd = epics::pvData;

static Pds::Tpr::Client* tpr;

static void sigHandler(int signal)
{
  psignal(signal, "pvcam received signal");
  tpr->stop();
  ::exit(signal);
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  const char* pvname = 0;
  int rate   = -1;
  int evcode = -1;
  bool verbose = false;

  while ( (c=getopt( argc, argv, "p:e:r:v"))!=EOF) {
    switch(c) {
    case 'p':
      pvname = optarg;
      break;
    case 'e':
      evcode = strtoul(optarg,NULL,0);
      break;
    case 'r':
      rate = strtoul(optarg,NULL,0);
      break;
    case 'v':
      verbose = true;
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if (!pvname) {
    usage(argv[0]);
    return -1;
  }

  //  Pds_Epics::EpicsPVA pv(pvname);
  pvac::ClientProvider provider("pva");
  pvac::ClientChannel  channel(provider.connect(pvname));
  pvd::PVStructure::const_shared_pointer cpv = channel.get();
  pvd::PVTimeStamp pts;

  const pvd::PVFieldPtrArray& fields = cpv->getPVFields();
  for(unsigned i=0; i<fields.size(); i++) {
    const pvd::PVFieldPtr field = fields[i];
    printf("%s [%s] [%s]\n",
           field->getFieldName().c_str(),
           field->getFullName().c_str(),
           field->getField()->getID().c_str());
    if (field->getFieldName()==std::string("timeStamp"))
      pts.attach(field);
  }

  if (!pts.isAttached()) {
    printf("Timestamp field not found\n");
    exit(1);
  }

  //
  //  Open the timing receiver
  //
  tpr = new Pds::Tpr::Client("/dev/tpra",0,rate>=0);

  struct sigaction sa;
  sa.sa_handler = sigHandler;
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGINT ,&sa,NULL);
  sigaction(SIGABRT,&sa,NULL);
  sigaction(SIGKILL,&sa,NULL);
  sigaction(SIGSEGV,&sa,NULL);

  if (rate >= 0)
    tpr->start(Pds::Tpr::TprBase::FixedRate(rate));
  else if (evcode >= 0)
    tpr->start(Pds::Tpr::TprBase::EventCode(evcode));

  tpr->release();

  uint16_t image[128];
  pvd::shared_vector<const uint16_t> pimage(image,0,128);
  for(unsigned i=0; i<pimage.size(); i++)
    image[i] = (i&0xffff);

  while(1) {
    const Pds::Tpr::Frame* frame = tpr->advance();
    uint64_t ts = frame->timeStamp;
    for(unsigned i=0; i<4; i++) {
      image[i] = (ts&0xffff);
      ts >>= 16;
    }
      
    channel.put(cpv)
      .set<const uint16_t>("value",pimage)
      .exec();

    if (verbose)
      printf("Timestamp %lu.%09u\n",frame->timeStamp>>32,unsigned(frame->timeStamp&0xffffffff));
  }

  return 0;
}

