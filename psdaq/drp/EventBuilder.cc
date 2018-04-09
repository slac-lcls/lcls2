#include "EventBuilder.hh"

#include <rdma/fi_domain.h>

#include <assert.h>
#include <unistd.h>                     // sysconf
#include <stdlib.h>                     // posix_memalign
#include <stdio.h>                      // perror

using namespace XtcData;
using namespace Pds::Eb;

MyDgram::MyDgram(unsigned pulseId, uint64_t val)
{
    seq = XtcData::Sequence(Sequence::Event, TransitionId::L1Accept, TimeStamp(), PulseId(pulseId));
    env = 0;
    xtc = Xtc(TypeId(TypeId::Data, 0), TheSrc(Level::Segment, ContribId));
    _data = val;
    xtc.alloc(sizeof(_data));
}

static size_t calcBatchSize(unsigned maxEntries, size_t maxSize)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  size_t size      = sizeof(Dgram) + maxEntries * maxSize;
  size             = alignment * ((size + alignment - 1) / alignment);
  return size;
}

static void* allocBatchRegion(unsigned maxBatches, size_t maxBatchSize)
{
  size_t   alignment = sysconf(_SC_PAGESIZE);
  size_t   size      = maxBatches * maxBatchSize;
  assert((size & (alignment - 1)) == 0);
  void*    region    = nullptr;
  int      ret       = posix_memalign(&region, alignment, size);
  if (ret)
  {
    perror("posix_memalign");
    return nullptr;
  }

  return region;
}

void eb_rcvr(MyBatchManager& myBatchMan)
{
    char* ifAddr = nullptr;
    std::string srvPort = "32832"; // add 64 to the client base
    unsigned numEb = 1;
    size_t maxBatchSize = calcBatchSize(maxEntries, maxSize);
    void* region = allocBatchRegion(maxBatches, maxBatchSize);
    EbLfServer myEbLfServer(ifAddr, srvPort, numEb);
    printf("*** rcvr %d %zd\n",maxBatches,maxBatchSize);
    myEbLfServer.connect(ContribId, region, maxBatches * maxBatchSize, EbLfServer::PEERS_SHARE_BUFFERS);
    unsigned nreceive = 0;
    unsigned none = 0;
    unsigned nzero = 0;
    while(1) {
        fi_cq_data_entry wc;
        if (myEbLfServer.pend(&wc))  continue;
        unsigned     idx   = wc.data & 0x00ffffff;
        unsigned     srcId = wc.data >> 24;
        const Dgram* batch = (const Dgram*)(myEbLfServer.lclAdx(srcId, idx * maxBatchSize));

        myEbLfServer.postCompRecv(srcId);

        // printf("received batch %p %d\n",batch,idx);
        const Batch* input  = myBatchMan.batch(idx);
        const Dgram* result = (const Dgram*)batch->xtc.payload();
        const Dgram* last   = (const Dgram*)batch->xtc.next();
        while(result != last) {
            nreceive++;
            // printf("--- result %lx\n",*(uint64_t*)(result->xtc.payload()));
            uint64_t val = *(uint64_t*)(result->xtc.payload());
            result = (Dgram*)result->xtc.next();
            if (val==0) {
                nzero++;
            } else if (val==1) {
                none++;
            } else {
                printf("error %ld\n",val);
            }
            if (nreceive%10000==0) printf("%d %d %d\n",nreceive,none,nzero);
        }
        delete input;
    }
}
