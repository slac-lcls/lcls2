#include "EventBuilder.hh"

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

void eb_rcvr(MyBatchManager& myBatchMan)
{
    std::string srvPort = "32832"; // add 64 to the client base
    unsigned numEb = 1;
    size_t maxBatchSize = sizeof(Dgram) + maxEntries * maxSize;
    EbFtServer myEbFtServer(srvPort, numEb, maxBatches * maxBatchSize, EbFtServer::PEERS_SHARE_BUFFERS);
    printf("*** rcvr %d %zd\n",maxBatches,maxBatchSize);
    myEbFtServer.connect(ContribId);
    unsigned nreceive = 0;
    unsigned none = 0;
    unsigned nzero = 0;
    while(1) {
        uint64_t data;
        if (myEbFtServer.pend(&data))  continue;
        const Dgram* batch = (const Dgram*)data;
        unsigned idx = ((const char*)batch - myEbFtServer.base()) / maxBatchSize;
        // printf("received batch %p %d\n",batch,idx);
        const Batch*  input  = myBatchMan.batch(idx);

        const Dgram*  result = (const Dgram*)batch->xtc.payload();
        const Dgram*  last   = (const Dgram*)batch->xtc.next();
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
