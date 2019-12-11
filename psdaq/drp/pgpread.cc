#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>

#define MAX_RET_CNT_C 1000
static int fd;
std::atomic<bool> terminate;

unsigned dmaDest(unsigned lane, unsigned vc)
{
    return (lane<<8) | vc;
}

void int_handler(int dummy)
{
    terminate.store(true, std::memory_order_release);
    // dmaUnMapDma();
}

int main(int argc, char* argv[])
{
    int c, channel;

    channel = 0;
    std::string device;
    while((c = getopt(argc, argv, "c:d:")) != EOF) {
        switch(c) {
            case 'd':
                device = optarg;
                break;
            case 'c':
                channel = atoi(optarg);
                break;
        }
    }

    terminate.store(false, std::memory_order_release);
    signal(SIGINT, int_handler);

    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (unsigned i=0; i<4; i++) {
        dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, channel));
    }

    std::cout<<"device  "<<device<<'\n';
    fd = open(device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<<device<<'\n';
        return -1;
    }

    uint32_t dmaCount, dmaSize;
    void** dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (dmaBuffers == NULL ) {
        printf("Failed to map dma buffers!\n");
        return -1;
    }
    printf("dmaCount %u  dmaSize %u\n", dmaCount, dmaSize);

    dmaSetMaskBytes(fd, mask);


    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dmaDest[MAX_RET_CNT_C];
    while (1) {
        if (terminate.load(std::memory_order_acquire) == true) {
            close(fd);
            printf("closed\n");
            break;
        }

        int32_t ret = dmaReadBulkIndex(fd, MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dmaDest);
        for (int b=0; b < ret; b++) {
            uint32_t index = dmaIndex[b];
            uint32_t size = dmaRet[b];
            uint32_t dest = dmaDest[b] >> 8;
            const Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(dmaBuffers[index]);
            XtcData::TransitionId::Value transition_id = event_header->service();

            printf("Size %u B | Dest %u | Transition id %d | pulse id %lu | event counter %u | index %u\n",
                   size, dest, transition_id, event_header->pulseId(), event_header->evtCounter, index);
            printf("env %08x\n", event_header->env);
        }
	    if ( ret > 0 ) dmaRetIndexes(fd, ret, dmaIndex);
	    //sleep(0.1)
    }
    printf("finished\n");
}
