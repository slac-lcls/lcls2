//
//  Test code for synchronous transitions in a set of readout groups
//  Use this code to receive the transitions
//  Use the following to generate transitions at some ludicrous rate:
//     python <release>/psdaq/psdaq/cas/xpmtrantest.py -x 2 -p 15 -r 500. -t 10
//
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdio>
#include "drp.hh"
#include "DataDriver.h"
#include "DmaDest.h"

#define MAX_RET_CNT_C 1000

struct DmaBuffer
{
    int32_t size;
    uint32_t index;
    void* data;
};

struct PGPEvent
{
    DmaBuffer buffers[PGP_MAX_LANES];
    // uint8_t bufferMask = 0;
    uint8_t counter = 0;
};

void monitorFunc(const uint64_t& nevents, const uint64_t& bytes)
{
    auto t = std::chrono::steady_clock::now();
    uint64_t oldNevents = nevents;
    uint64_t oldBytes = bytes;
    while(1) {
        sleep(1);
        auto oldt = t;
        t = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double seconds = duration / 1.0e6;
        double eventRate = double(nevents - oldNevents) / seconds;
        double dataRate = double(bytes - oldBytes) / seconds;
        printf("%.2e Hz  |  %.2e MB/s\n", eventRate, dataRate / 1.0e6);

        oldNevents = nevents;
        oldBytes = bytes;
    }
}

static inline void set_reg32(int fd, int reg, uint32_t value) {
  dmaWriteRegister(fd, reg, value);
}

int main()
{
    //
    //  Configure the DrpTDet image to put each lane on a different readout group
    //

    const int nlanes = 4;

    int fd = open("/dev/datadev_1", O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening /dev/datadev_1"<<'\n';
        return -1;
    }

#define TRG_LANES(i)     (0x00C20100 + i*0x100)

    for(unsigned i=0; i<nlanes; i++) {
        set_reg32( fd, 0x00a00000+4*i, (1<<31));
        set_reg32( fd, 0x00800084+32*i, 0x1f00);
        set_reg32( fd, TRG_LANES(i)+4, i);
        set_reg32( fd, TRG_LANES(i)+8, 16);
        set_reg32( fd, TRG_LANES(i)+0, 3);
    }

    //
    //  Register to receive the DMAs for each of these lanes
    //
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    uint32_t rxFlags[MAX_RET_CNT_C];
    uint32_t rxErrors[MAX_RET_CNT_C];

    std::vector<PGPEvent> pgpEvents(2*131072);

    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (int i=0; i<nlanes; i++) {
        dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
    }
    if (dmaSetMaskBytes(fd, mask)) {
        printf("Failed to allocate lane/vc\n");
        return -1;
    }

    uint32_t dmaCount, dmaSize;
    void** dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (dmaBuffers == NULL ) {
        std::cout<<"Failed to map dma buffers!\n";
        return -1;
    }
    printf("dmaCount %u  dmaSize %u\n", dmaCount, dmaSize);

    uint32_t lastComplete;
    uint32_t lastEvtCounter[4] = {0, 0, 0, 0};
    uint64_t lastPid[4] = {0, 0, 0, 0};
    uint64_t nevents = 0;
    uint64_t bytes = 0;
    std::thread monitorThread(monitorFunc, std::ref(nevents), std::ref(bytes));

    std::vector<int> receivedCounts(dmaCount, 0);
    std::vector<int> releaseCounts(dmaCount, 0);

    //
    //  Process the received transitions and verify the timestamps are equal across all lanes/groups
    //
    while (1) {
        int32_t ret = dmaReadBulkIndex(fd, MAX_RET_CNT_C, dmaRet, dmaIndex, rxFlags, rxErrors, dest);
        for (int b=0; b < ret; b++) {
            int32_t size = dmaRet[b];
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            uint32_t flag = rxFlags[b];
            uint32_t error = rxErrors[b];
            bytes += size;
            receivedCounts[index]++;
            const uint32_t* data = (uint32_t*)dmaBuffers[index];

            uint64_t pid = *reinterpret_cast<const uint64_t*>(data);

            uint32_t evtCounter = data[5] & 0xffffff;
            uint32_t current = evtCounter % pgpEvents.size();
            PGPEvent* event = &pgpEvents[current];

            DmaBuffer* buffer = &event->buffers[lane];
            buffer->size = size;
            uint32_t oldIndex = buffer->index;
            buffer->index = index;
            buffer->data = dmaBuffers[index];
            event->counter++;

            if (evtCounter != (lastEvtCounter[lane] + 1)) {
                printf("\033[0;31m");
                printf("Jump in last lane: lane[%u] 0x%x -> 0x%x | difference %d\n",
                       lane, lastEvtCounter[lane], evtCounter, evtCounter - lastEvtCounter[lane]);
                printf("\033[0m");
                printf("pid %ld\n", int64_t(pid) - int64_t(lastPid[lane]));
                printf("index %u %u\n", oldIndex, index);
                printf("counts %d %d\n", receivedCounts[index], releaseCounts[index]);
                printf("flag %u | error %u\n", flag, error);
            }
            lastEvtCounter[lane] = evtCounter;
            lastPid[lane] = pid;

            if (event->counter == nlanes) {
                if (evtCounter != (lastComplete + 1)) {
                    printf("\033[0;31m");
                    printf("Jump in complete l1Count %u -> %u | difference %d\n",
                           lastComplete, evtCounter, evtCounter - lastComplete);
                    printf("\033[0m");
                    //printf("index %u\n", index);

                    for (unsigned e=lastComplete+1; e<evtCounter; e++) {
                        PGPEvent* brokenEvent = &pgpEvents[e % pgpEvents.size()];
                        // printf("broken event: counter %d\n", brokenEvent->counter);
                        brokenEvent->counter = 0;

                    }
                }
                else {
                    //  Check timestamp consistency
                    uint64_t ts = *(uint64_t*)event->buffers[0].data;
                    for (int l=1; l<nlanes; l++) {
                        uint64_t tsl = *(uint64_t*)event->buffers[l].data;
                        if (tsl != ts)
                            printf("lane %u  evtCounter 0x%x  ts 0x%lx [0x%lx]\n", 
                                   l, data[5]&0xffffff, tsl, ts);
                    }
                }
                lastComplete = evtCounter;
                event->counter = 0;
                nevents++;
            }
            // printf("lane %u | size %d | evtCounter %u | index %u\n", lane, size, evtCounter, index);
        }
        if ( ret > 0 ) {
            for (int b=0; b < ret; b++) {
                uint32_t index = dmaIndex[b];
                releaseCounts[index]++;
            }
            dmaRetIndexes(fd, ret, dmaIndex);
        }
    }
}
