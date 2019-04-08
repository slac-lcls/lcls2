#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdio>
#include "DataDriver.h"

#define MAX_RET_CNT_C 1000

struct DmaBuffer
{
    int32_t size;
    uint32_t index;
    void* data;
};

struct PGPEvent
{
    DmaBuffer buffers[4];
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

int main()
{
    int nlanes = 4;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    uint32_t rxFlags[MAX_RET_CNT_C];
    uint32_t rxErrors[MAX_RET_CNT_C];

    std::vector<PGPEvent> pgpEvents(2*131072);

    int fd = open("/dev/datadev_1", O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening /dev/datadev_1"<<'\n';
        return -1;
    }

    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    for (int i=0; i<nlanes; i++) {
        dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
    }
    dmaSetMaskBytes(fd, mask);

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
                    /*
                    for (int l=0; l<nlanes; l++) {
                        const uint32_t* data = (uint32_t*)event->buffers[l].data;
                        printf("lane %u  evtCounter %u\n", l, data[5]&0xffffff);
                    }
                    */
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
