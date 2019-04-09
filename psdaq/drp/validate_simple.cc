#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdio>
#include "DataDriver.h"

#define MAX_RET_CNT_C 1000

int main()
{
    int nlanes = 4;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    uint32_t rxFlags[MAX_RET_CNT_C];
    uint32_t rxErrors[MAX_RET_CNT_C];

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

    uint32_t lastEvtCounter[4] = {0, 0, 0, 0};

    while (1) {
        int32_t ret = dmaReadBulkIndex(fd, MAX_RET_CNT_C, dmaRet, dmaIndex, rxFlags, rxErrors, dest);
        for (int b=0; b < ret; b++) {
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            uint32_t flag = rxFlags[b];
            uint32_t error = rxErrors[b];

            const uint32_t* data = (uint32_t*)dmaBuffers[index];
            uint32_t evtCounter = data[5] & 0xffffff;

            if (evtCounter != (lastEvtCounter[lane] + 1)) {
                printf("\033[0;31m");
                printf("Jump in last lane: lane[%u] %u -> %u | difference %d\n",
                       lane, lastEvtCounter[lane], evtCounter, evtCounter - lastEvtCounter[lane]);
                printf("\033[0m");
                printf("index %u\n", index);
                printf("flag %u | error %u\n", flag, error);
            }
            lastEvtCounter[lane] = evtCounter;
        }
        if ( ret > 0 ) {
            dmaRetIndexes(fd, ret, dmaIndex);
        }
    }
}
