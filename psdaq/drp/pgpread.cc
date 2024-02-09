#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <AxisDriver.h>
#include <stdlib.h>
#include "drp.hh"
#include "psdaq/service/EbDgram.hh"
#include "EventBatcher.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <unistd.h>
#include <getopt.h>
#include "psdaq/mmhw/TriggerEventManager2.hh"


typedef Pds::Mmhw::TriggerEventManager2 TEM;

#define MAX_RET_CNT_C 1000
static int fd;
static void dmaWriteRegister(int, uint32_t*, uint32_t);
std::atomic<bool> terminate;

using namespace Drp;

void dmaWriteRegister(int fd, uint32_t* addr, uint32_t val)
{
    uintptr_t addri = (uintptr_t)addr;
    dmaWriteRegister(fd, addri&0xffffffff, val);
}

unsigned dmaDest(unsigned lane, unsigned vc)
{
    return (lane<<PGP_MAX_LANES) | vc;
}

void int_handler(int dummy)
{
    terminate.store(true, std::memory_order_release);
    // dmaUnMapDma();
}

static void show_usage(const char* p)
{
    printf("Usage: %s -d <device file> [-c <virtChan>] [-l <laneMask>] [-r]\n",p);
    printf("       -r  Has batcher event builder\n");
    printf("       -v  verbose\n");
}

int main(int argc, char* argv[])
{
    int c, virtChan;

    virtChan = 0;
    uint8_t laneMask = (1 << PGP_MAX_LANES) - 1;
    std::string device;
    unsigned lverbose = 0;
    bool lrogue = false, timing_kcu_enable=false;
    bool lusage = false;
    while((c = getopt(argc, argv, "c:d:l:tvrh?")) != EOF) {
        switch(c) {
            case 'd':
                device = optarg;
                break;
            case 'c':
                virtChan = atoi(optarg);
                break;
            case 'l':
                laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'r':
                lrogue = true;
                break;
            case 'v':
                ++lverbose;
                break;
            case 't':
                timing_kcu_enable = true;
                break;
            default:
                lusage = true;
                break;
        }
    }

    if (lusage) {
        show_usage(argv[0]);
        return -1;
    }

    terminate.store(false, std::memory_order_release);
    signal(SIGINT, int_handler);

    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);

    for (unsigned i=0; i<PGP_MAX_LANES; i++) {
        if (laneMask & (1 << i)) {
            if (virtChan<0) {
                for(unsigned j=0; j<4; j++) {
                    uint32_t dest = dmaDest(i, j);
                    printf("setting lane %u, dest 0x%x \n",i,dest);
                    dmaAddMaskBytes((uint8_t*)mask, dest);
                }
            }
            else {
                uint32_t dest = dmaDest(i, virtChan);
                printf("setting lane %u, dest 0x%x \n",i,dest);
                dmaAddMaskBytes((uint8_t*)mask, dest);
            }
        }
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

    if (dmaSetMaskBytes(fd, mask)) {
        printf("Failed to allocate lane/vc\n");
        const unsigned* u = reinterpret_cast<const unsigned*>(mask);
        for(unsigned i=0; i<DMA_MASK_SIZE/4; i++)
            printf("%08x%c", u[i], (i%8)==7?'\n':' ');
        return -1;
    }

    if (timing_kcu_enable){
        unsigned m_readoutGroup = 0;
        int links = 1;

        TEM* mem_pointer = (TEM*)0x00C20000;
        TEM* tem = new (mem_pointer) TEM;
        for(unsigned i=0; i<8; i++) {
            if (links&(1<<i)) {
                Pds::Mmhw::TriggerEventBuffer& b = tem->det(i);
                dmaWriteRegister(fd, &b.enable, (1<<2)      );  // reset counters
                dmaWriteRegister(fd, &b.pauseThresh, 16     );
                dmaWriteRegister(fd, &b.group , m_readoutGroup);
                dmaWriteRegister(fd, &b.enable, 3           );  // enable

                dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<30));  // clear
                dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<31));  // enable
            }

        }

    }

    uint64_t nevents = 0L;
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
            uint32_t vc   = dmaDest[b] & 0xff;
            const Pds::TimingHeader* event_header;
            if (!lrogue)
                event_header = reinterpret_cast<Pds::TimingHeader*>(dmaBuffers[index]);
            else {
                EvtBatcherHeader& ebh = *(EvtBatcherHeader*)(dmaBuffers[index]);
                event_header = reinterpret_cast<Pds::TimingHeader*>(ebh.next());
                EvtBatcherSubFrameTail& ebsft = *(EvtBatcherSubFrameTail*)((char*)(dmaBuffers[index])+size-ebh.lineWidth(ebh.width));
                printf("\nEventBatcherHeader: vers %d seq %d width %d sfsize %d\n",ebh.version,ebh.sequence_count,ebh.width,ebsft.size());
            }
            XtcData::TransitionId::Value transition_id = event_header->service();

            ++nevents;

            if (lverbose || (transition_id != XtcData::TransitionId::L1Accept)) {
                printf("Size %u B | Dest %u.%u | Transition id %d | pulse id %lu | event counter %u | index %u\n",
                       size, dest, vc, transition_id, event_header->pulseId(), event_header->evtCounter, index);
                if (lverbose > 1) {
                    printf("env %08x\n", event_header->env);
                    for(unsigned i=0; i<((size+3)>>2); i++)
                        printf("%08x%c",reinterpret_cast<uint32_t*>(dmaBuffers[index])[i], (i&7)==7 ? '\n':' ');
                }
                else {
                    const uint32_t* p = reinterpret_cast<const uint32_t*>(event_header+1);
                    printf("env %08x | payload %08x %08x %08x %08x\n", event_header->env,p[0],p[1],p[2],p[3]);
                }
            }
        }
	    if ( ret > 0 ) dmaRetIndexes(fd, ret, dmaIndex);
	    //sleep(0.1)
    }
    printf("finished: nEvents %lu\n", nevents);
}
