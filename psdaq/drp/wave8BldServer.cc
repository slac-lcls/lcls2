#include <atomic>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "psdaq/aes-stream-drivers/AxisDriver.h"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/bld/Server.hh"
#include "psdaq/bld/Header.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <stdexcept>
#include "EventBatcher.hh"
#include "drp.hh"

#define MAX_RET_CNT_C 1000

static int fd;
static std::atomic<bool> terminate;

using namespace Drp;

static unsigned dmaDest(unsigned lane, unsigned vc)
{
    return (lane << PGP_MAX_LANES) | vc;
}

static void int_handler(int) {
    terminate.store(true, std::memory_order_release);
}

struct ProcStreamLayout {
    uint32_t quadrantSel;
    uint32_t ptrigCount;
    double   intensity;
    double   posX;
    double   posY;
};

struct Wave8BldPayload {
    double intensity;
    double positionXRaw;
    double positionYRaw;
};

// TriggerEventManager2 register layout (from TriggerEventManager2.hh):
//   TEM2 base = 0x00C20000
//   TEB[0] = base + sizeof(TEM2) = base + 0x9000
//   TEB registers: enable@+0, group@+4, pauseThresh@+8, triggerDelay@+0xC
static void init_kcu_teb(int dma_fd, unsigned lane, unsigned readoutGroup)
{
    const uint32_t TEB0 = 0x00C29000 + lane * 0x1000;
    dmaWriteRegister(dma_fd, TEB0 + 0x00, 1 << 2);
    dmaWriteRegister(dma_fd, TEB0 + 0x04, readoutGroup);
    dmaWriteRegister(dma_fd, TEB0 + 0x08, 16);
    dmaWriteRegister(dma_fd, TEB0 + 0x0C, 42);
    dmaWriteRegister(dma_fd, TEB0 + 0x00, 3);
    dmaWriteRegister(dma_fd, 0x00a00000 + 4 * (lane & 3), 1 << 30);
    dmaWriteRegister(dma_fd, 0x00a00000 + 4 * (lane & 3), 1 << 31);
    printf("kcu1500 TEB[%u] enabled (group=%u)\n", lane, readoutGroup);
}

static int setup_mc(unsigned addr, unsigned port, unsigned iface)
{
    printf("setup_mc %x/%u iface=%x\n", addr, port, iface);

    int fd_mc = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_mc < 0) { perror("socket"); return -1; }

    int y = 1;
    if (setsockopt(fd_mc, SOL_SOCKET, SO_BROADCAST, &y, sizeof(y)) < 0) {
        perror("SO_BROADCAST"); return -1;
    }

    sockaddr_in sa{};
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = htonl(iface);
    sa.sin_port        = htons(port);
    if (::bind(fd_mc, (sockaddr*)&sa, sizeof(sa)) < 0) {
        perror("bind"); return -1;
    }

    sockaddr_in dst{};
    dst.sin_family      = AF_INET;
    dst.sin_addr.s_addr = htonl(addr);
    dst.sin_port        = htons(port);
    if (connect(fd_mc, (sockaddr*)&dst, sizeof(dst)) < 0) {
        perror("connect"); return -1;
    }

    in_addr mif;
    mif.s_addr = htonl(iface);
    if (setsockopt(fd_mc, IPPROTO_IP, IP_MULTICAST_IF, &mif, sizeof(mif)) < 0) {
        perror("IP_MULTICAST_IF"); return -1;
    }

    return fd_mc;
}

static void show_usage(const char* p)
{
    printf("Usage: %s -d <device> [-l <lane>] [-m <mcast_addr>] [-p <port>] [-i <iface>] [-g <group>] [-v]\n", p);
    printf("  -d  DMA device (e.g. /dev/datadev_0)\n");
    printf("  -l  PGP lane number (default 1; use 0 if DAQ fiber is on lane 0)\n");
    printf("  -m  Multicast address (dotted decimal); omit for print-only mode\n");
    printf("  -p  UDP port (default 11001)\n");
    printf("  -i  Outgoing NIC address (dotted decimal, default 0.0.0.0)\n");
    printf("  -g  XPM readout group (default 0)\n");
    printf("  -v  Verbose — print intensity/posX/posY for every event\n");
}

int main(int argc, char* argv[])
{
    std::string device;
    std::string mcast_addr;
    unsigned    mc_port      = 11001;
    std::string iface_str    = "0.0.0.0";
    unsigned    readoutGroup = 0;
    unsigned    lane         = 1;
    bool        verbose      = false;

    int c;
    while ((c = getopt(argc, argv, "d:l:m:p:i:g:vh?")) != EOF) {
        switch (c) {
        case 'd': device       = optarg;         break;
        case 'l': lane         = atoi(optarg);   break;
        case 'm': mcast_addr   = optarg;         break;
        case 'p': mc_port      = atoi(optarg);   break;
        case 'i': iface_str  = optarg;           break;
        case 'g': readoutGroup = atoi(optarg);   break;
        case 'v': verbose    = true;             break;
        default:  show_usage(argv[0]); return 1;
        }
    }

    if (device.empty()) { show_usage(argv[0]); return 1; }

    terminate.store(false, std::memory_order_release);
    signal(SIGINT, int_handler);

    // Open DMA device — DAQ lane is lane 1, VC 1
    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);
    dmaAddMaskBytes(mask, dmaDest(lane, 1));
    printf("DMA dest: lane %u VC 1 = 0x%x\n", lane, dmaDest(lane, 1));

    fd = open(device.c_str(), O_RDWR);
    if (fd < 0) { perror(device.c_str()); return 1; }

    uint32_t dmaCount, dmaSize;
    void** dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (!dmaBuffers) { printf("Failed to map DMA buffers\n"); return 1; }
    printf("dmaCount=%u dmaSize=%u\n", dmaCount, dmaSize);

    if (dmaSetMaskBytes(fd, mask)) {
        perror("dmaSetMaskBytes");
        printf("Is another process holding %s lane %u VC 1?\n", device.c_str(), lane);
        return 1;
    }

    // Enable kcu1500 TriggerEventBuffer so events are forwarded to DMA
    init_kcu_teb(fd, lane, readoutGroup);

    // Set up multicast socket (optional — omit -m for print-only mode)
    int fd_mc = -1;
    if (!mcast_addr.empty()) {
        unsigned addr = ntohl(inet_addr(mcast_addr.c_str()));
        unsigned iface = ntohl(inet_addr(iface_str.c_str()));
        fd_mc = setup_mc(addr, mc_port, iface);
        if (fd_mc < 0) return 1;
    } else {
        printf("No multicast address given — print-only mode\n");
    }

    Pds::Bld::Server* bldServer = fd_mc >= 0 ? new Pds::Bld::Server(fd_mc) : nullptr;

    uint64_t nevents = 0, npublished = 0;
    int32_t  dmaRet  [MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dmaDst  [MAX_RET_CNT_C];

    while (!terminate.load(std::memory_order_acquire)) {

        int ret = dmaReadBulkIndex(fd, MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dmaDst);

        for (int b = 0; b < ret; b++) {
            uint32_t index = dmaIndex[b];
            uint32_t size  = dmaRet[b];

            EvtBatcherHeader& ebh = *(EvtBatcherHeader*)(dmaBuffers[index]);
            const Pds::TimingHeader* th =
                reinterpret_cast<const Pds::TimingHeader*>(ebh.next());

            ++nevents;

            // Find tdest=11 (ProcStream) by iterating backwards over subframe tails
            EvtBatcherIterator it(&ebh, size);
            const ProcStreamLayout* ps = nullptr;
            EvtBatcherSubFrameTail* tail;
            while ((tail = it.next()) != nullptr) {
                if (tail->tdest() == 11) {
                    if (tail->size() != sizeof(ProcStreamLayout)) {
                        printf("WARNING: ProcStream size %u != expected %zu — skipping\n",
                               tail->size(), sizeof(ProcStreamLayout));
                        break;
                    }
                    ps = reinterpret_cast<const ProcStreamLayout*>(tail->data());
                    break;
                }
            }

            if (!ps) continue;

            if (verbose)
                printf("pulseId=%lu intensity=%.6f posX=%.6f posY=%.6f\n",
                       th->pulseId(), ps->intensity, ps->posX, ps->posY);

            if (bldServer) {
                Wave8BldPayload payload{ps->intensity, ps->posX, ps->posY};
                bldServer->publish(th->pulseId(), th->time.value(),
                                   reinterpret_cast<const char*>(&payload),
                                   sizeof(payload));
                ++npublished;
            }
        }

        if (ret > 0) {
            if (bldServer) bldServer->flush();
            dmaRetIndexes(fd, ret, dmaIndex);
        }
    }

    printf("finished: nEvents=%lu nPublished=%lu\n", nevents, npublished);
    delete bldServer;
    if (fd_mc >= 0) close(fd_mc);
    close(fd);
    return 0;
}
