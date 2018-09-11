#include <getopt.h>
#include "pgpdriver.h"

static uint8_t* pci_resource;

static void print_dma_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32(pci_resource, DMA_LANES(i) + addr);
        printf(" %8u", (reg >> offset) & mask);
    }
    printf("\n");
}

static void print_field(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    uint32_t reg = get_reg32(pci_resource, addr);
    printf(" %8u", (reg >> offset) & mask);
    printf("\n");
}

static void print_word (const char* name, int addr) { print_field(name,addr,0,0xffffffff); }

static void print_dti_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32(pci_resource, addr+16*i);
        printf(" %8u", (reg >> offset) & mask);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    int device_id = 0x2031;
    int c;
    while((c = getopt(argc, argv, "d:")) != EOF) {
        switch(c) {
            case 'd': device_id = strtol(optarg, NULL, 0); break;
        }
    }

    AxisG2Device dev(device_id);
    pci_resource = dev.reg();

    uint32_t version = get_reg32(pci_resource, VERSION);
    uint32_t scratch = get_reg32(pci_resource, SCRATCH);
    uint32_t uptime_count = get_reg32(pci_resource, UP_TIME_CNT);
    uint32_t lanes = get_reg32(pci_resource, RESOURCES) & 0xf;
    char build_string[256];
    for (int i=0; i<64; i++) {
        reinterpret_cast<uint32_t*>(build_string)[i] = get_reg32(pci_resource, 0x0800 + i*4);
    }  

    printf("-- Core Axi Version --\n");
    printf("  firmware version  :  %x\n", version);
    printf("  scratch           :  %x\n", scratch);
    printf("  uptime count      :  %d\n", uptime_count);
    printf("  build string      :  %s\n", build_string);
    
    printf("  lanes             :  %u\n", lanes);
    uint32_t fifo_depth = get_reg32(pci_resource, CLIENTS(0) + FIFO_DEPTH);
    printf("dcountRamAddr  [%u] : 0x%x\n", 0, fifo_depth & 0xffff);
    printf("dcountWriteDesc[%u] : 0x%x\n", 0, fifo_depth >> 16);

    printf("\n-- dmaLane Registers --\n");
    print_dma_lane("client", CLIENT, 0, 0xf);
    print_dma_lane("blockSize", BLOCK_SIZE, 0, 0xf);
    print_dma_lane("dcountTransfer", FIFO_DEPTH, 0, 0xffff);
    print_dma_lane("blocksFree", MEM_STATUS, 0, 0x3ff);

    print_dma_lane("blocksQueued", MEM_STATUS, 12, 0x3ff);
    print_dma_lane("tready", MEM_STATUS, 25, 1);
    print_dma_lane("wbusy", MEM_STATUS, 26, 1);
    print_dma_lane("wSlaveBusy", MEM_STATUS, 27, 1);
    print_dma_lane("rMasterBusy", MEM_STATUS, 28, 1);
    print_dma_lane("mm2s_err", MEM_STATUS, 29, 1);
    print_dma_lane("s2mm_err", MEM_STATUS,30, 1);
    print_dma_lane("memReady", MEM_STATUS, 31, 1);

    // TDetSemi
    print_field("partition", 0x00a00000,  0, 0xf);
    print_field("length"   , 0x00a00000, 16, 0x7fff);
    print_field("enable"   , 0x00a00000, 31, 1);
    print_field("id"       , 0x00a00004,  0, 0xffffffff);

    print_dti_lane("cntL0", 0x00a00010, 0, 0xffffff);
    print_dti_lane("cntOF", 0x00a00010, 24, 0xff);
    print_dti_lane("cntL1A", 0x00a00014, 0, 0xffffff);
    print_dti_lane("cntL1R", 0x00a00018, 0, 0xffffff);
    print_dti_lane("cntWrFifo", 0x00a01c, 0, 0xff);
    print_dti_lane("cntRdFifo", 0x00a01c, 8, 0xff);
    print_dti_lane("cntMsgDelay", 0x00a01c, 16, 0xffff);

    // TDetTiming
    print_word("SOFcounts" , 0x00c00000);
    print_word("EOFcounts" , 0x00c00004);
    print_word("Msgcounts" , 0x00c00008);
    print_word("CRCerrors" , 0x00c0000c);
    print_word("RxRecClks" , 0x00c00010);
    print_word("RxRstDone" , 0x00c00014);
    print_word("RxDecErrs" , 0x00c00018);
    print_word("RxDspErrs" , 0x00c0001c);
    print_word("CSR"       , 0x00c00020);
    print_word("MsgDelay"  , 0x00c00024);
    print_word("TxRefClks" , 0x00c00028);
    print_word("BuffByCnts", 0x00c0002c);

    return 1;

    uint8_t* p = pci_resource;

    uint32_t* i2c_mux   = (uint32_t*)(p + 0x00e00000);
    printf("i2c_mux : 0x%x\n", i2c_mux[0]);
    i2c_mux[0] = 2;
    printf("i2c_mux : 0x%x\n", i2c_mux[0]);

    uint32_t* si570   = (uint32_t*)(p + 0x00e00c00);
    static const unsigned hsd_divn[] = {4,5,6,7,9,11};
    unsigned v = si570[7];
    unsigned hs_div = hsd_divn[(v>>5)&7];
    unsigned n1 = (v&0x1f)<<2;
    v = si570[8];
    n1 |= (v>>6)&3;
    uint64_t rfreq = v&0x3f;
    rfreq <<= 32;
    rfreq |= ((si570[ 9]&0xff)<<24) |
      ((si570[10]&0xff)<<16) |
      ((si570[11]&0xff)<< 8) |
      ((si570[12]&0xff)<< 0);
 
    double f = (156.25 * double(hs_div * n1)) * double(1<<28)/ double(rfreq);

    printf("Read: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
           hs_div, n1, rfreq, f);

    return 0;

    //  Freeze DCO
    v = si570[137];
    v |= (1<<4);
    si570[137] = v;

    hs_div = 3;
    n1     = 4;
    rfreq  = uint64_t(5200. / f * double(1<<28));

    si570[ 7] = ((hs_div&7)<<5) | ((n1>>2)&0x1f);
    si570[ 8] = ((n1&3)<<6) | ((rfreq>>32)&0x3f);
    si570[ 9] = (rfreq>>24)&0xff;
    si570[10] = (rfreq>>16)&0xff;
    si570[11] = (rfreq>> 8)&0xff;
    si570[12] = (rfreq>> 0)&0xff;

    //  Unfreeze DCO
    v = si570[137];
    v &= ~(1<<4);
    si570[137] = v;

    v = si570[135];
    v |= (1<<6);
    si570[135] = v;
}

