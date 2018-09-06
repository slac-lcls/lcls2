#include <getopt.h>
#include <unistd.h>
#include "pgpdriver.h"

static uint8_t* pci_resource;

static void print_dma_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32(pci_resource, DMA_LANES(i) + addr);
        printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void print_field(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    uint32_t reg = get_reg32(pci_resource, addr);
    printf(" %8x", (reg >> offset) & mask);
    printf("\n");
}

static void print_word (const char* name, int addr) { print_field(name,addr,0,0xffffffff); }

static void print_dti_lane(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
        uint32_t reg = get_reg32(pci_resource, addr+16*i);
        printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void select_si570()
{
  uint32_t* i2c_mux   = (uint32_t*)(pci_resource + 0x00e00000);
  printf("i2c_mux : 0x%x\n", i2c_mux[0]);
  i2c_mux[0] = (1<<2);
  printf("i2c_mux : 0x%x\n", i2c_mux[0]);
}

static void reset_si570()
{
  uint32_t* si570   = (uint32_t*)(pci_resource + 0x00e00800);

  //  Reset to factory defaults
  unsigned v = si570[135];
  v |= 1;
  si570[135] = v;
  do { usleep(100); } while (si570[135]&1);
}

static double read_si570()
{
  //  Read factory calibration for 156.25 MHz
  uint32_t* si570   = (uint32_t*)(pci_resource + 0x00e00800);

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
 
  double f = (156.25 * double(hs_div * (n1+1))) * double(1<<28)/ double(rfreq);

  printf("Read: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, f);

  return f;
}

static void set_si570(double f)
{
  //  Program for 1300/7 MHz
  uint32_t* si570   = (uint32_t*)(pci_resource + 0x00e00800);

  //  Freeze DCO
  unsigned v = si570[137];
  v |= (1<<4);
  si570[137] = v;

  unsigned hs_div = 3; // =7
  unsigned n1     = 3; // =4
  uint64_t rfreq  = uint64_t(5200. / f * double(1<<28));

  si570[ 7] = ((hs_div&7)<<5) | ((n1>>2)&0x1f);
  si570[ 8] = ((n1&3)<<6) | ((rfreq>>32)&0x3f);
  si570[ 9] = (rfreq>>24)&0xff;
  si570[10] = (rfreq>>16)&0xff;
  si570[11] = (rfreq>> 8)&0xff;
  si570[12] = (rfreq>> 0)&0xff;

  printf("Wrote: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, f);

  //  Unfreeze DCO
  v = si570[137];
  v &= ~(1<<4);
  si570[137] = v;

  v = si570[135];
  v |= (1<<6);
  si570[135] = v;
}

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <device_id>  [default: 0x2031]\n");
  printf("         -c              [setup clock synthesizer]\n");
  printf("         -r              [reset clock synthesizer]\n");
  printf("         -s              [dump status]\n");
  printf("         -t              [reset timing counters]\n");
  printf("         -T              [reset timing PLL]\n");
  printf("         -m              [measure clock freq]\n");
  printf("         -C partition[,length[,links]] [configure simcam]\n");
  printf("         -I id           [set base link ID]\n");
}

int main(int argc, char* argv[])
{
    int device_id  = 0x2031;
    bool reset_clk = false;
    bool setup_clk = false;
    bool status    = false;
    bool timingRst = false;
    bool tcountRst = false;
    bool measure   = false;
    int id         = 0;
    int partition  = -1;
    int length     = 320;
    int links      = 0xff;
    char* endptr;

    int c;
    while((c = getopt(argc, argv, "cd:mrstTC:I:")) != EOF) {
      switch(c) {
      case 'd': device_id = strtol(optarg, NULL, 0); break;
      case 'c': setup_clk = true; break;
      case 'm': measure   = true; break;
      case 'r': reset_clk = true; break;
      case 's': status    = true; break;
      case 't': tcountRst = true; break;
      case 'T': timingRst = true; break;
      case 'C': partition = strtoul(optarg,&endptr,0);
        if (*endptr==',') {
          length = strtoul(endptr+1,&endptr,0);
          if (*endptr==',')
            links = strtoul(endptr+1,NULL,0);
        }
        break;
      case 'I': id = strtoul(optarg, NULL, 0); break;
      default: usage(argv[0]); return 0;
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

    if (timingRst && device_id == 0x2031) {
      printf("Reset timing PLL\n");
      unsigned v = get_reg32(pci_resource, 0x00c00020);
      v |= 0x80;
      set_reg32(pci_resource, 0x00c00020, v);
      usleep(10);
      v &= ~0x80;
      set_reg32(pci_resource, 0x00c00020, v);
      usleep(100);
      v |= 0x8;
      set_reg32(pci_resource, 0x00c00020, v);
      usleep(10);
      v &= ~0x8;
      set_reg32(pci_resource, 0x00c00020, v);
      usleep(100000);
    }

    if (tcountRst) {
      printf("Reset timing counters\n");
      unsigned v = get_reg32(pci_resource, 0x00c00020) | 1;
      set_reg32(pci_resource, 0x00c00020, v);
      usleep(10);
      v &= ~0x1;
      set_reg32(pci_resource, 0x00c00020, v);
    }

    if (status) {
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
      print_field("length"   , 0x00a00000,  4, 0xffffff);
      print_field("enable"   , 0x00a00000, 31, 1);
      print_field("id"       , 0x00a00004,  0, 0xffffffff);

      print_dti_lane("cntL0", 0x00a00010, 0, 0xffffff);
      print_dti_lane("cntOF", 0x00a00010, 24, 0xff);
      print_dti_lane("cntL1A", 0x00a00014, 0, 0xffffff);
      print_dti_lane("cntL1R", 0x00a00018, 0, 0xffffff);
      print_dti_lane("cntWrFifo", 0x00a01c, 0, 0xff);
      print_dti_lane("cntRdFifo", 0x00a01c, 8, 0xff);
      print_dti_lane("cntMsgDelay", 0x00a01c, 16, 0xffff);

      if (device_id == 0x2031) {
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
        print_field("  linkUp" , 0x00c00020, 1, 1);
        print_field("  polar"  , 0x00c00020, 2, 1);
        print_field("  clksel" , 0x00c00020, 4, 1);
        print_field("  ldown"  , 0x00c00020, 5, 1);
        print_word("MsgDelay"  , 0x00c00024);
        print_word("TxRefClks" , 0x00c00028);
        print_word("BuffByCnts", 0x00c0002c);
      }
    }

    if (reset_clk && device_id == 0x2031) {
      select_si570();
      reset_si570();
    }

    if (setup_clk && device_id == 0x2031) {
      select_si570();
      reset_si570();

      double f = read_si570();
      set_si570(f);
      read_si570();
    }

    if (measure) {
      unsigned tv = get_reg32(pci_resource,0x00c00028);
      unsigned rv = get_reg32(pci_resource,0x00c00010);
      usleep(1000000);
      unsigned tw = get_reg32(pci_resource,0x00c00028);
      unsigned rw = get_reg32(pci_resource,0x00c00010);
      printf("TxRefClk: %f MHz\n", double(tw-tv)*16.e-6);
      printf("RxRecClk: %f MHz\n", double(rw-rv)*16.e-6);
    }

    if (id)
      set_reg32(pci_resource, 0x00a00004, id);

    if (partition >= 0) {
      unsigned v = ((partition&0xf)<<0) |
        ((length&0xffffff)<<4) |
        (links ? (1<<31):0);
      set_reg32(pci_resource, 0x00a00000, v);
    }

    return 0;
}

