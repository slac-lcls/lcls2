#include <getopt.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <string.h>
#include "DataDriver.h"

//  DrpTDet register map
//    Master (device = 0x2030)
//      0x00800000 MigToPciDma
//      0x00A00000 TDetSemi
//      0x00C00000 TDetTiming
//      0x00E00000 I2C Devices
//    Slave (device = 0x2030)
//      0x00800000 MigToPciDma
//      0x00A00000 TDetSemi
//
//  MigToPciDma
//    0x00.1  monEnable
//    0x80-0x9c[Lane 0], 0xa0-0xbc [Lane 1], 0xc0-0xdc [Lane 2], 0xe0-0xfc [Lane 3]
//      0x84.8  blocksPause
//      0x88.0  blocksFree
//      0x88.12 blocksQueued
//      0x8C.0  writeQueCnt
//      0x90.0  wrIndex
//      0x94.0  wcIndex
//      0x98.0  rdIndex
//      ..
//      0x100    monClkRate|Slow|Fast|Lock
//      0x104    monClkRate|Slow|Fast|Lock
//      0x108    monClkRate|Slow|Fast|Lock
//      0x10C    monClkRate|Slow|Fast|Lock
//
//  TDetSemi
//    0x00.0  partition
//    0x00.3  clear
//    0x00.4  length
//    0x00.28 enable
//    0x04.0  id
//    0x08.0  partitionAddr
//    0x0c.0  modPrsL
//    0x10-0x1c [Lane 0], ...
//      0x10.0  cntL0
//      0x10.24 cntOflow
//      0x14.0  cntL1A
//      0x18.0  cntL1R
//      0x1c.0  cntWrFifo
//      0x1c.0  cntRdFifo
//      0x1c.16 msgDelay
//
//  TimingCore (0x00c00000)
//
//  TimingGtCore (0x00c10000)
//
//  TriggerEventManager (0x00c20000)
//
//  I2C Devices
//    0x000-0x3FC I2C Mux
//    0x400-0x7FC QSFP1, QSFP0, EEPROM { I2C Mux = 1,4,5 }
//    0x800-0xBFC Si570                { I2C Mux = 2 }
//    0xC00-0xFFC Fan                  { I2C Mux = 3 }
//

#define CLIENTS(i)       (0x00800080 + i*0x20)
#define DMA_LANES(i)     (0x00800100 + i*0x20)
#define TRG_LANES(i)     (0x00C20100 + i*0x100)
#define SI570(i)         (0x00e00800 +i*4)


static int fd;

static inline uint32_t get_reg32(int reg) {
  unsigned v;
  dmaReadRegister(fd, reg, &v);
  return v;
}

static inline void set_reg32(int reg, uint32_t value) {
  dmaWriteRegister(fd, reg, value);
}

static void print_mig_lane(const char* name, int addr, int offset, int mask)
{
    const unsigned MIG_LANES = 0x00800080;
    printf("%20.20s", name);
    for(int i=0; i<4; i++) {
      uint32_t reg = get_reg32( MIG_LANES + i*32 + addr);
      printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void print_clk_rate(const char* name, int addr) 
{
    const unsigned CLK_BASE = 0x00800100;
    printf("%20.20s", name);
    uint32_t reg = get_reg32( CLK_BASE + addr);
    printf(" %f MHz", double(reg&0x1fffffff)*1.e-6);
    if ((reg>>29)&1) printf(" [slow]");
    if ((reg>>30)&1) printf(" [fast]");
    if ((reg>>31)&1) printf(" [locked]");
    printf("\n");
}

static void print_field(const char* name, int addr, int offset, int mask)
{
    printf("%20.20s", name);
    uint32_t reg = get_reg32( addr);
    printf(" %8x", (reg >> offset) & mask);
    printf("\n");
}

static void print_word (const char* name, int addr) { print_field(name,addr,0,0xffffffff); }

static void print_lane(const char* name, int addr, int offset, int stride, int mask, unsigned nl=8)
{
    printf("%20.20s", name);
    for(int i=0; i<nl; i++) {
        uint32_t reg = get_reg32( addr+stride*i);
        printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void select_si570()
{
  unsigned i2c_mux = get_reg32(0x00e00000);
  printf("i2c_mux : 0x%x\n", i2c_mux);
  i2c_mux = (1<<2);
  set_reg32(0x00e00000,i2c_mux);
  printf("i2c_mux : 0x%x\n", i2c_mux);
}

static void reset_si570()
{
//  Reset to factory defaults

  unsigned v = get_reg32(SI570(135));
  v |= 1;
  set_reg32(SI570(135), v);
  do { usleep(100); } while (get_reg32(SI570(135))&1);
}

static double read_si570()
{
  //  Read factory calibration for 156.25 MHz
  static const unsigned hsd_divn[] = {4,5,6,7,9,11};
  unsigned v = get_reg32(SI570(7));
  unsigned hs_div = hsd_divn[(v>>5)&7];
  unsigned n1 = (v&0x1f)<<2;
  v = get_reg32(SI570(8));
  n1 |= (v>>6)&3;
  uint64_t rfreq = v&0x3f;
  rfreq <<= 32;
  rfreq |= ((get_reg32(SI570( 9))&0xff)<<24) |
    ((get_reg32(SI570(10))&0xff)<<16) |
    ((get_reg32(SI570(11))&0xff)<< 8) |
    ((get_reg32(SI570(12))&0xff)<< 0);
 
  double f = (156.25 * double(hs_div * (n1+1))) * double(1<<28)/ double(rfreq);

  printf("Read: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, f);

  return f;
}

static void set_si570(double f)
{
  //  Program for 1300/7 MHz

  //  Freeze DCO
  unsigned v = get_reg32(SI570(137));
  v |= (1<<4);
  set_reg32(SI570(137),v);

  unsigned hs_div = 3; // =7
  unsigned n1     = 3; // =4
  uint64_t rfreq  = uint64_t(5200. / f * double(1<<28));

  set_reg32(SI570( 7),((hs_div&7)<<5) | ((n1>>2)&0x1f));
  set_reg32(SI570( 8),((n1&3)<<6) | ((rfreq>>32)&0x3f));
  set_reg32(SI570( 9),(rfreq>>24)&0xff);
  set_reg32(SI570(10),(rfreq>>16)&0xff);
  set_reg32(SI570(11),(rfreq>> 8)&0xff);
  set_reg32(SI570(12),(rfreq>> 0)&0xff);

  printf("Wrote: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, f);

  //  Unfreeze DCO
  v = get_reg32(SI570(137));
  v &= ~(1<<4);
  set_reg32(SI570(137),v);

  v = get_reg32(SI570(135));
  v |= (1<<6);
  set_reg32(SI570(135),v);
}

static void measure_clks(double& txrefclk, double& rxrefclk)
{
  unsigned tv = get_reg32(0x00c00028);
  unsigned rv = get_reg32(0x00c00010);
  usleep(1000000);
  unsigned tw = get_reg32(0x00c00028);
  unsigned rw = get_reg32(0x00c00010);
  txrefclk = double(tw-tv)*16.e-6;
  printf("TxRefClk: %f MHz\n", txrefclk);
  rxrefclk = double(rw-rv)*16.e-6;
  printf("RxRecClk: %f MHz\n", rxrefclk);
}

static void dump_ring()
{
  unsigned base = 0x00c10000;
  // clear
  unsigned csr = get_reg32(base);
  csr |= (1<<30);
  set_reg32(base,csr);
  usleep(1);
  csr &= ~(1<<30);
  set_reg32(base,csr);
  // enable
  csr |= (1<<31);
  set_reg32(base,csr);
  printf("csr %08x\n",csr);
  usleep(100);
  // disable
  csr &= ~(1<<31);
  set_reg32(base,csr);
  // dump
  unsigned dataWidth = 20;
  unsigned mask = dataWidth < 32 ? (1<<dataWidth)-1 : 0xffffffff;
  unsigned cmask = (dataWidth+3)/4;
  //  csr = get_reg32(base);
  unsigned len = csr & 0xfffff;
  if (len == 0)  len=64;

  uint32_t* buff = new uint32_t[len];
  for(unsigned i=0; i<len; i++)
    buff[i] = get_reg32(base+4*i)&mask;

  printf("csr %08x  mask 0x%x  cmask %u  dataWidth %u\n", 
         csr, mask, cmask, dataWidth);
  for(unsigned i=0; i<len; i++)
    printf("%0*x%c", cmask, buff[i], (i&0x7)==0x7 ? '\n':' ');

  delete[] buff;
}

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <device file> [default /dev/datadev_1]\n");
  printf("         -c              [setup clock synthesizer]\n");
  printf("         -s              [dump status]\n");
  printf("         -S              [dump status ring buffers]\n");
  printf("         -m              [disable DRAM monitoring]\n");
  printf("         -M              [enable DRAM monitoring]\n");
  printf("         -t              [reset timing counters]\n");
  printf("         -T              [reset timing PLL]\n");
  printf("         -F              [reset frame counters]\n");
  printf("         -C partition[,length[,links]] [configure simcam]\n");
  printf("Requires -b or -d\n");
}

int main(int argc, char* argv[])
{
    bool setup_clk = false;
    bool reset_clk = false;
    bool status    = false;
    bool ringb     = false;
    bool timingRst = false;
    bool tcountRst = false;
    bool frameRst  = false;
    int dramMon    = -1;
    int delayVal   = -1;
    bool updateId  = true;
    int partition  = -1;
    int length     = 320;
    int links      = 0xff;
    const char* dev = "/dev/datadev_1";
    char* endptr;
    
    int c;
    while((c = getopt(argc, argv, "cd:rsStTmMFD:C:")) != EOF) {
      switch(c) {
      case 'd': dev = optarg; break;
      case 'c': setup_clk = true; updateId = true; break;
      case 'r': reset_clk = true; break;
      case 's': status    = true; break;
      case 'S': ringb     = true; break;
      case 't': tcountRst = true; break;
      case 'T': timingRst = true; break;
      case 'm': dramMon   = 0;    break;
      case 'M': dramMon   = 1;    break;
      case 'F': frameRst  = true; break;
      case 'D': delayVal  = strtoul(optarg,&endptr,0); break;
      case 'C': partition = strtoul(optarg,&endptr,0);
        if (*endptr==',') {
          length = strtoul(endptr+1,&endptr,0);
          if (*endptr==',')
            links = strtoul(endptr+1,NULL,0);
        }
        break;
      default: usage(argv[0]); return 0;
      }
    }
    
    if ( (fd = open(dev,O_RDWR)) < 0) {
      perror("Opening device file");
      return 1;
    }

    bool core_pcie = true;

    { 
      struct AxiVersion axiv;
      axiVersionGet(fd, &axiv);

      printf("-- Core Axi Version --\n");
      printf("  firmware version  :  %x\n", axiv.firmwareVersion);
      printf("  scratch           :  %x\n", axiv.scratchPad);
      printf("  uptime count      :  %d\n", axiv.upTimeCount);
      printf("  build string      :  %s\n", axiv.buildString);

      for(unsigned i=0; i<64; i++) {
        uint32_t userValue = axiv.userValues[i];
        printf("%08x%c", userValue, (i&7)==7 ? '\n':' ');
      }

      core_pcie = (axiv.userValues[2] == 0);
    }

    //
    //  Update ID advertised on timing link
    //
    if (updateId && core_pcie) {
      struct addrinfo hints;
      struct addrinfo* result;

      memset(&hints, 0, sizeof(struct addrinfo));
      hints.ai_family = AF_INET;       /* Allow IPv4 or IPv6 */
      hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
      hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */

      char hname[64];
      gethostname(hname,64);
      int s = getaddrinfo(hname, NULL, &hints, &result);
      if (s != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
        exit(EXIT_FAILURE);
      }

      sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;

      unsigned id = 0xfb000000 | 
        (ntohl(saddr->sin_addr.s_addr)&0xffff);
      set_reg32( 0x00c20020, id);
    }

    //
    //  Measure si570 clock output
    //
    if (core_pcie) {
      double txrefclk, rxrefclk;
      measure_clks(txrefclk,rxrefclk);
      setup_clk |= ( txrefclk < 185 || txrefclk > 186 );

      if (setup_clk) {
        select_si570();
        reset_si570();
        
        double f = read_si570();
        set_si570(f);
        read_si570();
        
        double txrefclk, rxrefclk;
        measure_clks(txrefclk,rxrefclk);
      }
      else if (reset_clk) {
        select_si570();
        reset_si570();
        
        double f = read_si570();

        double txrefclk, rxrefclk;
        measure_clks(txrefclk,rxrefclk);
      }

      timingRst |= setup_clk;
      if (timingRst) {
        printf("Reset timing PLL\n");
        unsigned v = get_reg32( 0x00c00020);
        v |= 0x80;
        set_reg32( 0x00c00020, v);
        usleep(10);
        v &= ~0x80;
        set_reg32( 0x00c00020, v);
        usleep(100);
        v |= 0x8;
        set_reg32( 0x00c00020, v);
        usleep(10);
        v &= ~0x8;
        set_reg32( 0x00c00020, v);
        usleep(100000);
      }
      
      tcountRst |= timingRst;
      if (tcountRst) {
        printf("Reset timing counters\n");
        unsigned v = get_reg32( 0x00c00020) | 1;
        v &= ~(1<<5);  // clear linkDown latch
        set_reg32( 0x00c00020, v);
        usleep(10);
        v &= ~0x1;
        set_reg32( 0x00c00020, v);
      }
    }

    if (dramMon==1) {
      unsigned v = get_reg32( 0x00800000);
      v |= 1;
      set_reg32( 0x00800000,v);
    }
    else if (dramMon==0) {
      unsigned v = get_reg32( 0x00800000);
      v &= ~1;
      set_reg32( 0x00800000,v);
    }

    //  set new defaults for pause threshold
    const unsigned MIG_LANES = 0x00800080;
    for(int i=0; i<4; i++) {
        unsigned v = get_reg32( MIG_LANES + i*32 + 4);
        v &= ~(0x3ff<<8);
        v |= (0x3f<<8);
        set_reg32( MIG_LANES + i*32+4, v);
    }

    if (status) {
      //      uint32_t lanes = get_reg32( RESOURCES);
      uint32_t lanes = 4;
      printf("  lanes             :  %u\n", lanes);

      printf("  monEnable         :  %u\n", get_reg32( 0x00800000)&1);

      printf("\n-- migLane Registers --\n");
      print_mig_lane("blockSize  ", 0, 0, 0x1f);
      print_mig_lane("blocksPause", 4, 8, 0x3ff);
      print_mig_lane("blocksFree ", 8, 0, 0x1ff);
      print_mig_lane("blocksQued ", 8,12, 0x1ff);
      print_mig_lane("writeQueCnt",12, 0, 0xff);
      print_mig_lane("wrIndex    ",16, 0, 0x1ff);
      print_mig_lane("wcIndex    ",20, 0, 0x1ff);
      print_mig_lane("rdIndex    ",24, 0, 0x1ff);

      print_clk_rate("axilOther  ",0);
      print_clk_rate("timingRef  ",4);
      print_clk_rate("migA       ",8);
      print_clk_rate("migB       ",12);

      // TDetSemi
      print_lane("length"    , 0x00a00000,  0, 4, 0xffffff, 4);
      print_lane("clear"     , 0x00a00000, 30, 4, 0x1, 4);
      print_lane("enable"    , 0x00a00000, 31, 4, 0x1, 4);

      { printf("%20.20s", "messagedelay");
        for(unsigned i=0; i<8; i++) {
          uint32_t reg = get_reg32( 0x00c20000+i*4);
          printf(" %8x", reg & 0xff); }
        printf("\n"); }

      print_field("localid"  , 0x00c20020,  0, 0xffffffff);
      print_field("remoteid" , 0x00c20024,  0, 0xffffffff);

      print_lane("enable"     , 0x00c20100,  0, 256, 0x7);
      print_lane("partition"  , 0x00c20104,  0, 256, 0xf);
      print_lane("cntL0"      , 0x00c20114,  0, 256, 0xffffffff);
      print_lane("cntL1A"     , 0x00c20118,  0, 256, 0xffffffff);
      print_lane("cntL1R"     , 0x00c2011c,  0, 256, 0xffffffff);
      print_lane("cntTra"     , 0x00c20120,  0, 256, 0xffffffff);
      print_lane("cntFrame"   , 0x00c20124,  0, 256, 0xffffffff);
      print_lane("cntTrig"    , 0x00c20128,  0, 256, 0xffffffff);
      print_lane("fullToTrig" , 0x00c20138,  0, 256, 0xfff);
      print_lane("nfullToTrig", 0x00c2013c,  0, 256, 0xfff);

      //      if (core_pcie) {
      if (1) {
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

        print_field("RxAlign_tgt ",0x00c10100,0,0xff);
        print_field("RxAlign_mask",0x00c10100,8,0xff);
        print_field("RxAlign_last",0x00c10104,0,0xff);
        for(unsigned i=0; i<40; i++)
          printf("%02x%c", (get_reg32(0x00c10000+4*(i/4))>>(4*(i%4)))&0xff, (i%10)==9?'\n':' ');
        printf("\n");
      }
    }

    if (ringb) {
      dump_ring();
    }

    if (delayVal>=0) {
      unsigned v = get_reg32(0x00c00024);
      v |= (1<<31);
      set_reg32(0x00c00024,v);
      usleep(1);
      set_reg32(0x00c00024,delayVal);
    }

    if (frameRst) {
      unsigned v = get_reg32( 0x00a00000);
      unsigned w = v;
      w &= ~(0xf<<28);    // disable and drain
      set_reg32( 0x00a00000,w);
      usleep(1000);
      w |=  (1<<3);       // reset
      set_reg32( 0x00a00000,w);
      usleep(1);         
      set_reg32( 0x00a00000,v);
    }

    if (partition >= 0) {
      unsigned v =
        ((length&0xffffff)<<0) |
        (1<<31);
      set_reg32( 0x00a00000, v);
      unsigned w = get_reg32( 0x00a00000);
      printf("Configured partition [%u], length [%u], links [%x]: [%x](%x)\n",
             partition, length, links, v, w);
      for(unsigned i=0; i<4; i++)
        if (links&(1<<i)) {
          set_reg32( 0x00800084+32*i, 0x1f00);
          set_reg32( TRG_LANES(i)+4, partition);
          set_reg32( TRG_LANES(i)+8, 16);
          set_reg32( TRG_LANES(i)+0, 3);
        }
        else {
          set_reg32( TRG_LANES(i), 0);
        }
    }

    return 0;
}

