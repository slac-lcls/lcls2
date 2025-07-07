#include <getopt.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <string.h>
#include "psdaq/aes-stream-drivers/DataDriver.h"
#include "Si570.hh"

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

/*
typedef struct {
    unsigned addr;
    const char* name;
} drpreg_t;

static drpreg_t drpregs[] = { {0x02, "cdr_swap_mode"},
                              {0x03, "rxbufreset_time"},
                              {0x04, "rxcdrphreset_time"},
                              {0x05, "rxpmareset_time"},
                              {0x06, "rxdfe_cfg1"},
                              {0x09, "txpmareset_time"},
                              {0x0b, "rxpmaclk_sel"},
                              {0x0c, "txprogclk_sel"},
                              {0x0e, "rxcdr_cfg0"},
                              {0x0f, "rxcdr_cfg1"},
                              {0x10, "rxcdr_cfg2"},
                              {0x11, "rxcdr_cfg3"},
                              {0x12, "rxcdr_cfg4"},
                              {0x13, "rxcdr_lock_cfg0"},
                              {0x17, "rxbuffer_cfg"},
                              {0x26, "rxdfe_he_cfg0"},
                              {0x27, "align_comma_word"},
                              {0x28, "cpll_fbdiv"},
                              {0x29, "cpll_lock_cfg"},
                              {0x2a, "cpll_refclk_div"},
                              {0x2b, "cpll_init_cfg0"},
                              {0x2c, "dec_pcomma_detect"},
                              {0x2d, "rxcdr_lock_cfg1"},
                              {0x2e, "rxcfok_cfg1"},
                              {0x2f, "rxdfe_h2_cfg0"},
                              {0x30, "rxdfe_h2_cfg1"},
                              {0x31, "rxcfok_cfg2"},
                              {0x32, "rxlpm_cfg"},
                              {0x33, "rxlpm_kh_cfg0"},
                              {0x34, "rxlpm_kh_cfg1"},
                              {0x35, "rxdfelpm_kl_cfg0"},
                              {0x36, "rxdfelpm_kl_cfg1"},
                              {0x37, "rxlpm_os_cfg0"},
                              {0x38, "rxlpm_os_cfg1"},
                              {0x39, "rxlpm_gc_cfg"},
                              {0x3a, "dmonitor_cfg1"},
                              {0x3d, "rxdfe_hc_cfg0"},
                              {0x3e, "txprogdiv_cfg"},
                              {0x50, "rxdfe_hc_cfg1"},
                              {0x52, "rx_dfe_agc_cfg"},
                              {0x53, "rxdfe_cfg0"},
                              {0x54, "rxdfe_cfg1"},
                              {0x55, "align_mcomma"},
                              {0x56, "align_pcomma"},
                              {0x57, "txdly_lcfg"},
                              {0x58, "rxdfe_os_cfg0"},
                              {0x59, "rxphdly_cfg"},
                              {0x5a, "rxdfe_os_cfg1"},
                              {0x5b, "rxdly_cfg"},
                              {0x5c, "rxdly_lcfg"},
                              {0x5d, "rxdfe_hf_cfg0"},
                              {0x5e, "rxdfe_hd_cfg0"},
                              {0x5f, "rxbias_cfg0"},
                              {0x61, "rxph_monitor_sel"},
                              {0x62, "rxsum_dfetap"},
                              {0x63, "rxout_div"},
                              {0x64, "rxsig_valid_dly"},
                              {0x65, "rxbuf_thresh_ovflw"},
                              {0x66, "rxbuf_thresh_ovrd"},
                              {0x67, "rxbuf_reset"},
                              {0x6d, "rxclk25"},
                              {0x6e, "txphdly_cfg0"},
                              {0x6f, "txphdly_cfg1"},
                              {0x70, "txdly_cfg"},
                              {0x71, "txph_mon"},
                              {0x72, "rxcdr_lock_cfg2"},
                              {0x73, "txph_cfg"},
                              {0x74, "term_rcal_cfg"},
                              {0x75, "rxdfe_hf_cfg1"},
                              {0x76, "term_rcal_ovrd"},
                              {0x7a, "txclk25"},
                              {0x7b, "txdeemph"},
                              {0x7c, "txgearbox"},
                              {0x7d, "txrxdetect_cfg"},
                              {0x7e, "txclkmux_en"},
                              {0x7f, "txmargin_full01"},
                              {0x80, "txmargin_full23"},
                              {0x81, "txmargin_full4_low0"},
                              {0x82, "txmargin_low12"},
                              {0x83, "txmargin_low34"},
                              {0x84, "rxdfe_hd_cfg1"},
                              {0x85, "txintdatawidth"},
                              {0x8a, "8a"},
                              {0x8b, "8b"},
                              {0x8c, "8c"},
 };
*/

#define CLIENTS(i)       (0x00800080 + i*0x20)
#define DMA_LANES(i)     (0x00800100 + i*0x20)

#define NEWTEM
#ifdef NEWTEM
#define XMA_REG(i)       (0x00C48000 + i)
#define TEB_REG(i)       (0x00C49000 + i)
#define TRG_LANES(i)     (0x00C49000 + i*0x100)
#else
#define XMA_REG(i)       (0x00C20000 + i)
#define TEB_REG(i)       (0x00C20100 + i)
#define TRG_LANES(i)     (0x00C20100 + i*0x100)
#endif

//#define NEWMUX
#ifdef NEWMUX
#define SI570(i)         (0x00072000 +i*4)
#else
#define SI570(i)         (0x00e00800 +i*4)
#endif

static int fd;
static int lanes = 4;

static inline uint32_t get_reg32(int reg) {
  unsigned v;
  if (-1==dmaReadRegister(fd, reg, &v))
      printf("  Error in dmaReadRegister(%d, %x, %p)\n",fd,reg,&v);
  return v;
}

static inline void set_reg32(int reg, uint32_t value) {
  dmaWriteRegister(fd, reg, value);
}

static inline uint32_t get_i2c(int reg) {
#ifdef NEWMUX
    //    reg &= 0xffff;
    printf("get_i2c %x\n",reg);
    set_reg32(0x70008,reg);
    set_reg32(0x70000,1);
    uint32_t status;
    unsigned tmo=0;
    while(1) {
        status = get_reg32(0x70004);
        if ((status&1)==0)
            break;
        tmo++;
        usleep(1000);
    }
    printf("get_i2c wait for done [%x]\n",tmo);
    tmo=0;
    while(1) {
        status = get_reg32(0x70004);
        if (status&1)
            break;
        usleep(1000);
        tmo++;
    }
    uint32_t value = get_reg32(0x7000c);
    printf("get_i2c returning %x [%x : %x]\n",value,(status>>1),tmo);
    return value;
#else
    return get_reg32(reg);
#endif
}

static inline void set_i2c(int reg, uint32_t value) {
#ifdef NEWMUX
    //    reg &= 0xffff;
    printf("set_i2c %x %x\n",reg,value);
    set_reg32(0x70008,reg);
    set_reg32(0x7000c,value);
    set_reg32(0x70000,0);
    unsigned tmo=0;
    while(1) {
        value = get_reg32(0x70004);
        if ((value&1)==0)
            break;
        tmo++;
        usleep(1000);
    }
    printf("set_i2c wait for done [%x]\n",tmo);
    tmo=0;
    while(1) {
        value = get_reg32(0x70004);
        if (value&1)
            break;
        tmo++;
        usleep(1000);
    }
    printf("set_i2c done [%x : %x]\n",(value>>1),tmo);
#else
    set_reg32(reg,value);
#endif
}

static void print_hmb_lane(const char* name, int addr, int offset, int mask)
{
    const unsigned HMB_LANES = 0x00800000;
    printf("%20.20s", name);
    for(int i=0; i<lanes; i++) {
      uint32_t reg = get_reg32( HMB_LANES + i*256 + addr);
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

static const char* field_format = "%20.20s";

static void print_field(const char* name, int addr, int offset, int mask)
{
    printf(field_format, name);
    uint32_t reg = get_reg32( addr);
    printf(" %8x", (reg >> offset) & mask);
    printf("\n");
}

static void print_word (const char* name, int addr) { print_field(name,addr,0,0xffffffff); }

static void print_lane(const char* name, int addr, int offset, int stride, int mask, unsigned nl=8)
{
    printf("%20.20s", name);
    for(unsigned i=0; i<nl; i++) {
        uint32_t reg = get_reg32( addr+stride*i);
        printf(" %8x", (reg >> offset) & mask);
    }
    printf("\n");
}

static void select_si570()
{
#ifdef NEWMUX
    return; // handled by firmware
#else
  unsigned i2c_mux = get_reg32(0x00e00000);
  printf("i2c_mux : 0x%x\n", i2c_mux);
  i2c_mux = (1<<2);
  set_reg32(0x00e00000,i2c_mux);
  printf("i2c_mux : 0x%x\n", i2c_mux);
#endif
}

static void reset_si570()
{
//  Reset to factory defaults

  unsigned v = get_i2c(SI570(135));
  v |= 1;
  set_i2c(SI570(135), v);
  do { usleep(100); } while (get_i2c(SI570(135))&1);
}

static double read_si570()
{
  //  Read factory calibration for 156.25 MHz
  unsigned v = get_i2c(SI570(7));
  unsigned hs_div = ((v>>5)&7) + 4;
  unsigned n1 = (v&0x1f)<<2;
  v = get_i2c(SI570(8));
  n1 |= (v>>6)&3;
  uint64_t rfreq = v&0x3f;
  rfreq <<= 32;
  rfreq |= ((get_i2c(SI570( 9))&0xff)<<24) |
    ((get_i2c(SI570(10))&0xff)<<16) |
    ((get_i2c(SI570(11))&0xff)<< 8) |
    ((get_i2c(SI570(12))&0xff)<< 0);

  double f = (156.25 * double(hs_div * (n1+1))) * double(1<<28)/ double(rfreq);

  printf("Reg[7:12]:");
  for(unsigned i=7; i<13; i++)
      printf(" %02x", get_i2c(SI570(i)));
  printf("\n");

  printf("Read: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, f);

  return f;
}

static void set_si570(double f)
{
  //  Program for 1300/7 MHz

  //  Freeze DCO
  unsigned v = get_i2c(SI570(137));
  v |= (1<<4);
  set_i2c(SI570(137),v);

  unsigned hs_div = 3; // =7
  unsigned n1     = 3; // =4
  uint64_t rfreq  = uint64_t(5200. / f * double(1<<28));

  set_i2c(SI570( 7),((hs_div&7)<<5) | ((n1>>2)&0x1f));
  set_i2c(SI570( 8),((n1&3)<<6) | ((rfreq>>32)&0x3f));
  set_i2c(SI570( 9),(rfreq>>24)&0xff);
  set_i2c(SI570(10),(rfreq>>16)&0xff);
  set_i2c(SI570(11),(rfreq>> 8)&0xff);
  set_i2c(SI570(12),(rfreq>> 0)&0xff);

  printf("Reg[7:12]:");
  for(unsigned i=7; i<13; i++)
      printf(" %02x", get_i2c(SI570(i)));
  printf("\n");

  printf("Wrote: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div+4, n1, rfreq, f);

  //  Unfreeze DCO
  v = get_i2c(SI570(137));
  v &= ~(1<<4);
  set_i2c(SI570(137),v);

  v = get_i2c(SI570(135));
  v |= (1<<6);
  set_i2c(SI570(135),v);
}

static void set_si570_119m(double f)
{
  //  Program for 119 MHz

  //  Freeze DCO
  unsigned v = get_i2c(SI570(137));
  v |= (1<<4);
  set_i2c(SI570(137),v);

  // DCO = 476M * 11 = 5236M
  // HSDIV = 11
  // N1 = 4
  unsigned hs_div = 7; // =11
  unsigned n1     = 3; // =4
  uint64_t rfreq  = uint64_t(5236. / f * double(1<<28));

  set_i2c(SI570( 7),((hs_div&7)<<5) | ((n1>>2)&0x1f));
  set_i2c(SI570( 8),((n1&3)<<6) | ((rfreq>>32)&0x3f));
  set_i2c(SI570( 9),(rfreq>>24)&0xff);
  set_i2c(SI570(10),(rfreq>>16)&0xff);
  set_i2c(SI570(11),(rfreq>> 8)&0xff);
  set_i2c(SI570(12),(rfreq>> 0)&0xff);

  printf("Reg[7:12]:");
  for(unsigned i=7; i<13; i++)
      printf(" %02x", get_i2c(SI570(i)));
  printf("\n");

  printf("Wrote: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div+4, n1, rfreq, f);

  //  Unfreeze DCO
  v = get_i2c(SI570(137));
  v &= ~(1<<4);
  set_i2c(SI570(137),v);

  v = get_i2c(SI570(135));
  v |= (1<<6);
  set_i2c(SI570(135),v);
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
  printf("         -V              [dump registers]\n");
  printf("         -m              [disable DRAM monitoring]\n");
  printf("         -M              [enable DRAM monitoring]\n");
  printf("         -U              [user reset to clear readout pipeline]\n");
  printf("         -t              [reset timing counters]\n");
  printf("         -T              [reset timing PLL]\n");
  printf("         -R              [reset timing receiver]\n");
  printf("         -F              [reset frame counters]\n");
  printf("         -C partition[,length[,links]] [configure simcam]\n");
  printf("         -L              [toggle loopback mode]\n");
  printf("         -1              [use 119MHz for refclk]\n");
  printf("Requires -b or -d\n");
}

int main(int argc, char* argv[])
{
    bool setup_clk = false;
    bool reset_clk = false;
    bool status    = false;
    bool ringb     = false;
    bool dumpReg   = false;
    long long int gpu_bypass = 0;
    bool timingRst = false;
    bool rxTimRst = false;
    bool tcountRst = false;
    bool frameRst  = false;
    bool userRst   = false;
    bool loopback  = false;
    int clksel     = 1; // LCLS2 default
    int dramMon    = -1;
    int delayVal   = -1;
    bool updateId  = true;
    int partition  = -1;
    int length     = 320;
    int links      = 0xff;
    const char* dev = "/dev/datadev_1";
    char* endptr;

    int c;
    while((c = getopt(argc, argv, "cd:l:rRsStTULmMFVD:C:1:w:")) != EOF) {
      switch(c) {
      case '1': clksel = 0; break;
      case 'd': dev = optarg; break;
      case 'c': setup_clk = true; updateId = true; break;
      case 'l': lanes = strtoul(optarg,&endptr,0); break;
      case 'L': loopback = true; break;
      case 'r': reset_clk = true; break;
      case 'R': rxTimRst  = true; break;
      case 's': status    = true; break;
      case 'S': ringb     = true; break;
      case 't': tcountRst = true; break;
      case 'T': timingRst = true; break;
      case 'U': userRst   = true; break;
      case 'V': dumpReg   = true; break;
      case 'm': dramMon   = 0;    break;
      case 'M': dramMon   = 1;    break;
      case 'F': frameRst  = true; break;
      case 'D': delayVal  = strtoul(optarg,&endptr,0); break;
      case 'w': gpu_bypass = strtoul(optarg,&endptr,0); break;
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

    // Complain if all arguments weren't consummed
    if (optind < argc) {
        printf("Unrecognized argument:\n");
        while (optind < argc)
            printf("  %s ", argv[optind++]);
        printf("\n");
        usage(argv[0]);
        return 1;
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

      if (strstr((char*)axiv.buildString,"DrpTDet")==0) {
          printf("Unexpected firmware image. Exiting.\n");
          return 1;
      }
    }

    //
    //  Update ID advertised on timing link
    //
    //    if (updateId && core_pcie) {
    if (updateId) {
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

      while(result) {
          sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;
          unsigned ip = ntohl(saddr->sin_addr.s_addr);
          if ((ip>>16)==0xac15) {
              unsigned id = 0xfb000000 | (ip&0xffff);
              set_reg32( XMA_REG(0x20), id);
              break;
          }
          result = result->ai_next;
      }

      if (!result) {
          printf("No 172.21 address found.  Defaulting");
          unsigned id = 0xfb000000;
          set_reg32( XMA_REG(0x20), id);
      }

    }

    //
    //  Measure si570 clock output
    //
    if (1) {
        //    if (core_pcie) {
      // double txrefclk, rxrefclk;
      // measure_clks(txrefclk,rxrefclk);

      // static const unsigned txrefclk_min[] = {118,185};
      // static const unsigned txrefclk_max[] = {120,186};

      // setup_clk |= ( txrefclk < txrefclk_min[clksel] ||
      //                txrefclk > txrefclk_max[clksel] );

      // if (setup_clk) {
      //   select_si570();
      //   reset_si570();

      //   double f = read_si570();
      //   if (clksel)
      //       set_si570(f);
      //   else
      //       set_si570_119m(f);
      //   read_si570();

      //   double txrefclk, rxrefclk;
      //   measure_clks(txrefclk,rxrefclk);
      // }
      // else if (reset_clk) {
      //   select_si570();
      //   reset_si570();

      //   // Revisit: Unused:  double f = read_si570();

      //   double txrefclk, rxrefclk;
      //   measure_clks(txrefclk,rxrefclk);
      // }

      timingRst |= setup_clk;

      //
      //  Dump timing link stats before resetting
      //
      if (timingRst || rxTimRst || tcountRst) {
        print_word("SOFcounts" , 0x00c80000);
        print_word("RxRstDone" , 0x00c80014);
        print_word("RxDecErrs" , 0x00c80018);
        print_word("RxDspErrs" , 0x00c8001c);
      }

      if (timingRst) {
        printf("Reset timing PLL\n");
        unsigned v = get_reg32( 0x00c80020);
        /*
        ** PLL reset can hang the timing link (8_0000 TimingCore)
        */
        v |= 0x80;
        set_reg32( 0x00c80020, v);
        usleep(1000);
        v &= ~0x80;
        set_reg32( 0x00c80020, v);
        usleep(1000000);
        v |= 0x8;
        set_reg32( 0x00c80020, v);
        usleep(1000);
        v &= ~0x8;
        set_reg32( 0x00c80020, v);
        usleep(1000000);
      }

      if (rxTimRst) {
        printf("Reset timing Rx\n");
        unsigned v = get_reg32( 0x00c80020);
        v |= 0x8;
        set_reg32( 0x00c80020, v);
        usleep(1000);
        v &= ~0x8;
        set_reg32( 0x00c80020, v);
        usleep(1000000);
      }

      tcountRst |= rxTimRst;
      tcountRst |= timingRst;
      if (tcountRst) {
        printf("Reset timing counters\n");
        unsigned v = get_reg32( 0x00c80020) | 1;
        v &= ~(1<<5);  // clear linkDown latch
        set_reg32( 0x00c80020, v);
        usleep(100);
        v &= ~0x1;
        set_reg32( 0x00c80020, v);
      }
    }

    // if (userRst) {
    //   set_reg32( 0x00800000,(1<<31));
    // }

    // if (dramMon==1) {
    //   unsigned v = get_reg32( 0x00800000);
    //   v |= 1;
    //   set_reg32( 0x00800000,v);
    // }
    // else if (dramMon==0) {
    //   unsigned v = get_reg32( 0x00800000);
    //   v &= ~1;
    //   set_reg32( 0x00800000,v);
    // }

    // //  set new defaults for pause threshold
    // const unsigned MIG_LANES = 0x00800080;
    // for(int i=0; i<lanes; i++) {
    //     unsigned v = get_reg32( MIG_LANES + i*32 + 4);
    //     v &= ~(0x3ff<<8);
    //     v |= (0x3f<<8);
    //     set_reg32( MIG_LANES + i*32+4, v);
    // }

    if (status) {
      //      uint32_t lanes = get_reg32( RESOURCES);
      //      uint32_t lanes = 4;
      printf("  lanes             :  %u\n", lanes);

      // printf("  monEnable         :  %u\n", get_reg32( 0x00800000)&1);

      printf("\n-- hmbLane AxiStreamDmaV2Fifo Registers --\n");
      print_hmb_lane("Version", 0, 0, 0xf);
      print_hmb_lane("baseAddr(24b)", 4, 8, 0xffffff);
      print_hmb_lane("buffFrameWidth", 24, 0, 0xff);
      print_hmb_lane("axibuffWidth", 24, 8, 0xff);
      print_hmb_lane("burstByte", 24, 16, 0xfff);
      print_hmb_lane("rdBuffCnt", 28, 0, 0xffff);
      print_hmb_lane("wrBuffCnt", 28, 16, 0xffff);
      print_hmb_lane("pauseCnt", 32, 0, 0xffff);
      print_hmb_lane("sAxisCtrlpause", 32, 16, 0x1);

      // print_clk_rate("axilOther  ",0);
      // print_clk_rate("timingRef  ",4);
      // print_clk_rate("migA       ",8);
      // print_clk_rate("migB       ",12);

      // TDetSemi
      print_lane("length"    , 0x00a00000,  0, 4, 0xffffff, lanes);
      print_lane("clear"     , 0x00a00000, 30, 4, 0x1, lanes);
      print_lane("enable"    , 0x00a00000, 31, 4, 0x1, lanes);

      // TriggerEventManager
      { printf("%20.20s", "messagedelay");
        for(unsigned i=0; i<8; i++) {
            uint32_t reg = get_reg32(XMA_REG(i*4));
          printf(" %8x", reg & 0xff); }
        printf("\n"); }

      print_field("localid"  , XMA_REG(0x20),  0, 0xffffffff);
      print_field("remoteid" , XMA_REG(0x24),  0, 0xffffffff);

      print_lane("enable"     , TEB_REG(0x00),  0, 256, 0x7);
      print_lane("group"      , TEB_REG(0x04),  0, 256, 0xf);
      print_lane("pauseThr"   , TEB_REG(0x08),  0, 256, 0x1f);
      print_lane("pauseOF"    , TEB_REG(0x10),  0, 256, 0xd);
      print_lane("modes"      , TEB_REG(0x10), 16, 256, 0x7);
      print_lane("cntFifoWr"  , TEB_REG(0x10),  4, 256, 0x1f);
      print_lane("cntL0"      , TEB_REG(0x14),  0, 256, 0xffffffff);
      print_lane("cntL1A"     , TEB_REG(0x18),  0, 256, 0xffffffff);
      print_lane("cntL1R"     , TEB_REG(0x1c),  0, 256, 0xffffffff);
      print_lane("cntTra"     , TEB_REG(0x20),  0, 256, 0xffffffff);
      print_lane("cntFrame"   , TEB_REG(0x24),  0, 256, 0xffffffff);
      print_lane("cntTrig"    , TEB_REG(0x28),  0, 256, 0xffffffff);
      print_lane("word0"      , TEB_REG(0x30),  0, 256, 0xffffffff);
      print_lane("fullToTrig" , TEB_REG(0x38),  0, 256, 0xfff);
      print_lane("nfullToTrig", TEB_REG(0x3c),  0, 256, 0xfff);

      //      if (core_pcie) {
      if (1) {
        // TDetTiming
        print_word("SOFcounts" , 0x00c80000);
        print_word("EOFcounts" , 0x00c80004);
        print_word("Msgcounts" , 0x00c80008);
        print_word("CRCerrors" , 0x00c8000c);
        print_word("RxRecClks" , 0x00c80010);
        print_word("RxRstDone" , 0x00c80014);
        print_word("RxDecErrs" , 0x00c80018);
        print_word("RxDspErrs" , 0x00c8001c);
        print_word("CSR"       , 0x00c80020);
        print_field("  linkUp" , 0x00c80020, 1, 1);
        print_field("  polar"  , 0x00c80020, 2, 1);
        print_field("  clksel" , 0x00c80020, 4, 1);
        print_field("  ldown"  , 0x00c80020, 5, 1);
        print_word("MsgDelay"  , 0x00c80024);
        print_word("TxRefClks" , 0x00c80028);
        print_word("BuffByCnts", 0x00c8002c);

        print_field("RxAlign_tgt ",0x00c10100,0,0xff);
        print_field("RxAlign_mask",0x00c10100,8,0xff);
        print_field("RxAlign_last",0x00c10104,0,0xff);
        for(unsigned i=0; i<40; i++)
          printf("%02x%c", (get_reg32(0x00c10000+4*(i/4))>>(4*(i%4)))&0xff, (i%10)==9?'\n':' ');
        printf("\n");

        if (dumpReg) {
            field_format = "%30.30s";
            print_field("gth_rxalign_resetIn"  , 0x00c10180, 0, 1);
            print_field("gth_rxalign_resetDone", 0x00c10180, 1, 1);
            print_field("gth_rxalign_resetErr" , 0x00c10180, 2, 1);
            print_field("gth_rxalign_r_l0cked" , 0x00c10180, 3, 1);
            print_field("gth_rxalign_r_rst"    , 0x00c10180, 4, 1);
            print_field("gth_rxalign_r_rstcnt" , 0x00c10180, 8, 0xf);
            print_field("gth_rxalign_r_state"  , 0x00c10180, 12, 3);

            print_field("gth_txstatus_clkactive", 0x00c10184, 0, 1);
            print_field("gth_txstatus_bypassrst", 0x00c10184, 1, 1);
            print_field("gth_txstatus_pllreset" , 0x00c10184, 2, 1);
            print_field("gth_txstatus_datareset", 0x00c10184, 3, 1);
            print_field("gth_txstatus_resetdone", 0x00c10184, 4, 1);

            print_field("gth_rxstatus_clkactive", 0x00c10184,16, 1);
            print_field("gth_rxstatus_bypassrst", 0x00c10184,17, 1);
            print_field("gth_rxstatus_pllreset" , 0x00c10184,18, 1);
            print_field("gth_rxstatus_datareset", 0x00c10184,19, 1);
            print_field("gth_rxstatus_resetdone", 0x00c10184,20, 1);
            print_field("gth_rxstatus_bypassdon", 0x00c10184,21, 1);
            print_field("gth_rxstatus_bypasserr", 0x00c10184,22, 1);
            print_field("gth_rxstatus_cdrstable", 0x00c10184,23, 1);

            print_field("gth_rxctrl0out", 0x00c10188, 0, 0xffff);
            print_field("gth_rxctrl1out", 0x00c1018C, 0, 0xffff);
            print_field("gth_rxctrl3out", 0x00c10190, 0, 0xff);

            /*
              for(unsigned i=0; i<sizeof(drpregs)/sizeof(drpreg_t); i++)
              print_field(drpregs[i].name, 0x00c18000+drpregs[i].addr*4, 0, 0xff);
            */
            for(unsigned i=0; i<0x100; i++) {
                printf("drp_%02x : %04x  %04x  %04x\n", i,
                       get_reg32(0x00c18000+4*i)&0xffff,
                       get_reg32(0x00c18400+4*i)&0xffff,
                       get_reg32(0x00c18800+4*i)&0xffff);
            }
        }
      }
    }

    if (ringb) {
      dump_ring();
    }

    if (delayVal>=0) {
      unsigned v = get_reg32(0x00c80024);
      v |= (1<<31);
      set_reg32(0x00c80024,v);
      usleep(1);
      set_reg32(0x00c80024,delayVal);
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

    if (partition >= 0 && partition<8) {
      unsigned v =
        ((length&0xffffff)<<0) |
        (1<<31);
      printf("Configured partition [%u], length [%u], links [%x]: [%x]\n",
             partition, length, links, v);
      for(int i=0; i<8; i++)
        if (links&(1<<i)) {
          set_reg32( 0x00a00000+4*i, v);
          // set_reg32( 0x00800084+32*i, 0x1f00);
          set_reg32( TRG_LANES(i)+4, partition);
          set_reg32( TRG_LANES(i)+8, 16);
          set_reg32( TRG_LANES(i)+0, 3);
        }
    }
    else if (partition >= 8) {
        for(int i=0; i<8; i++)
            if (links&(1<<i)) {
                set_reg32( 0x00a00000+4*i, (1<<30));
                set_reg32( TRG_LANES(i), 0);
            }
    }

    // if (loopback) {
    //     usleep(100000); // allow some settling time before changing loopback
    //     unsigned lb = get_reg32( 0x00c30100 );
    //     printf("Loopback read %x\n",lb);
    //     lb ^= 2;
    //     set_reg32( 0x00c30100, lb);
    //     printf("Loopback wrote %x\n",lb);
    // }

   if (gpu_bypass>0) {
        unsigned v = get_reg32( 0x0002802c);
        print_word("demux route dest, mask",0x0002802c);
        v = gpu_bypass;
        set_reg32( 0x0002802c, v);
        printf("gpu bypassed \n");
   }

 return 0;
}


