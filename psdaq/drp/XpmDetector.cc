
#include "XpmDetector.hh"
#include "Si570.hh"
#include "AxisDriver.h"
#include "DataDriver.h"
#include "psalg/utils/SysLog.hh"
#include "psdaq/mmhw/TprCore.hh"
#include "psdaq/mmhw/TriggerEventManager2.hh"
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;
using Pds::Mmhw::Reg;
using Pds::Mmhw::TprCore;

static void resetTimingPll  (TprCore&);

//typedef Pds::Mmhw::TriggerEventManager TEM;
typedef Pds::Mmhw::TriggerEventManager2 TEM;

namespace Drp {

    //  hardware register model
    class DrpTDet {
    public:
        uint32_t reserved_to_80_0000[0x800000/4];
        uint32_t migToPci[0x200000/4];  // 0x0080_0000
        Reg      tDetSemi[8];           // 0x00A0_0000
        uint32_t reserved_to_c0_0000[(0x200000-sizeof(tDetSemi))/4];
        TprCore  tpr;                   // 0x00C0_0000
        uint32_t reserved_to_c2_0000[(0x020000-sizeof(TprCore))/4];
        TEM      tem;                   // 0x00C2_0000
        uint32_t reserved_to_e0_0000[(0x1E0000-sizeof(TEM))/4];
        Reg      i2c     [0x200];       // 0x00E0_0000
        Si570    si570;                 // 0x00E0_0800
    };

XpmDetector::XpmDetector(Parameters* para, MemPool* pool) :
    Detector(para, pool)
{
    int fd = pool->fd();
    Pds::Mmhw::Reg::set(fd);
    DrpTDet& hw = *new(0) DrpTDet;
    TprCore& tpr = hw.tpr;

    static const double flo[] = {115.,180.};
    static const double fhi[] = {125.,190.};

    unsigned index = (para->kwargs["timebase"]=="119M") ? 0:1;

    // Check timing reference clock, program if necessary
    double clkr = tpr.txRefClockRate();
    logging::info("Timing RefClk %f MHz\n", clkr);
    if (clkr < flo[index] || clkr > fhi[index]) {
      AxiVersion vsn;
      axiVersionGet(fd, &vsn);
      if (vsn.userValues[2]) {  // Only one PCIe interface has access to I2C bus
         logging::error("Si570 clock needs programming.  This PCIe interface has no I2C access.");
         return;
      }

      //  Flush I2C by reading the Mux
      unsigned mux = hw.i2c[0];
      logging::info("I2C mux:  0x%x\n", mux); // Force the read not to be optimized out
      //  Set the I2C Mux
      hw.i2c[0] = 1<<2;

      hw.si570.program(index);

      unsigned v = tpr.CSR;
      logging::info("Skip reset timing PLL\n");
      // v |= 0x80;
      // dmaWriteRegister(fd, 0x00C00020, v);
      // usleep(10);
      // v &= ~0x80;
      // dmaWriteRegister(fd, 0x00C00020, v);
      // usleep(100);
      logging::info("Reset timing data path\n");
      v |= 0x8;
      tpr.CSR = v;
      usleep(10);
      v &= ~0x8;
      tpr.CSR = v;
      usleep(100000);
    }
}

json XpmDetector::connectionInfo(const nlohmann::json& msg)
{
    DrpTDet& hw = *new(0) DrpTDet;
    TEM& tem     = hw.tem;

    //  Advertise ID on the timing link
    {
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
              tem.xma().txId = id;
              break;
          }
          result = result->ai_next;
      }

      if (!result) {
          logging::info("No 172.21 address found.  Defaulting");
          unsigned id = 0xfb000000;
          tem.xma().txId = id;
      }
    }

    //  Retrieve the timing link ID
    uint32_t reg = tem.xma().rxId;
    printf("*** XpmDetector: timing link ID is %08x = %u\n", reg, reg);

    // there is currently a failure mode where the register reads
    // back as zero or 0xffffffff (incorrectly). This is not the best
    // longterm fix, but throw here to highlight the problem. the
    // difficulty is that Matt says this register has to work
    // so that an automated software solution would know which
    // xpm TxLink's to reset (a chicken-and-egg problem) - cpo
    // Also, register is corrupted when port number > 15 - Ric
    if (!reg || reg==0xffffffff || (reg & 0xff) > 15) {
        logging::critical("XPM Remote link id register illegal value: 0x%x. Trying RxPllReset.",reg);
        resetTimingPll(hw.tpr);

        reg = tem.xma().rxId;
        printf("*** XpmDetector: timing link ID is %08x = %u\n", reg, reg);

        if (!reg || reg==0xffffffff || (reg & 0xff) > 15) {
            logging::critical("XPM Remote link id register illegal value: 0x%x. Aborting. Try XPM TxLink reset.",reg);
            abort();
        }
    }
    int xpm  = (reg >> 20) & 0x0F;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

// setup up device to receive data over pgp
void XpmDetector::connect(const json& connect_json, const std::string& collectionId)
{
    logging::info("XpmDetector connect");
    m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];

    // FIXME make configureable
    m_length = 100;
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
    if (it != m_para->kwargs.end())
        m_length = stoi(it->second);

    int fd = m_pool->fd();
    int links = m_para->laneMask;

    AxiVersion vsn;
    axiVersionGet(fd, &vsn);
    if (vsn.userValues[2]) // Second PCIe interface has lanes shifted by 4
       links <<= 4;

    //Pds::Mmhw::TriggerEventManager* tem = new ((void*)0x00C20000) Pds::Mmhw::TriggerEventManager;
    DrpTDet& hw = *new(0) DrpTDet;
    TEM& tem    = hw.tem;
    Reg* det    = hw.tDetSemi;
    for(unsigned i=0; i<8; i++) {
        if (links&(1<<i)) {
            Pds::Mmhw::TriggerEventBuffer& b = tem.det(i);
            b.enable = 1<<2;  // reset counters
            b.pauseThresh = 16;
            b.group = m_readoutGroup;
            b.enable = 3;  // enable

            det[i&3] = 1<<30;  // clear
            det[i&3] = (m_length&0xffffff) | (1<<31);
        }
    }
}

unsigned XpmDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    return 0;
}

void XpmDetector::shutdown()
{
    int fd = m_pool->fd();
    int links = m_para->laneMask;

    AxiVersion vsn;
    axiVersionGet(fd, &vsn);
    if (vsn.userValues[2]) // Second PCIe interface has lanes shifted by 4
       links <<= 4;

    DrpTDet& hw = *new(0) DrpTDet;
    for(unsigned i=0; i<8; i++) {
        if (links&(1<<i)) {
            hw.tDetSemi[i&3] = (1<<30); // clear
            hw.tem.det(i).enable = 0;
        }
    }
}
}

void resetTimingPll(TprCore& tpr)
{
  uint32_t v = tpr.CSR;
  v |= 0x80;
  tpr.CSR = v;
  usleep(10);
  v &= ~0x80;
  tpr.CSR = v;
  usleep(100);
  v |= 0x8;
  tpr.CSR = v;
  usleep(10);
  v &= ~0x8;
  tpr.CSR = v;
  usleep(100000);
}
