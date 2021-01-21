
#include "XpmDetector.hh"
#include "Si570.hh"
#include "AxisDriver.h"
#include "DataDriver.h"
#include "psalg/utils/SysLog.hh"
#include "psdaq/mmhw/TriggerEventManager.hh"
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

static void dmaReadRegister (int, uint32_t*, uint32_t*);
static void dmaWriteRegister(int, uint32_t*, uint32_t);

namespace Drp {

XpmDetector::XpmDetector(Parameters* para, MemPool* pool) :
    Detector(para, pool)
{
    int fd = pool->fd();

    // Check timing reference clock, program if necessary
    unsigned ccnt0,ccnt1;
    dmaReadRegister(fd, 0x00C00028, &ccnt0);
    usleep(100000);
    dmaReadRegister(fd, 0x00C00028, &ccnt1);
    ccnt1 -= ccnt0;
    double clkr = double(ccnt1)*16.e-5;
    logging::info("Timing RefClk %f MHz\n", clkr);
    if (clkr < 180 || clkr > 190) {
      AxiVersion vsn;
      axiVersionGet(fd, &vsn);
      if (vsn.userValues[2]) {  // Only one PCIe interface has access to I2C bus
         logging::error("Si570 clock needs programming.  This PCIe interface has no I2C access.");
         return;
      }

      //  Flush I2C by reading the Mux
      unsigned mux;
      dmaReadRegister(fd, 0x00E00000, &mux);
      logging::info("I2C mux:  0x%x\n", mux); // Force the read not to be optimized out
      //  Set the I2C Mux
      dmaWriteRegister(fd, 0x00E00000, (1<<2));

      Si570 rclk(fd,0x00E00800);
      rclk.program();

      logging::info("Reset timing PLL\n");
      unsigned v;
      dmaReadRegister(fd, 0x00C00020, &v);
      v |= 0x80;
      dmaWriteRegister(fd, 0x00C00020, v);
      usleep(10);
      v &= ~0x80;
      dmaWriteRegister(fd, 0x00C00020, v);
      usleep(100);
      v |= 0x8;
      dmaWriteRegister(fd, 0x00C00020, v);
      usleep(10);
      v &= ~0x8;
      dmaWriteRegister(fd, 0x00C00020, v);
      usleep(100000);
    }
}

json XpmDetector::connectionInfo()
{
    int fd = m_pool->fd();

    Pds::Mmhw::TriggerEventManager* tem = new ((void*)0x00C20000) Pds::Mmhw::TriggerEventManager;

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
              dmaWriteRegister(fd, &tem->xma().txId, id);
              break;
          }
          result = result->ai_next;
      }

      if (!result) {
          logging::info("No 172.21 address found.  Defaulting");
          unsigned id = 0xfb000000;
          dmaWriteRegister(fd, &tem->xma().txId, id);
      }
    }

    //  Retrieve the timing link ID
    uint32_t reg;
    dmaReadRegister(fd, &tem->xma().rxId, &reg);

    // there is currently a failure mode where the register reads
    // back as zero or 0xffffffff (incorrectly). This is not the best
    // longterm fix, but throw here to highlight the problem. the
    // difficulty is that Matt says this register has to work
    // so that an automated software solution would know which
    // xpm TxLink's to reset (a chicken-and-egg problem) - cpo
    if (!reg || reg==0xffffffff) {
        logging::critical("XPM Remote link id register illegal value: 0x%x. Try XPM TxLink reset.",reg);
        abort();
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

    Pds::Mmhw::TriggerEventManager* tem = new ((void*)0x00C20000) Pds::Mmhw::TriggerEventManager;
    for(unsigned i=0, l=links; l; i++) {
        Pds::Mmhw::TriggerEventBuffer& b = tem->det(i);
        if (l&(1<<i)) {
            dmaWriteRegister(fd, &b.enable, (1<<2)      );  // reset counters
            dmaWriteRegister(fd, &b.pauseThresh, 16     );
            dmaWriteRegister(fd, &b.group , m_readoutGroup);
            dmaWriteRegister(fd, &b.enable, 3           );  // enable
            l &= ~(1<<i);

            dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<30));  // clear
            dmaWriteRegister(fd, 0x00a00000+4*(i&3), (m_length&0xffffff) | (1<<31));  // enable
          }
      }
}

unsigned XpmDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc)
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

    for(unsigned i=0, l=links; l; i++) {
        if (l&(1<<i)) {
          dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<30));  // clear
          l &= ~(1<<i);
        }
    }
}

}

void dmaReadRegister (int fd, uint32_t* addr, uint32_t* valp)
{
  uintptr_t addri = (uintptr_t)addr;
  dmaReadRegister(fd, addri&0xffffffff, valp);
  logging::debug("[%08lx] = %08x\n",addri,*valp);
}

void dmaWriteRegister(int fd, uint32_t* addr, uint32_t val)
{
  uintptr_t addri = (uintptr_t)addr;
  dmaWriteRegister(fd, addri&0xffffffff, val);
  logging::debug("[%08lx] %08x\n",addri,val);
}
