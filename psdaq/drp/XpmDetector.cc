
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
static bool lverbose = true;

namespace Drp {

XpmDetector::XpmDetector(Parameters* para, MemPool* pool) :
    Detector(para, pool)
{
    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return;
    }

    // Check timing reference clock, program if necessary
    unsigned ccnt0,ccnt1;
    dmaReadRegister(fd, 0x00C00028, &ccnt0);
    usleep(100000);
    dmaReadRegister(fd, 0x00C00028, &ccnt1);
    ccnt1 -= ccnt0;
    double clkr = double(ccnt1)*16.e-5;
    printf("Timing RefClk %f MHz\n", clkr);
    if (clkr < 180 || clkr > 190) {
      //  Flush I2C by reading the Mux
      unsigned mux;
      dmaReadRegister(fd, 0x00E00000, &mux);
      printf("I2C mux:  0x%x\n", mux); // Force the read not to be optimized out
      //  Set the I2C Mux
      dmaWriteRegister(fd, 0x00E00000, (1<<2));
      Si570 rclk(fd,0x00E00800);
      rclk.program();

      printf("Reset timing PLL\n");
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

    close(fd);
}

json XpmDetector::connectionInfo()
{
    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return json();
    }

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

      sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;

      unsigned id = 0xfb000000 |
        (ntohl(saddr->sin_addr.s_addr)&0xffff);
      dmaWriteRegister(fd, &tem->xma().txId, id);
    }

    //  Retrieve the timing link ID
    uint32_t reg;
    dmaReadRegister(fd, &tem->xma().rxId, &reg);

    close(fd);
    // there is currently a failure mode where the register reads
    // back as zero (incorrectly). This is not the best longterm
    // fix, but throw here to highlight the problem. - cpo
    if (!reg) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
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

    // FIXME make configureable
    m_length = 100;
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
    if (it != m_para->kwargs.end())
        m_length = stoi(it->second);

    int links = m_para->laneMask;

    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return;
    }

    int readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];

    Pds::Mmhw::TriggerEventManager* tem = new ((void*)0x00C20000) Pds::Mmhw::TriggerEventManager;
    for(unsigned i=0, l=links; l; i++) {
        Pds::Mmhw::TriggerEventBuffer& b = tem->det(i);
        if (l&(1<<i)) {
            dmaWriteRegister(fd, &b.enable, (1<<2)      );  // reset counters
            dmaWriteRegister(fd, &b.pauseThresh, 16     );
            dmaWriteRegister(fd, &b.group , readoutGroup);
            dmaWriteRegister(fd, &b.enable, 3           );  // enable
            l &= ~(1<<i);

            dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<30));  // clear
            dmaWriteRegister(fd, 0x00a00000+4*(i&3), (m_length&0xffffff) | (1<<31));  // enable
          }
      }

    close(fd);
}

}

void dmaReadRegister (int fd, uint32_t* addr, uint32_t* valp)
{
  uintptr_t addri = (uintptr_t)addr;
  dmaReadRegister(fd, addri&0xffffffff, valp);
  if (lverbose)
    printf("[%08lx] = %08x\n",addri,*valp);
}

void dmaWriteRegister(int fd, uint32_t* addr, uint32_t val)
{
  uintptr_t addri = (uintptr_t)addr;
  dmaWriteRegister(fd, addri&0xffffffff, val);
  if (lverbose)
    printf("[%08lx] %08x\n",addri,val);
}
