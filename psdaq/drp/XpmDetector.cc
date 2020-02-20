
#include "XpmDetector.hh"
#include "AxisDriver.h"
#include <unistd.h>
#include "psalg/utils/SysLog.hh"
#include "psdaq/mmhw/TriggerEventManager.hh"

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
}

json XpmDetector::connectionInfo()
{
    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return json();
    }

    Pds::Mmhw::TriggerEventManager* tem = new ((void*)0x00C20000) Pds::Mmhw::TriggerEventManager;

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
    int x = (reg >> 16) & 0xFF;
    int y = (reg >> 8) & 0xFF;
    int port = reg & 0xFF;
    std::string xpmIp = {"10.0." + std::to_string(x) + '.' + std::to_string(y)};
    json info = {{"xpm_ip", xpmIp}, {"xpm_port", port}};
    return info;
}

// setup up device to receive data over pgp
void XpmDetector::connect(const json& json, const std::string& collectionId)
{
    logging::info("XpmDetector connect");
    // FIXME make configureable
    int length = 100;
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
    if (it != m_para->kwargs.end())
        length = stoi(it->second);

    int links = m_para->laneMask;

    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return;
    }

    int readoutGroup = json["body"]["drp"][collectionId]["det_info"]["readout"];

    Pds::Mmhw::TriggerEventManager* tem = new ((void*)0x00C20000) Pds::Mmhw::TriggerEventManager;
    for(unsigned i=0, l=links; l; i++) {
      Pds::Mmhw::TriggerEventBuffer& b = tem->det(i);
      if (l&(1<<i)) {
        dmaWriteRegister(fd, &b.enable, (1<<2)      );  // reset counters
        dmaWriteRegister(fd, &b.pauseThresh, 16     );
        dmaWriteRegister(fd, &b.group , readoutGroup);
        dmaWriteRegister(fd, &b.enable, 3           );  // enable
        l &= ~(1<<i);
      }
    }

    dmaWriteRegister(fd, 0x00a00000, (1<<3));   // clear

    uint32_t v = ((length&0xffffff)<<4) | ((links&0xf)<<28);
    dmaWriteRegister(fd, 0x00a00000, v);

    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    logging::info("Configured readout group [%u], length [%u], links [%x]: [%x](%x)\n",
                  readoutGroup, length, links, v, w);
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
