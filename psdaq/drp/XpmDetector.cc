
#include "XpmDetector.hh"
#include "AxisDriver.h"
#include <unistd.h>
#include "psalg/utils/SysLog.hh"

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

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
    uint32_t reg;
    dmaReadRegister(fd, 0x00a00008, &reg);
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
    uint32_t v = ((readoutGroup&0xf)<<0) |
                  ((length&0xffffff)<<4) |
                  (links<<28);
    dmaWriteRegister(fd, 0x00a00000, v);
    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    logging::info("Configured readout group [%u], length [%u], links [%x]: [%x](%x)\n",
                  readoutGroup, length, links, v, w);
    for (unsigned i=0; i<4; i++) {
        if (links&(1<<i)) {
            // this is the threshold to assert deadtime (high water mark) for every link
            // 0x1f00 corresponds to 0x1f free buffers
            dmaWriteRegister(fd, 0x00800084+32*i, 0x1f00);
        }
    }
    close(fd);
}

}
