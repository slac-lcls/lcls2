
#include "XpmDetector.hh"
#include "AxisDriver.h"
#include <unistd.h>

using namespace XtcData;
using json = nlohmann::json;

namespace Drp {

XpmDetector::XpmDetector(Parameters* para, MemPool* pool) :
    Detector(para, pool)
{
}

json XpmDetector::connectionInfo()
{
    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<< m_para->device << '\n';
        return json();
    }
    uint32_t reg;
    dmaReadRegister(fd, 0x00a00008, &reg);
    close(fd);
    int x = (reg >> 16) & 0xFF;
    int y = (reg >> 8) & 0xFF;
    int port = reg & 0xFF;
    std::string xpmIp = {"10.0." + std::to_string(x) + '.' + std::to_string(y)};
    json info = {{"xpm_ip", xpmIp}, {"xpm_port", port}};
    return info;
}

// setup up device to receive data over pgp
void XpmDetector::connect(const json& json)
{
    std::cout<<"XpmDetector connect\n";
    // FIXME make configureable
    int length = 100;
    int links = m_para->laneMask;

    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<< m_para->device << '\n';
        return;
    }
    int partition = m_para->partition;
    uint32_t v = ((partition&0xf)<<0) |
                  ((length&0xffffff)<<4) |
                  (links<<28);
    dmaWriteRegister(fd, 0x00a00000, v);
    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    printf("Configured partition [%u], length [%u], links [%x]: [%x](%x)\n",
           partition, length, links, v, w);
    for (unsigned i=0; i<4; i++) {
        if (links&(1<<i)) {
            dmaWriteRegister(fd, 0x00800084+32*i, 0x1f00);
        }
    }
    close(fd);
}

}
