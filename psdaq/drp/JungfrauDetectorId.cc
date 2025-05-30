#include "JungfrauDetectorId.hh"

#include <fstream>
#include <algorithm>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

using namespace Drp;

static const uint64_t MOD_ID_BITS = 16;
static const uint64_t MOD_ID_MASK = (1<<MOD_ID_BITS) - 1;

JungfrauId::JungfrauId() :
    _id(0)
{}

JungfrauId::JungfrauId(uint64_t id) :
    _id(id)
{}

JungfrauId::JungfrauId(uint64_t board, uint64_t module) :
    _id((board<<MOD_ID_BITS) | (MOD_ID_MASK & module))
{}

JungfrauId::JungfrauId(const std::string& mac, uint64_t module) :
    _id((JungfrauIdLookup::mac_to_hex(mac)<<MOD_ID_BITS) | (MOD_ID_MASK & module))
{}

JungfrauId::~JungfrauId()
{}

uint64_t JungfrauId::full() const
{
    return _id;
}

 uint64_t JungfrauId::board() const
{
    return _id>>MOD_ID_BITS;
}

uint64_t JungfrauId::module() const
{
    return MOD_ID_MASK & _id;
}

JungfrauIdLookup::JungfrauIdLookup()
{}

JungfrauIdLookup::~JungfrauIdLookup()
{}

bool JungfrauIdLookup::has(const std::string& hostname)
{
    std::string ipAddr = host_to_ip(hostname);

    ArpCacheIter it = _arp.find(ipAddr);
    if (it == _arp.end()) {
        // refresh the arp cache and try again
        load();

        it = _arp.find(ipAddr);
    }

    return it != _arp.end();
}

const std::string& JungfrauIdLookup::operator[](const std::string& hostname)
{
    return _arp[host_to_ip(hostname)];
}

void JungfrauIdLookup::load()
{
    std::ifstream arpf("/proc/net/arp");
    if (arpf.is_open()) {
        std::string header, addr, mac, mask, dev;
        unsigned hw, flags;

        // ignore the header line
        std::getline(arpf, header);

        while(arpf >> addr >> std::hex >> hw >> flags >> mac >> mask >> dev) {
            _arp[addr] = mac;
        }

        arpf.close();
    }
}

std::string JungfrauIdLookup::host_to_ip(const std::string& hostname)
{
    return std::string(inet_ntoa(*(struct in_addr*)gethostbyname(hostname.c_str())->h_addr_list[0]));
}

uint64_t JungfrauIdLookup::mac_to_hex(std::string mac)
{
    mac.erase(std::remove(mac.begin(), mac.end(), ':'), mac.end());
    return strtoul(mac.c_str(), nullptr, 16);
}
