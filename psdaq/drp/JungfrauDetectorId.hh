#pragma once

#include <cstdint>
#include <string>
#include <map>

namespace Drp {

class JungfrauId {
    public:
        JungfrauId();
        JungfrauId(uint64_t id);
        JungfrauId(uint64_t board, uint64_t module);
        JungfrauId(const std::string& mac, uint64_t module);
        ~JungfrauId();
        uint64_t full() const;
        uint64_t board() const;
        uint64_t module() const;
    private:
        uint64_t _id;
};

typedef std::map<std::string, std::string> ArpCache;
typedef ArpCache::const_iterator ArpCacheIter;

class JungfrauIdLookup {
    public:
        JungfrauIdLookup();
        ~JungfrauIdLookup();

        bool has(const std::string& hostname);
        const std::string& operator[](const std::string& hostname);

        static std::string host_to_ip(const std::string& hostname);
        static uint64_t mac_to_hex(std::string mac);
    private:
        void load();

        ArpCache _arp;
};

}
