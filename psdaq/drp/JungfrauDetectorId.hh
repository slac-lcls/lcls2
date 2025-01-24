#pragma once

#include <cstdint>
#include <string>
#include <map>

namespace Drp {
  namespace Jungfrau {
    class DetId {
      public:
        DetId();
        DetId(uint64_t id);
        DetId(uint64_t board, uint64_t module);
        DetId(const std::string& mac, uint64_t module);
        ~DetId();
        uint64_t full() const;
        uint64_t board() const;
        uint64_t module() const;
      private:
        uint64_t _id;
    };

    typedef std::map<std::string, std::string> ArpCache;
    typedef ArpCache::const_iterator ArpCacheIter;

    class DetIdLookup {
      public:
        DetIdLookup();
        ~DetIdLookup();

        bool has(const std::string& hostname);
        const std::string& operator[](const std::string& hostname);

        static std::string host_to_ip(const std::string& hostname);
        static uint64_t mac_to_hex(std::string mac);
      private:
        void load();

        ArpCache _arp;
    };
  }
}
