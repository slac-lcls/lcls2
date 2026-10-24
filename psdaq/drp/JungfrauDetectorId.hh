/**
 * @file
 * @brief JungfrauId, a 64-bit board-plus-module identifier, and JungfrauIdLookup, which finds the MAC address of a Jungfrau host.
 */
#pragma once

#include <cstdint>
#include <string>
#include <map>

namespace Drp {

/** 64-bit Jungfrau ID: a board value in the upper bits and a 16-bit module number in the low 16 bits. */
class JungfrauId {
    public:
        /** Construct with ID 0. */
        JungfrauId();
        /** Construct from a full 64-bit ID. */
        JungfrauId(uint64_t id);
        /** Construct the ID as board shifted left by 16 bits ORed with the low 16 bits of module. */
        JungfrauId(uint64_t board, uint64_t module);
        /** Like JungfrauId(board, module), with the board value parsed from the MAC address string mac by JungfrauIdLookup::mac_to_hex(). */
        JungfrauId(const std::string& mac, uint64_t module);
        /** Does nothing; empty body. */
        ~JungfrauId();
        /** Return the full 64-bit ID. */
        uint64_t full() const;
        /** Return the ID shifted right by 16 bits (the board part). */
        uint64_t board() const;
        /** Return the low 16 bits of the ID (the module part). */
        uint64_t module() const;
    private:
        uint64_t _id;
};

/** Map from IP address string to MAC address string, filled by JungfrauIdLookup from /proc/net/arp or from the device. */
typedef std::map<std::string, std::string> ArpCache;
/** Const iterator over an ArpCache. */
typedef ArpCache::const_iterator ArpCacheIter;

/** Finds the MAC address of a host: first in /proc/net/arp, then by logging in to the device with telnet and reading the eth0 HWaddr from its ifconfig output. */
class JungfrauIdLookup {
    public:
        /** Construct with an empty cache. */
        JungfrauIdLookup();
        /** Does nothing; empty body. */
        ~JungfrauIdLookup();

        /** Return true if a MAC address is known for the IP address of hostname. If it is not cached, /proc/net/arp is reloaded, and if still missing the device is queried over telnet (port 23). */
        bool has(const std::string& hostname);
        /** Return the cached MAC address for the IP address of hostname; a missing entry is inserted as an empty string. */
        const std::string& operator[](const std::string& hostname);

        /** Resolve hostname with gethostbyname() and return its first address as a dotted IPv4 string. A failed lookup is not checked (null pointer dereference). */
        static std::string host_to_ip(const std::string& hostname);
        /** Remove the colons from mac and parse the remaining text as a hexadecimal number with strtoul(). */
        static uint64_t mac_to_hex(std::string mac);
    private:
        void load();
        void load(const std::string& hostname, const std::string& port);

        ArpCache _arp;
};

}
