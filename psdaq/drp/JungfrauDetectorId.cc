#include "JungfrauDetectorId.hh"

#include <fstream>
#include <algorithm>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Telnet protocol constants (RFC 854).
const unsigned char IAC  = 255;
const unsigned char DONT = 254;
const unsigned char DO   = 253;
const unsigned char WONT = 252;
const unsigned char WILL = 251;
const unsigned char SB   = 250;
const unsigned char SE   = 240;

// The interface whose hardware address we report.
const char* const kInterface = "eth0";

// The shell prompt the device presents; both phases end when we see it.
//
// BusyBox hush builds the prompt as "$USER:/> ", and USER only gets set if we
// accept NEW-ENVIRON (option 39) and hand over a user name. Since we refuse all
// options, USER is empty and the device prints a bare ":/>" -- an interactive
// telnet, which does negotiate, is the reason you see "root:/>" by hand. Match
// only the invariant tail so either form is recognised.
const char* const kPrompt = ":/>";

const int kPromptMs   = 10000;  // ceiling on reaching the first prompt
const int kCommandMs  = 15000;  // ceiling on the command completing
const int kConnectSecs = 10;

int64_t nowMs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

int connectTo(const char* host, const char* port)
{
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* res = NULL;
    int err = getaddrinfo(host, port, &hints, &res);
    if (err != 0) {
        fprintf(stderr, "Error: telnet - cannot resolve %s:%s - %s\n", host, port, gai_strerror(err));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo* ai = res; ai != NULL; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0)
            continue;

        struct timeval tv;
        tv.tv_sec = kConnectSecs;
        tv.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0)
            break;

        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        fprintf(stderr, "Error: telent - cannot connect to %s:%s - %s\n", host, port, strerror(errno));
        return -1;
    }

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

bool writeAll(int fd, const void* buf, size_t len)
{
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

// Consumes telnet IAC sequences from the incoming byte stream, appending only
// the payload to `out` and queueing our refusals into `reply`.
class TelnetFilter {
public:
    TelnetFilter() : m_state(DATA), m_command(0) {}

    void feed(const unsigned char* buf, size_t len, std::string& out, std::string& reply)
    {
        for (size_t i = 0; i < len; ++i) {
            unsigned char c = buf[i];
            switch (m_state) {
            case DATA:
                if (c == IAC)
                    m_state = SAW_IAC;
                else if (c != '\r')  // CR LF -> LF
                    out.push_back(static_cast<char>(c));
                break;

            case SAW_IAC:
                if (c == IAC) {  // escaped 0xFF
                    out.push_back(static_cast<char>(IAC));
                    m_state = DATA;
                } else if (c == DO || c == DONT || c == WILL || c == WONT) {
                    m_command = c;
                    m_state = SAW_COMMAND;
                } else if (c == SB) {
                    m_state = IN_SUBNEG;
                } else {
                    m_state = DATA;  // two-byte command, nothing to do
                }
                break;

            case SAW_COMMAND:
                // Refuse everything: the peer wanting to enable an option gets
                // DONT, the peer asking us to enable one gets WONT. DONT and
                // WONT are already refusals, so they need no answer -- replying
                // to them is what makes two peers negotiate in a loop forever.
                if (m_command == WILL || m_command == DO) {
                    reply.push_back(static_cast<char>(IAC));
                    reply.push_back(static_cast<char>(m_command == WILL ? DONT : WONT));
                    reply.push_back(static_cast<char>(c));
                }
                m_state = DATA;
                break;

            case IN_SUBNEG:
                if (c == IAC)
                    m_state = SUBNEG_IAC;
                break;

            case SUBNEG_IAC:
                m_state = (c == SE) ? DATA : IN_SUBNEG;
                break;
            }
        }
    }

private:
    enum State { DATA, SAW_IAC, SAW_COMMAND, IN_SUBNEG, SUBNEG_IAC };
    State m_state;
    unsigned char m_command;
};

// Reads (answering negotiation as it goes) until `kPrompt` shows up in `out`,
// appending payload bytes to it. Returns the offset of the prompt, or
// std::string::npos on timeout, remote close, or error.
size_t readToPrompt(int fd, TelnetFilter& filter, std::string& out, int timeoutMs)
{
    const int64_t deadline = nowMs() + timeoutMs;
    const size_t promptLen = strlen(kPrompt);

    // Only rescan the tail: a prompt can straddle two reads, but never more
    // than promptLen-1 bytes of already-scanned text.
    size_t scanned = 0;

    while (true) {
        size_t at = out.find(kPrompt, scanned);
        if (at != std::string::npos)
            return at;
        scanned = (out.size() >= promptLen) ? out.size() - promptLen + 1 : 0;

        int64_t remaining = deadline - nowMs();
        if (remaining <= 0) {
            fprintf(stderr, "Error: telnet - timed out waiting for prompt \"%s\"\n", kPrompt);
            return std::string::npos;
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        struct timeval tv;
        tv.tv_sec = remaining / 1000;
        tv.tv_usec = (remaining % 1000) * 1000;

        int rc = select(fd + 1, &rfds, NULL, NULL, &tv);
        if (rc < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "Error: telnet - select failed: %s\n", strerror(errno));
            return std::string::npos;
        }
        if (rc == 0) {
            fprintf(stderr, "Error: telnet - timed out waiting for prompt \"%s\"\n", kPrompt);
            return std::string::npos;
        }

        unsigned char buf[4096];
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n == 0) {
            fprintf(stderr, "Error: telnet - connection closed before prompt \"%s\"\n", kPrompt);
            return std::string::npos;
        }
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "Error: telnet - recv failed: %s\n", strerror(errno));
            return std::string::npos;
        }

        std::string reply;
        filter.feed(buf, static_cast<size_t>(n), out, reply);
        if (!reply.empty() && !writeAll(fd, reply.data(), reply.size())) {
            fprintf(stderr, "Error: telnet - send failed: %s\n", strerror(errno));
            return std::string::npos;
        }
    }
}

// Pulls the hardware address out of the `iface` block of ifconfig output.
// Interface blocks start in column zero and their continuation lines are
// indented, so a block runs until the next unindented line.
bool extractHwAddr(const std::string& text, const char* iface, std::string& mac)
{
    const size_t ifaceLen = strlen(iface);
    const char* const kTag = "HWaddr";
    bool inBlock = false;

    for (size_t pos = 0; pos < text.size(); ) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos)
            eol = text.size();
        const std::string line = text.substr(pos, eol - pos);
        pos = eol + 1;

        if (!line.empty() && !isspace(static_cast<unsigned char>(line[0]))) {
            // A new interface block begins here; it is ours only if the name
            // matches exactly, so "eth0" does not pick up "eth0:1".
            inBlock = line.compare(0, ifaceLen, iface) == 0 &&
                      (line.size() == ifaceLen ||
                       isspace(static_cast<unsigned char>(line[ifaceLen])));
        }
        if (!inBlock)
            continue;

        size_t at = line.find(kTag);
        if (at == std::string::npos)
            continue;

        // The address is the next whitespace-delimited token after the tag.
        size_t start = line.find_first_not_of(" \t", at + strlen(kTag));
        if (start == std::string::npos)
            continue;
        size_t end = line.find_first_of(" \t", start);
        mac = line.substr(start, (end == std::string::npos) ? std::string::npos
                                                            : end - start);
        return !mac.empty();
    }
    return false;
}

// Drops the shell's echo of `cmd` from the front of `text`, if present.
void stripEcho(std::string& text, const char* cmd)
{
    size_t start = text.find_first_not_of("\r\n");
    if (start == std::string::npos)
        return;
    if (text.compare(start, strlen(cmd), cmd) != 0)
        return;

    size_t eol = text.find('\n', start + strlen(cmd));
    text.erase(0, (eol == std::string::npos) ? text.size() : eol + 1);
}

}  // namespace

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
        if (it == _arp.end()) {
            // if not in arp try directly connecting to the detector
            load(ipAddr, "23");

            it = _arp.find(ipAddr);
        }
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

void JungfrauIdLookup::load(const std::string& hostname, const std::string& port)
{
    int fd = connectTo(hostname.c_str(), port.c_str());
    if (fd < 0)
        return;

    TelnetFilter filter;

    // Phase 1: the server pushes its IAC options (refused inside readToPrompt)
    // and its banner; we wait for the prompt before typing anything.
    std::string banner;
    if (readToPrompt(fd, filter, banner, kPromptMs) == std::string::npos) {
        close(fd);
        return;
    }

    // Phase 2: run the command; its output is everything up to the next prompt.
    const char* kCommand = "ifconfig";
    std::string line = std::string(kCommand) + "\r\n";
    if (!writeAll(fd, line.data(), line.size())) {
        fprintf(stderr, "Error: telnet - send failed: %s\n", strerror(errno));
        close(fd);
        return;
    }

    std::string output;
    size_t promptAt = readToPrompt(fd, filter, output, kCommandMs);
    if (promptAt == std::string::npos) {
        close(fd);
        return;
    }
    output.erase(promptAt);  // drop the trailing prompt
    stripEcho(output, kCommand);

    // Log out cleanly so the device does not keep the session open; these
    // devices often accept only one telnet session at a time.
    const char* kExit = "exit\r\n";
    writeAll(fd, kExit, strlen(kExit));
    close(fd);

    std::string mac;
    if (!extractHwAddr(output, kInterface, mac)) {
        fprintf(stderr, "Error: telnet - no HWaddr found for %s in ifconfig output:\n%s\n",
                kInterface, output.c_str());
        return;
    }

    // Set all the letters in the mac to lower case
    for (char &c : mac) {
        c = std::tolower(static_cast<unsigned char>(c));
    }

    _arp[hostname] = mac;
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
