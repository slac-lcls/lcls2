#include "JungfrauData.hh"

#include "xtcdata/xtc/TimeStamp.hh"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <getopt.h>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <iostream>
#include <ctime>

constexpr const char* defHost = "10.0.0.10";
constexpr unsigned defPort = 8192;
constexpr double defRate = 1.; //Hz
constexpr unsigned defPacket = 0;
constexpr unsigned defPacketToDrop = 1;

static std::chrono::microseconds secs_to_us(double secs)
{
    std::chrono::duration<double> dursecs{secs};
    return std::chrono::duration_cast<std::chrono::microseconds>(dursecs);
}

static void showUsage(const char* p)
{
    std::cout << "Usage: " << p << " [-v|--version] [-h|--help]" << std::endl;
    std::cout << "[-H|--host <host>] [-P|--port <port>] [-r|--rate <rate>] [-n|--num <numframes>]" << std::endl;
    std::cout << "[-d|--drop <framenum>] [-p|--packet <packet>] [-N|--ndrops <num>]" << std::endl;
    std::cout << " Options:" << std::endl;
    std::cout << "    -H|--host     <host>                  set the receiver host ip (default: " << defHost << ")" << std::endl;
    std::cout << "    -P|--port     <port>                  set the receiver udp port number (default: " << defPort << ")" << std::endl;
    std::cout << "    -r|--rate     <rate>                  set the rate to generate frames at (default: " << defRate << " Hz)" << std::endl;
    std::cout << "    -n|--num      <num>                   set the number of frames to generate (default: forever)" << std::endl;
    std::cout << "    -d|--drop     <framenum>              set the frame numbers to drop packets from" << std::endl;
    std::cout << "    -p|--packet   <packet>                set the starting index of packet to drop" << std::endl;
    std::cout << "    -N|--ndrops   <num>                   set the number of packets to drop" << std::endl;
}

static void showVersion(const char* p)
{
    std::cout << "Version: " << p << " Ver 1.0.0" << std::endl;
}

using namespace Drp::JungfrauData;

int main(int argc, char* argv[])
{
    const char*         strOptions  = ":vhH:P:r:n:d:p:N:";
    const struct option loOptions[] =
    {
        {"version",     0, 0, 'v'},
        {"help",        0, 0, 'h'},
        {"host",        1, 0, 'H'},
        {"port",        1, 0, 'P'},
        {"rate",        1, 0, 'r'},
        {"num",         1, 0, 'n'},
        {"drop",        1, 0, 'd'},
        {"packet",      1, 0, 'p'},
        {"ndrops",      1, 0, 'N'},
    };

    int sock = -1;
    uint64_t maxFrames = 0;
    double rate = defRate;
    std::chrono::microseconds interval = secs_to_us(1./rate);
    unsigned port = defPort;
    const char* host = defHost;
    unsigned dropStart = defPacket;
    unsigned dropLen = defPacketToDrop;
    sockaddr_in sa;
    bool lUsage = false;
    std::unordered_set<uint64_t> drops;

    int optionIndex  = 0;
    while ( int opt = getopt_long(argc, argv, strOptions, loOptions, &optionIndex ) ) {
        if ( opt == -1 ) break;

        switch (opt) {
        case 'h':
            showUsage(argv[0]);
            return 0;
        case 'v':
            showVersion(argv[0]);
            return 0;
        case 'H':
            host = optarg;
            break;
        case 'P':
            port = std::strtoul(optarg, NULL, 0);
            break;
        case 'r':
            rate = std::strtod(optarg, NULL);
            interval = secs_to_us(1./rate);
            break;
        case 'n':
            maxFrames = std::strtoull(optarg, NULL, 0);
            break;
        case 'd':
            drops.insert(std::strtoull(optarg, NULL, 0));
            break;
        case 'p':
            dropStart = std::strtoul(optarg, NULL, 0);
            break;
        case 'N':
            dropLen = std::strtoul(optarg, NULL, 0);
            break;
        case '?':
            if (optopt)
                std::cerr << argv[0] << ": Unknown option: " << char(optopt) << std::endl;
            else
                std::cerr << argv[0] << ": Unknown option: " << argv[optind-1] << std::endl;
            lUsage = true;
            break;
        case ':':
            std::cerr << argv[0] << ": Missing argument for " << char(optopt) << std::endl;
            lUsage = true;
            break;
        default:
            lUsage = true;
            break;
        }       
    }

    if (dropStart >= PacketNum) {
        std::cout << "Invalid packet drop start: " << dropStart << " - must be less than " << PacketNum << "!" << std::endl;
        return 1;
    }

    if (dropStart + dropLen > PacketNum) {
        std::cout << "Invalid packet drop length: " << dropStart << " + " << dropLen << " = " << dropStart + dropLen
                  << " must be less than " << PacketNum << "!" << std::endl;
        return 1;
    }

    if (lUsage) {
        showUsage(argv[0]);
        return 1;
    }

    hostent* entries = gethostbyname(host);
    if (entries) {
        sock = socket(AF_INET, SOCK_DGRAM, 0);
      
        unsigned addr = htonl(*(in_addr_t*)entries->h_addr_list[0]);

        sa.sin_family = AF_INET;
        sa.sin_addr.s_addr = htonl(addr);
        sa.sin_port        = htons(port);
    } else {
        std::cerr << "Invalid hostname: " << host << std::endl;
        return 1;
    }

    JungfrauPacket packet;

    packet.header.framenum = 0;
    packet.header.exptime = 100;
    packet.header.moduleID = 0xdead;
    packet.header.detectortype = 3;
    packet.header.headerVersion = 2;

    uint16_t pixel_val = 0;
    // fille the packet
    for (size_t p = 0; p < PixelPerPacket; p++) {
        if (p % Rows == 0) pixel_val++;
        packet.data[p] = pixel_val;
    }

    std::cout << "Sending ";
    if (maxFrames > 0) {
        std::cout << maxFrames << " ";
    }
    std::cout << "packets at " << rate << " Hz" << std::endl;
    if (!drops.empty()) {
        std::cout << "Dropping packets for the following frames:";
        for (const auto& num : drops) {
            std::cout << " " << num;   
        }
        std::cout << std::endl;
    }

    while ((maxFrames == 0) || (packet.header.framenum < maxFrames)) {
        packet.header.framenum++;
        bool drop = drops.find(packet.header.framenum) != drops.end();
        // fake timestamp to put in the packets - jungfrau ts normally in clock ticks
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);
        packet.header.timestamp = XtcData::TimeStamp(ts).value();


        for (uint32_t i=0; i < PacketNum; i++) {
            if (drop && (i >= dropStart) && (i < dropStart + dropLen)) {
                std::cout << "Dropping packet " << i << " from frame " << packet.header.framenum << "!" << std::endl;
                continue;
            }
            packet.header.packetnum = i;
            sendto(sock, &packet, PacketSize, 0, (struct sockaddr*) &sa, sizeof(sa));
        }

        std::cout << "Sent frame: " << packet.header.framenum << std::endl;

        std::this_thread::sleep_for(interval);
    }
}
