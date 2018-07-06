#include <vector>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <limits.h>
#include <ifaddrs.h>
#include <linux/if_packet.h>
#include <netinet/ether.h>
#include <netdb.h>
#include "Collection.hh"

std::string get_infiniband_address()
{
    struct ifaddrs* ifaddr;
    getifaddrs(&ifaddr);

    char host[NI_MAXHOST];
    char* infiniband_name = nullptr;
    // find name of first infiniband device
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }
        int family = ifa->ifa_addr->sa_family;
        if (family == AF_PACKET) {
            struct sockaddr_ll* s = (struct sockaddr_ll*)ifa->ifa_addr;
            if (s->sll_hatype == ARPHRD_INFINIBAND) {
                infiniband_name = ifa->ifa_name;
                break;
            }
        }
    }
    if (infiniband_name == nullptr) {
        printf("Warning: No infiniband device found!\n");
        return std::string();
    }

    // get address of the first infiniband device found above
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }
        int family = ifa->ifa_addr->sa_family;

        if ((family == AF_INET) && (strcmp(ifa->ifa_name, infiniband_name)==0)) {
            int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                                 host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
            if (s != 0) {
                printf("getnameinfo() failed: %s\n", gai_strerror(s));
            }
            // printf("address %s: <%s>\n", ifa->ifa_name, host);
        }
    }
    freeifaddrs(ifaddr);
    return std::string(host);
}

Collection::Collection()
{
    const int base_port = 29980;
    m_context = zmq_ctx_new();
    char buffer[1024];
    m_dealer = zmq_socket(m_context, ZMQ_DEALER);
    snprintf(buffer, 1024, "tcp://%s:%d", "localhost", base_port);
    if (zmq_connect(m_dealer, buffer) == -1) {
        perror("zmq_connect");
    }

    m_sub = zmq_socket(m_context, ZMQ_SUB);
    if (zmq_setsockopt(m_sub, ZMQ_SUBSCRIBE, "", 0) == -1) {
        perror("zmq_setsockopt(ZMQ_SUBSCRIBE)");
    }
    sprintf(buffer, "tcp://%s:%d", "localhost", base_port + 10);
    printf("%s\n", buffer);
    if (zmq_connect(m_sub, buffer) == -1) {
        perror("zmq_connect");
    }
}

Collection::~Collection()
{
    zmq_close(m_dealer);
    zmq_close(m_sub);
    zmq_ctx_destroy(m_context);
}

std::vector<ZmqMessage> Collection::wait_for_msg(const std::string& key)
{
    zmq_pollitem_t items[] = {
        {m_dealer, 0, ZMQ_POLLIN, 0},
        {m_sub, 0, ZMQ_POLLIN, 0}};

    while (1) {
        zmq_poll(items, 2, -1);
        void* socket;
        // check for event on DEALER socket
        if (items[0].revents) {
            socket = m_dealer;
        }
        // check for event on SUB socket
        if (items[1].revents) {
            socket = m_sub;
        }
        auto frames = recv_multipart(socket);
        std::string msg_key((char*)frames[0].data(), frames[0].size());
        printf("received key = %s\n", msg_key.c_str());
        if (msg_key == key) {
            return frames;
        }
        else {
            printf("Unexpected key %s", msg_key.c_str());
        }
    }
}

json Collection::partition_info(const std::string& level)
{
    auto frames = wait_for_msg("PLAT");

    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    int pid = getpid();
    json response;
    response[level] = {{"procInfo", {{"host", hostname}, {"pid", pid}}}};
    std::string s = response.dump();
    zmq_send(m_dealer, "HELLO", 5, ZMQ_SNDMORE);
    zmq_send(m_dealer, s.c_str(), s.length(), 0);

    frames = wait_for_msg("ALLOC");
    std::string answer((char*)frames[1].data(), frames[1].size());
    return json::parse(answer);
}

json Collection::connection_info(json& msg)
{
    zmq_send(m_dealer, "CONNECTINFO", 11, ZMQ_SNDMORE);
    std::string s = msg.dump();
    zmq_send(m_dealer, s.c_str(), s.length(), 0);

    auto frames = wait_for_msg("CONNECT");
    std::string answer((char*)frames[1].data(), frames[1].size());
    return json::parse(answer);
}

std::vector<ZmqMessage> recv_multipart(void* socket)
{
    std::vector<ZmqMessage> msgs;
    int more;
    size_t more_size = sizeof(more);
    do {
        ZmqMessage part;
        int rc = zmq_msg_recv(&part.msg, socket, 0);
        assert (rc != -1);
        msgs.emplace_back(std::move(part));
        rc = zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size);
        assert(rc == 0);
    } while (more);
    return msgs;
}
