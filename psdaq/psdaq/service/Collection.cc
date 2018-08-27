#include <vector>
#include <iostream>
#include <iomanip>
#include <functional>
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
            printf("address %s: <%s>\n", ifa->ifa_name, host);
        }
    }
    freeifaddrs(ifaddr);
    return std::string(host);
}

Collection::Collection(const std::string& manager_hostname, 
                       int platform, const std::string& level) : m_level(level)
{
    const int base_port = 29980;
    m_context = zmq_ctx_new();
    char buffer[1024];
    m_push = zmq_socket(m_context, ZMQ_PUSH);
    snprintf(buffer, 1024, "tcp://%s:%d", manager_hostname.c_str(), base_port + platform);
    if (zmq_connect(m_push, buffer) == -1) {
        perror("zmq_connect");
    }

    m_sub = zmq_socket(m_context, ZMQ_SUB);
    if (zmq_setsockopt(m_sub, ZMQ_SUBSCRIBE, "", 0) == -1) {
        perror("zmq_setsockopt(ZMQ_SUBSCRIBE)");
    }
    sprintf(buffer, "tcp://%s:%d", manager_hostname.c_str(), base_port + 10 + platform);
    printf("%s\n", buffer);
    if (zmq_connect(m_sub, buffer) == -1) {
        perror("zmq_connect");
    }

    m_handle_request["plat"] = std::bind(&Collection::handle_plat, this, 
                                         std::placeholders::_1);
    m_handle_request["alloc"] = std::bind(&Collection::handle_alloc, this, 
                                          std::placeholders::_1);
    m_handle_request["connect"] = std::bind(&Collection::handle_connect, this, 
                                            std::placeholders::_1);
}

Collection::~Collection()
{
    zmq_close(m_push);
    zmq_close(m_sub);
    zmq_ctx_destroy(m_context);
}

void Collection::connect()
{
    while (1) {
        json msg = recv_json(m_sub);
        std::string key = msg["header"]["key"];
        std::cout << std::setw(4) << msg << "\n\n";
        printf("received key = %s\n", key.c_str());
        m_handle_request[key](msg);
        if (key == "connect") {
            break;
        }
    }
}

void Collection::handle_plat(json& msg)
{
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    int pid = getpid();
    m_id = std::hash<std::string>{}(std::string(hostname) + std::to_string(pid));
    json body;
    body[m_level] = {{"proc_info", {{"host", hostname}, {"pid", pid}}}};
    json reply = create_msg("plat", msg["header"]["msg_id"], m_id, body); 
    std::string s = reply.dump();
    zmq_send(m_push, s.c_str(), s.length(), 0);
}

void Collection::handle_alloc(json& msg)
{
    // partition_info = json::parse(s);
    std::string infiniband_address = get_infiniband_address();
    printf("infiniband address %s\n", infiniband_address.c_str());
    json body = {{m_level, {{"connect_info", {{"infiniband", infiniband_address}}}}}};
    json reply = create_msg("alloc", msg["header"]["msg_id"], m_id, body);
    std::string s = reply.dump();
    zmq_send(m_push, s.c_str(), s.length(), 0);
    m_state = "alloc";
}

void Collection::handle_connect(json& msg)
{
    // ignore message if not in alloc state
    if (m_state == "alloc") {
        // FIXME actually check that connectionn is successful before sending response
        cmstate = msg["body"];
        json body = json({});
        json reply = create_msg("ok", msg["header"]["msg_id"], m_id, body);
        std::cout << std::setw(4) << reply << "\n\n";
        std::string s = reply.dump();
        zmq_send(m_push, s.c_str(), s.length(), 0);
        m_state = "connect";
    }
}

json create_msg(const std::string& key, const std::string& msg_id, size_t sender_id, json& body)
{
    json msg;
    msg["header"] = { {"key", key}, {"msg_id", msg_id}, {"sender_id", sender_id} };
    msg["body"] = body;
    return msg;
}

json recv_json(void* socket)
{
    ZmqMessage frame;
    int rc = zmq_msg_recv(&frame.msg, socket, 0);
    assert (rc != -1);
    return json::parse((char*)frame.data());

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
