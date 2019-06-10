#include <errno.h>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <limits.h>
#include <ifaddrs.h>
#include <linux/if_packet.h>
#include <netinet/ether.h>
#include <netdb.h>
#include "Collection.hh"

using json = nlohmann::json;

json createMsg(const std::string& key, const std::string& msg_id, size_t sender_id, json& body)
{
    json msg;
    msg["header"] = { {"key", key}, {"msg_id", msg_id}, {"sender_id", sender_id} };
    msg["body"] = body;
    return msg;
}

json createAsyncErrMsg(const std::string& alias, const std::string& errMsg)
{
    json body = json({});
    body["err_info"] = alias + ": " + errMsg;
    return createMsg("error", "0", 0, body);
}

std::string getNicIp()
{
    struct ifaddrs* ifaddr;
    getifaddrs(&ifaddr);

    char host[NI_MAXHOST];
    char* interface_name = nullptr;
    char* ethernet_name  = nullptr;
    // find name of first infiniband, otherwise fall back ethernet
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }
        int family = ifa->ifa_addr->sa_family;
        if (family == AF_PACKET) {
            struct sockaddr_ll* s = (struct sockaddr_ll*)ifa->ifa_addr;
            if (s->sll_hatype == ARPHRD_INFINIBAND) {
                if (!interface_name) interface_name = ifa->ifa_name;
            }
            else if (s->sll_hatype == ARPHRD_ETHER) {
                if (!ethernet_name) ethernet_name  = ifa->ifa_name;
            }
        }
    }
    if (interface_name == nullptr) {
        printf("Warning: No infiniband device found!");
        if (ethernet_name == nullptr) {
            printf("  And no ethernet either!\n");
            return std::string();
        }
        printf("  Falling back to ethernet.\n");
        interface_name = ethernet_name;
    }

    // get address of the first infiniband device found above
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }
        int family = ifa->ifa_addr->sa_family;

        if ((family == AF_INET) && (strcmp(ifa->ifa_name, interface_name)==0)) {
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

ZmqSocket::ZmqSocket(ZmqContext* context, int type) : m_context(context)
{
    socket = zmq_socket((*m_context)(), type);
}

void ZmqSocket::connect(const std::string& host)
{
    int rc = zmq_connect(socket, host.c_str());
    if (rc != 0) {
        throw std::runtime_error{"zmq_connect failed with error " + std::to_string(rc)};
    }
}

void ZmqSocket::bind(const std::string& host)
{
    int rc = zmq_bind(socket, host.c_str());
    if (rc != 0) {
        throw std::runtime_error{"zmq_bind failed with error " + std::to_string(rc)};
    }
}

void ZmqSocket::setsockopt(int option, const void* optval, size_t optvallen)
{
    int rc = zmq_setsockopt(socket, option, optval, optvallen);
    if (rc != 0) {
        throw std::runtime_error{"zmq_setsockopt failed with error " + std::to_string(rc)};
    }
}

std::string ZmqSocket::recv()
{
    ZmqMessage frame;
    int rc = zmq_msg_recv(&frame.msg, socket, 0);
    if (rc == -1) {
        fprintf(stderr,"Collection.cc: zmq_msg_recv bad return value %d\n",rc);
        fprintf(stderr,"This can happen when a process is sent a signal.  Exiting.\n");
        return std::string();
    }
    return std::string((char*)frame.data(), frame.size());
}

json ZmqSocket::recvJson()
{
    ZmqMessage frame;
    int rc = zmq_msg_recv(&frame.msg, socket, 0);
    if (rc == -1) {
        fprintf(stderr,"Collection.cc: zmq_msg_recv bad return value %d\n",rc);
        fprintf(stderr,"This can happen when a process is sent a signal.  Exiting.\n");
        return json({});
    }
    char* begin = (char*)frame.data();
    char* end   = begin + frame.size();
    return json::parse(begin, end);
}

std::vector<ZmqMessage> ZmqSocket::recvMultipart()
{
    std::vector<ZmqMessage> frames;
    int more;
    size_t more_size = sizeof(more);
    do {
        ZmqMessage frame;
        int rc = zmq_msg_recv(&frame.msg, socket, 0);
        if (rc == -1) {
            fprintf(stderr,"Collection.cc: zmq_msg_recv bad return value %d\n",rc);
            fprintf(stderr,"This can happen when a process is sent a signal.  Exiting.\n");
            frames.clear();
            break;                      // Let application wind down
        }
        frames.emplace_back(std::move(frame));
        rc = zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size);
        if (rc != 0) {
            fprintf(stderr,"Collection.cc: zmq_getsockopt bad return value %d\n",rc);
            fprintf(stderr,"This can happen when a process is sent a signal.  Exiting.\n");
            frames.clear();
            break;                      // Let application wind down
        }
    } while (more);
    return frames;
}

void ZmqSocket::send(const std::string& msg)
{
    int ret = zmq_send(socket, msg.c_str(), msg.length(), 0);
    if (ret == -1) {
        std::cout<<"Error sending zmq message:  "<<msg<<'\n';
    }
}

int ZmqSocket::poll(short events, long timeout)
{
    zmq_pollitem_t item;
    item.socket = socket;
    item.events = events;
    return zmq_poll(&item, 1, timeout);
}


CollectionApp::CollectionApp(const std::string &managerHostname,
                             int platform,
                             const std::string &level,
                             const std::string &alias) :
    m_level(level),
    m_alias(alias),
    m_pushSocket{&m_context, ZMQ_PUSH},
    m_subSocket{&m_context, ZMQ_SUB},
    m_inprocRecv{&m_context, ZMQ_PAIR}
{
    const int base_port = 29980;

    m_pushSocket.connect({"tcp://" + managerHostname + ":" + std::to_string(base_port + platform)});

    m_subSocket.connect({"tcp://" + managerHostname + ":" + std::to_string(base_port + 10 + platform)});
    m_subSocket.setsockopt(ZMQ_SUBSCRIBE, "all", 3);
    std::cout<<std::string{"tcp://" + managerHostname + ":" + std::to_string(base_port + 10 + platform)}<<std::endl;
    m_inprocRecv.bind("inproc://drp");

    // register callbacks
    m_handleMap["rollcall"] = std::bind(&CollectionApp::handleRollcall, this, std::placeholders::_1);
    m_handleMap["alloc"] = std::bind(&CollectionApp::handleAlloc, this, std::placeholders::_1);
    m_handleMap["connect"] = std::bind(&CollectionApp::handleConnect, this, std::placeholders::_1);
    m_handleMap["disconnect"] = std::bind(&CollectionApp::handleDisconnect, this, std::placeholders::_1);
    m_handleMap["reset"] = std::bind(&CollectionApp::handleReset, this, std::placeholders::_1);
    m_handleMap["configure"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["unconfigure"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["enable"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["disable"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["configUpdate"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
}

void CollectionApp::handleRollcall(const json &msg)
{
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    int pid = getpid();
    m_id = std::hash<std::string>{}(std::string(hostname) + std::to_string(pid));
    json body;
    body[m_level] = {{"proc_info", {{"host", hostname}, {"pid", pid}, {"alias", m_alias}}}};
    json answer = createMsg("rollcall", msg["header"]["msg_id"], m_id, body);
    reply(answer);
}

void CollectionApp::handleAlloc(const json &msg)
{
    // check if own id is in included in the msg
    auto it = std::find(msg["body"]["ids"].begin(), msg["body"]["ids"].end(), m_id);
    if (it != msg["body"]["ids"].end()) {
        std::cout<<"subscribing to partition\n";
        m_subSocket.setsockopt(ZMQ_SUBSCRIBE, "partition", 9);

        json info = connectionInfo();
        json body = {{m_level, info}};
        std::cout << "body handleAlloc  " << std::setw(4) << body << "\n\n";
        json answer = createMsg("alloc", msg["header"]["msg_id"], m_id, body);

        reply(answer);
    }
    else {
        m_subSocket.setsockopt(ZMQ_UNSUBSCRIBE, "partition", 9);
    }
}

void CollectionApp::reply(const json& msg)
{
    m_pushSocket.send(msg.dump());
}

void CollectionApp::run()
{
    while (1) {
        zmq_pollitem_t items[] = {
            { m_subSocket.socket, 0, ZMQ_POLLIN, 0 },
            { m_inprocRecv.socket, 0, ZMQ_POLLIN, 0 }
        };
        if (zmq_poll(items, 2, -1) == -1) {
            if (errno == EINTR)  break;
        }

        // received zeromq message from the collection
        if (items[0].revents & ZMQ_POLLIN) {
            std::vector<ZmqMessage> frames = m_subSocket.recvMultipart();
            if (frames.size() < 2)  break;  // Revisit: Terminate condition
            char* begin = (char*)frames[1].data();
            char* end = begin + frames[1].size();
            json msg = json::parse(begin, end);
            std::string topic((char*)frames[0].data(), frames[0].size());
            std::cout<<"topic:  "<<topic<<'\n';

            std::string key = msg["header"]["key"];
            std::cout<<"received key = "<<key<<'\n';
            std::cout << std::setw(4) << msg << "\n\n";
            if (m_handleMap.find(key) == m_handleMap.end()) {
                std::cout<<"unknown key  "<<key<<'\n';
            }
            else {
                m_handleMap[key](msg);
            }
        }

        // received inproc message from timing system
        if (items[1].revents & ZMQ_POLLIN) {
            // forward message to contol system with pulseId as the message id
            std::string pulseId = m_inprocRecv.recv();
            std::cout<<"inproc message received  "<<pulseId<<'\n';
            json body;
            json answer = createMsg("timingTransition", pulseId, getId(), body);
            reply(answer);
        }
    }
}
