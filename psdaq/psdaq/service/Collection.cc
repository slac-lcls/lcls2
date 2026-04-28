#include <errno.h>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>
#include <ifaddrs.h>
#include <linux/if_packet.h>
#include <netinet/ether.h>
#include <netdb.h>
#include <net/if.h>
#include <sys/resource.h>
#include <ctime>
#include <csignal>
#include <sys/stat.h>
#include "Collection.hh"
#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

using json = nlohmann::json;

static std::string      lAlias;
static struct sigaction lSigAction;
static void (*lShutdownAction)();


// https://stackoverflow.com/questions/17965/how-to-generate-a-core-dump-in-linux-on-a-segmentation-fault
static void cleanup()
{
  sigemptyset(&lSigAction.sa_mask);
}

static void dumpStack(const std::string& filePath)
{
  /* Got this routine from http://www.whitefang.com/unix/faq_toc.html
  ** Section 6.5. Modified to redirect to file to prevent clutter
  */
  std::string gdb("gdb -q -n -ex \"thread apply all bt\" -ex detach -ex quit -p " +
                  std::to_string(getpid()) + " > " + filePath);

  system(gdb.c_str());
}

static std::string mkFilePath(const std::string& filename)
{
  time_t rawTime;      std::time(&rawTime);
  std::tm timeInfo{};  localtime_r(&rawTime, &timeInfo);
  char buffer[128];
  auto sz = snprintf(buffer, sizeof(buffer), "%s", getenv("HOME"));
  std::strftime(&buffer[sz], sizeof(buffer)-sz, "/%Y/%m", &timeInfo);
  struct stat sb;
  if (!stat(buffer, &sb) && S_ISDIR(sb.st_mode)) {
    std::strftime(&buffer[sz], sizeof(buffer)-sz, "/%Y/%m/%d_%H:%M:%S", &timeInfo);
  } else {
    std::strftime( buffer,     sizeof(buffer),   "./%Y_%m_%d_%H:%M:%S", &timeInfo);
  }
  std::string filePath(buffer);
  char hostname[HOST_NAME_MAX];  gethostname(hostname, sizeof(hostname));
  return filePath + "_" + std::string(hostname) + ":" + filename;
}

static void sigHandler(int sig)
{
  logging::debug("sigHandler received signal %d\n", sig);
  if (sig == SIGSEGV || sig == SIGBUS)
  {
    std::string filePath(mkFilePath(lAlias + ".dump"));
    dumpStack(filePath);
    logging::critical("FATAL: %s Fault - logged StackTrace to %s",
                      (sig == SIGSEGV) ? "Segmentation" : "Bus",
                      filePath.c_str());
  }
  if (sig == SIGINT || sig == SIGQUIT || sig == SIGTERM)
  {
    static unsigned callCount{0};
    if (callCount++ == 0)
    {
      logging::warning("Shutting down on %s signal",
                       (sig == SIGINT) ? "INT" : ((sig == SIGQUIT) ? "QUIT" : "TERM"));

      lShutdownAction();
      return;                           // Allow graceful shutdown
    }
    logging::critical("Aborting on 2nd signal");
  }
  exit(EXIT_FAILURE);
}

void initShutdownSignals(const std::string& alias, void (*shutdownAction)())
{
  lAlias          = alias;
  lShutdownAction = shutdownAction;

  atexit(cleanup);

  lSigAction.sa_handler = sigHandler;
  //lSigAction.sa_flags   = SA_RESTART;  // Causes infinite resignal loop
  sigemptyset(&lSigAction.sa_mask);
  lSigAction.sa_flags = 0;
  sigaction(SIGINT, &lSigAction, (struct sigaction *)NULL);

  sigaddset(&lSigAction.sa_mask, SIGSEGV);
  sigaction(SIGSEGV, &lSigAction, (struct sigaction *)NULL);

  sigaddset(&lSigAction.sa_mask, SIGBUS);
  sigaction(SIGBUS, &lSigAction, (struct sigaction *)NULL);

  sigaddset(&lSigAction.sa_mask, SIGINT);
  sigaction(SIGINT,  &lSigAction, (struct sigaction *)NULL);

  sigaddset(&lSigAction.sa_mask, SIGQUIT);
  sigaction(SIGQUIT, &lSigAction, (struct sigaction *)NULL);

  sigaddset(&lSigAction.sa_mask, SIGTERM);
  sigaction(SIGTERM, &lSigAction, (struct sigaction *)NULL);
}

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

json createAsyncWarnMsg(const std::string& alias, const std::string& warnMsg)
{
    json body = json({});
    body["err_info"] = alias + ": " + warnMsg;
    return createMsg("warning", "0", 0, body);
}

bool checkResourceLimits()
{
    const struct Resources
    {
        const char*  name;              // Pretty name for printing
        const int    resource;          // RLIMIT_* value
        const rlim_t required;          // Value application needs for proper execution
        const bool   fatal;             // Fail if unsettable
    } limits[] = { { "MEMLOCK", RLIMIT_MEMLOCK, RLIM_INFINITY, true },
                   { "RTPRIO",  RLIMIT_RTPRIO,  99, false } };
    const auto size = sizeof(limits) / sizeof(*limits);
    struct rlimit rlimits[size];

    // Query current and maximum resource limits
    for (unsigned i = 0; i < size; ++i) {
        if (getrlimit(limits[i].resource, &rlimits[i]) < 0) {
            logging::error("Failed to get %s limit: %M", limits[i].name);
            return true;
        }
    }

    // Check whether maximum limits meet requirements
    bool bad = false;
    for (unsigned i = 0; i < size; ++i) {
        if (rlimits[i].rlim_max != limits[i].required) {
            logging::critical("Inadequate %s limit: got %jd, require %jd",
                              limits[i].name, (uintmax_t)rlimits[i].rlim_max, limits[i].required);
            bad |= limits[i].fatal;
        }
    }
    if (bad)  return true;

    // Raise current limits to maximums, if needed
    for (unsigned i = 0; i < size; ++i) {
        if (rlimits[i].rlim_cur != rlimits[i].rlim_max) {
            rlimits[i].rlim_cur  = rlimits[i].rlim_max;
            if (setrlimit(limits[i].resource, &rlimits[i]) < 0) {
                logging::error("Failed to set %s limit from %jd to %jd: %M",
                               limits[i].name, (uintmax_t)rlimits[i].rlim_cur, (uintmax_t)rlimits[i].rlim_max);
                return limits[i].fatal;
            }
            logging::debug("Raised %s limit from %jd to %jd",
                           limits[i].name, (uintmax_t)rlimits[i].rlim_cur, (uintmax_t)rlimits[i].rlim_max);
        }
    }
    return false;                       // No error
}

static
std::string _getNicIp(const struct ifaddrs* ifaddr,
                      const std::string& ifaceName)
{
    char host[NI_MAXHOST];
    host[0] = '\0';

    for (const struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }
        int family = ifa->ifa_addr->sa_family;
        std::string ifName(ifa->ifa_name);

        if ((family == AF_INET) && (ifName == ifaceName)) {
            int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                                host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
            if (s != 0) {
                logging::error("getnameinfo() failed: %s\n", gai_strerror(s));
            }
            logging::debug("Interface address %s: <%s>\n", ifa->ifa_name, host);
        }
    }
    if (!host[0]) {
        logging::critical("Interface address for NIC '%s' not found", ifaceName.c_str());
        abort();
    }
    return std::string(host);
}

std::string getNicIp(bool forceEnet)
{
    struct ifaddrs* ifaddr;
    getifaddrs(&ifaddr);

    char* interface_name = nullptr;
    char* ethernet_name  = nullptr;
    // find name of first infiniband, otherwise fall back to ethernet
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
            else if (s->sll_hatype == ARPHRD_ETHER &&
                     ifa->ifa_flags & IFF_RUNNING) {
                if (!ethernet_name) ethernet_name  = ifa->ifa_name;
            }
        }
    }
    if ((interface_name == nullptr) || forceEnet) {
        if (ethernet_name == nullptr) {
            logging::critical("No Infiniband or Ethernet interface found");
            abort();
        }
        if (!forceEnet)
            logging::warning("No Infiniband interface found - using Ethernet");
        else
            logging::warning("Using Ethernet instead of Infiniband");
        interface_name = ethernet_name;
    }

    // get address of the first device found above
    std::string host = _getNicIp(ifaddr, std::string(interface_name));

    freeifaddrs(ifaddr);
    return std::string(host);
}

std::string getNicIp(const std::string& ifaceName)
{
    struct ifaddrs* ifaddr;
    getifaddrs(&ifaddr);

    // get address of the interface
    std::string host = _getNicIp(ifaddr, ifaceName);

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
        logging::error("Collection.cc: zmq_msg_recv bad return value %d",rc);
        logging::error("This can happen when a process is sent a signal.  Exiting.");
        return std::string();
    }
    return std::string((char*)frame.data(), frame.size());
}

json ZmqSocket::recvJson()
{
    ZmqMessage frame;
    int rc = zmq_msg_recv(&frame.msg, socket, 0);
    if (rc == -1) {
        logging::error("Collection.cc: zmq_msg_recv bad return value %d",rc);
        logging::error("This can happen when a process is sent a signal.  Exiting.");
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
            logging::error("Collection.cc: zmq_msg_recv bad return value %d",rc);
            logging::error("This can happen when a process is sent a signal.  Exiting.");
            frames.clear();
            break;                      // Let application wind down
        }
        frames.emplace_back(std::move(frame));
        rc = zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size);
        if (rc != 0) {
            logging::error("Collection.cc: zmq_getsockopt bad return value %d",rc);
            logging::error("This can happen when a process is sent a signal.  Exiting.");
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
        logging::error("Error sending zmq message:  %s", msg.c_str());
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
    m_inprocRecv{&m_context, ZMQ_PAIR},
    m_nsubscribe_partition(0)
{
    // Check and raise required ulimit resources to maximum (if needed)
    if (checkResourceLimits()) {
        throw "Resource limit(s) error";
    }

    m_pushSocket.connect({"tcp://" + managerHostname + ":" + std::to_string(zmq_base_port + platform)});

    m_subSocket.connect({"tcp://" + managerHostname + ":" + std::to_string(zmq_base_port + 10 + platform)});
    m_subSocket.setsockopt(ZMQ_SUBSCRIBE, "all", 3);
    std::ostringstream ss;
    ss << std::string{"tcp://" + managerHostname + ":" + std::to_string(zmq_base_port + 10 + platform)};
    logging::debug("%s", ss.str().c_str());
    m_inprocRecv.bind("inproc://drp");

    // register callbacks
    m_handleMap["rollcall"] = std::bind(&CollectionApp::handleRollcall, this, std::placeholders::_1);
    m_handleMap["alloc"] = std::bind(&CollectionApp::handleAlloc, this, std::placeholders::_1);
    m_handleMap["dealloc"] = std::bind(&CollectionApp::handleDealloc, this, std::placeholders::_1);
    m_handleMap["connect"] = std::bind(&CollectionApp::handleConnect, this, std::placeholders::_1);
    m_handleMap["disconnect"] = std::bind(&CollectionApp::handleDisconnect, this, std::placeholders::_1);
    m_handleMap["reset"] = std::bind(&CollectionApp::handleReset, this, std::placeholders::_1);
    m_handleMap["configure"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["unconfigure"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["beginrun"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["endrun"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["beginstep"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["endstep"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["enable"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
    m_handleMap["disable"] = std::bind(&CollectionApp::handlePhase1, this, std::placeholders::_1);
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
        subscribePartition();       // ZMQ_SUBSCRIBE
        json info = connectionInfo(msg);
        json body = {{m_level, info}};
        std::ostringstream ss;
        ss << std::setw(4) << body;
        logging::debug("body handleAlloc  %s", ss.str().c_str());
        json answer = createMsg("alloc", msg["header"]["msg_id"], m_id, body);

        reply(answer);
    }
}

void CollectionApp::handleDealloc(const json &msg)
{
    unsubscribePartition();     // ZMQ_UNSUBSCRIBE
    connectionShutdown();
    json body = json({});
    json answer = createMsg("dealloc", msg["header"]["msg_id"], m_id, body);
    reply(answer);
}

void CollectionApp::reply(const json& msg)
{
    m_pushSocket.send(msg.dump());
}

void CollectionApp::subscribePartition()
{
    // check if already subscribed
    if (m_nsubscribe_partition==0) {
        logging::debug("subscribing to partition");
        m_subSocket.setsockopt(ZMQ_SUBSCRIBE, "partition", 9);
        m_nsubscribe_partition = 1;
    }
}

void CollectionApp::unsubscribePartition()
{
    // check if already subscribed
    if (m_nsubscribe_partition==1) {
        logging::debug("unsubscribing from partition");
        m_subSocket.setsockopt(ZMQ_UNSUBSCRIBE, "partition", 9);
        m_nsubscribe_partition = 0;
    }
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
            else logging::error("zmq_poll error: %u: %m\n", errno);
        }

        // received zeromq message from the collection
        if (items[0].revents & ZMQ_POLLIN) {
            std::vector<ZmqMessage> frames = m_subSocket.recvMultipart();
            if (frames.size() < 2)  break;  // Revisit: Terminate condition
            char* begin = (char*)frames[1].data();
            char* end = begin + frames[1].size();
            json msg = json::parse(begin, end);
            std::string topic((char*)frames[0].data(), frames[0].size());
            logging::debug("topic:  %s", topic.c_str());

            std::string key = msg["header"]["key"];
            time_t rawTime;  time (&rawTime);   // Current time
            logging::info("received key = %s @ %s", key.c_str(), ctime(&rawTime));
            std::ostringstream ss;
            ss << std::setw(4) << msg;
            logging::debug("%s", ss.str().c_str());
            if (m_handleMap.find(key) == m_handleMap.end()) {
                logging::error("unknown key = %s", key.c_str());
            }
            else {
                m_handleMap[key](msg);
            }
        }

        // received inproc message
        // ex 1: {"body":{"pulseId":2463859721849},"key":"pulseId"}
        if (items[1].revents & ZMQ_POLLIN) {
            json msg = m_inprocRecv.recvJson();
            //logging::debug("inproc json received  %s", msg.dump().c_str());
            std::string key = msg["key"];
            //logging::debug("inproc '%s' message received", key.c_str());
            if (key == "pulseId") {
                // forward message to control system with pulseId as the message id
                uint64_t pid = msg["body"]["pulseId"];
                //logging::debug("inproc pulseId received  %014lx", pid);
                json body;
                json answer = createMsg("timingTransition", std::to_string(pid), getId(), body);
                reply(answer);
            } else if ((key == "fileReport") || (key == "error") || (key == "chunkRequest")) {
                json answer = createMsg(key.c_str(), "0", 0, msg["body"]);
                logging::debug("reply %s: %s", key.c_str(), answer.dump().c_str());
                reply(answer);
            }
        }
    }
}
