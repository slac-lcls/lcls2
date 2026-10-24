/**
 * @file
 * @brief ZeroMQ helpers (context, message, socket), NIC address lookup, and CollectionApp, the base of processes that take part in the DAQ collection protocol.
 */
#pragma once

#include <string>
#include <vector>
#include <zmq.h>
#include <nlohmann/json.hpp>

/** Owns a ZeroMQ context. */
class ZmqContext
{
public:
    /** Create the context with zmq_ctx_new(). */
    ZmqContext() {m_context = zmq_ctx_new();}
    /** Return the raw ZeroMQ context pointer. */
    void* operator() () {return m_context;};
    /** Destroy the context with zmq_ctx_destroy(). */
    ~ZmqContext() {zmq_ctx_destroy(m_context);}
private:
    void* m_context;
};

/** Owns one zmq_msg_t frame; movable but not copyable. */
class ZmqMessage
{
public:
    /** Initialize an empty frame with zmq_msg_init(). */
    ZmqMessage() {zmq_msg_init(&msg);};
    /** Close the frame with zmq_msg_close(). */
    ~ZmqMessage() {zmq_msg_close(&msg);}
    /** Move constructor: take over the frame of m and reinitialize m as empty. */
    ZmqMessage(ZmqMessage&& m) noexcept : msg(m.msg) {zmq_msg_init(&m.msg);}
    /** Return a pointer to the frame contents. */
    void* data() {return zmq_msg_data(&msg);}
    /** Return the frame size in bytes. */
    size_t size() {return zmq_msg_size(&msg);}
    /** Deleted: frames are not copyable. */
    ZmqMessage(const ZmqMessage&) = delete;
    /** Deleted: frames are not copy-assignable. */
    void operator = (const ZmqMessage&) = delete;
private:
    zmq_msg_t msg;
    friend class ZmqSocket;
};

/** Wraps one ZeroMQ socket created from a ZmqContext. */
class ZmqSocket
{
public:
    /** Create a socket of the given ZeroMQ type in context. */
    ZmqSocket(ZmqContext* context, int type);
    /** Close the socket. */
    ~ZmqSocket() {zmq_close(socket);}
    /** Connect the socket to the endpoint host; throws std::runtime_error on failure. */
    void connect(const std::string& host);
    /** Bind the socket to the endpoint host; throws std::runtime_error on failure. */
    void bind(const std::string& host);
    /** Set a socket option; throws std::runtime_error on failure. */
    void setsockopt(int option, const void* optval, size_t optvallen);
    /** Receive one frame and return it as a string; logs an error and returns an empty string if the receive fails. */
    std::string recv();
    /** Receive one frame and parse it as JSON; logs an error and returns an empty JSON object if the receive fails. */
    nlohmann::json recvJson();
    /** Receive all frames of one multi-part message; returns an empty vector (after logging) if a receive or ZMQ_RCVMORE query fails. */
    std::vector<ZmqMessage> recvMultipart();
    /** Send msg as one frame; a failure is only logged. */
    void send(const std::string& msg);
    /** Poll this socket for events for up to timeout (zmq_poll() units) and return the zmq_poll() result. */
    int poll(short events, long timeout);
    void* socket;  ///< Raw ZeroMQ socket handle.
private:
    ZmqContext* m_context;
};

/** Return the IPv4 address of the interface named forceIface; aborts (after logging) if it has none. */
std::string getNicIp(const std::string& forceIface);
/** Return the IPv4 address of the first running Ethernet interface if forceEnet is true, otherwise of the first running Infiniband interface. Logs a critical message and aborts if there is no such interface; it does not fall back to the other type. */
std::string getNicIp(bool forceEnet);

/** Base of processes that join a DAQ collection: checks resource limits, connects a PUSH socket to tcp://manager:(zmq_base_port + platform) and a SUB socket (topic all) to port zmq_base_port + 10 + platform, binds inproc://drp, and dispatches incoming messages by key to the handle* methods. */
class CollectionApp
{
public:
    /** Check the resource limits (throwing a C string if they are inadequate), connect the PUSH and SUB sockets to managerHostname for platform, bind inproc://drp, and register handlers for rollcall, alloc, dealloc, connect, disconnect, reset and the phase-1 transitions (configure, unconfigure, beginrun, endrun, beginstep, endstep, enable, disable). */
    CollectionApp(const std::string& managerHostname, int platform, const std::string& level, const std::string& alias, const std::string& device=std::string());
    /** Does nothing; empty body. */
    virtual ~CollectionApp() {};
    /** Message loop: dispatch collection messages from the SUB socket to their handlers by header key, and forward pulseId, fileReport, error and chunkRequest messages from inproc://drp to the manager. Returns when zmq_poll() is interrupted or a malformed (fewer than 2 frames) message arrives. */
    void run();
    /** Base TCP port of the collection sockets. */
    enum {zmq_base_port = 29980 /**< 29980; the PUSH port is this plus the platform, the SUB port this plus 10 plus the platform. */ };
protected:
    void handleRollcall(const nlohmann::json& msg);
    void handleAlloc(const nlohmann::json& msg);
    virtual void handleDealloc(const nlohmann::json& msg);
    virtual nlohmann::json connectionInfo(const nlohmann::json& msg) = 0;
    virtual void connectionShutdown() {};
    virtual void handleConnect(const nlohmann::json& msg) = 0;
    virtual void handleDisconnect(const nlohmann::json& msg) {};
    virtual void handlePhase1(const nlohmann::json& msg) {};
    virtual void handleReset(const nlohmann::json& msg) = 0;
    void reply(const nlohmann::json& msg);
    size_t getId() const {return m_id;}
    const std::string& getLevel() const {return m_level;}
    const std::string& getAlias() const {return m_alias;}
    ZmqContext& context() {return m_context;}
    void subscribePartition();
    void unsubscribePartition();
private:
    std::string m_level;
    std::string m_alias;
    std::string m_device;
    ZmqContext m_context;
    ZmqSocket m_pushSocket;
    ZmqSocket m_subSocket;
    ZmqSocket m_inprocRecv;
    unsigned m_nsubscribe_partition;
    size_t m_id;
    std::unordered_map<std::string, std::function<void(nlohmann::json&)> > m_handleMap;
};

/** Return a JSON message with header key, msg_id and sender_id and the given body. */
nlohmann::json createMsg(const std::string& key, const std::string& msg_id, size_t sender_id, nlohmann::json& body);
/** Return an error message (key error, msg_id 0, sender 0) whose body err_info is alias: errMsg. */
nlohmann::json createAsyncErrMsg(const std::string& alias, const std::string& errMsg);
/** Return a warning message (key warning, msg_id 0, sender 0) whose body err_info is alias: warnMsg. */
nlohmann::json createAsyncWarnMsg(const std::string& alias, const std::string& warnMsg);

/** Check that the hard MEMLOCK limit is unlimited and the hard RTPRIO limit is 99, and raise the soft limits to the hard ones. Returns true on a fatal problem (MEMLOCK inadequate or not settable, or a getrlimit() failure), else false. */
bool checkResourceLimits();

/// Report an exception that escapes any thread, rather than letting std::terminate
/// print only its mangled type.  Call once per process, as early as possible.
void initExceptionReporting(const std::string& alias);

/** Install a handler for SIGINT, SIGQUIT and SIGTERM that calls shutdownAction on the first signal and exits on the second, and for SIGSEGV and SIGBUS that writes a gdb stack trace to a file named after alias and exits. */
void initShutdownSignals(const std::string& alias, void (*shutdownAction)());
