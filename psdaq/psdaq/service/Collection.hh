#pragma once

#include <string>
#include <zmq.h>
#include "json.hpp"

class ZmqContext
{
public:
    ZmqContext() {m_context = zmq_ctx_new();}
    void* operator() () {return m_context;};
    ~ZmqContext() {zmq_ctx_destroy(m_context);}
private:
    void* m_context;
};

class ZmqMessage
{
public:
    ZmqMessage() {zmq_msg_init(&msg);};
    ~ZmqMessage() {zmq_msg_close(&msg);}
    ZmqMessage(ZmqMessage&& m) noexcept : msg(m.msg) {zmq_msg_init(&m.msg);}
    void* data() {return zmq_msg_data(&msg);}
    size_t size() {return zmq_msg_size(&msg);}
    ZmqMessage(const ZmqMessage&) = delete;
    void operator = (const ZmqMessage&) = delete;
private:
    zmq_msg_t msg;
    friend class ZmqSocket;
};

class ZmqSocket
{
public:
    ZmqSocket(ZmqContext* context, int type);
    ~ZmqSocket() {zmq_close(socket);}
    void connect(const std::string& host);
    void bind(const std::string& host);
    void setsockopt(int option, const void* optval, size_t optvallen);
    std::string recv();
    nlohmann::json recvJson();
    std::vector<ZmqMessage> recvMultipart();
    void send(const std::string& msg);
    int poll(short events, long timeout);
    void* socket;
private:
    ZmqContext* m_context;
};

std::string getNicIp(const std::string& forceIface);
std::string getNicIp(bool forceEnet);

class CollectionApp
{
public:
    CollectionApp(const std::string& managerHostname, int platform, const std::string& level, const std::string& alias);
    virtual ~CollectionApp() {};
    void run();
protected:
    void handleRollcall(const nlohmann::json& msg);
    void handleAlloc(const nlohmann::json& msg);
    virtual nlohmann::json connectionInfo() = 0;
    virtual void handleConnect(const nlohmann::json& msg) = 0;
    virtual void handleDisconnect(const nlohmann::json& msg) {};
    virtual void handlePhase1(const nlohmann::json& msg) {};
    virtual void handleReset(const nlohmann::json& msg) = 0;
    void reply(const nlohmann::json& msg);
    size_t getId() const {return m_id;}
    const std::string& getLevel() const {return m_level;}
    const std::string& getAlias() const {return m_alias;}
    ZmqContext& context() {return m_context;}
private:
    std::string m_level;
    std::string m_alias;
    ZmqContext m_context;
    ZmqSocket m_pushSocket;
    ZmqSocket m_subSocket;
    ZmqSocket m_inprocRecv;
    size_t m_id;
    std::unordered_map<std::string, std::function<void(nlohmann::json&)> > m_handleMap;
};

nlohmann::json createMsg(const std::string& key, const std::string& msg_id, size_t sender_id, nlohmann::json& body);
nlohmann::json createAsyncErrMsg(const std::string& alias, const std::string& errMsg);
nlohmann::json createAsyncWarnMsg(const std::string& alias, const std::string& warnMsg);
