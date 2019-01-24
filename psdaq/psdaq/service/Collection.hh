#pragma once

#include <string>
#include <zmq.h>
#include "json.hpp"
using json = nlohmann::json;

class ZmqContext
{
public:
    ZmqContext() {context = zmq_ctx_new();}
    ~ZmqContext() {zmq_ctx_destroy(context);}
private:
    void* context;
    friend class ZmqSocket;
};

class ZmqMessage
{
public:
    ZmqMessage() {zmq_msg_init(&msg);};
    ~ZmqMessage() {zmq_msg_close(&msg);}
    ZmqMessage(ZmqMessage&& m) {msg = m.msg; zmq_msg_init(&m.msg);}
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
    ZmqSocket(std::shared_ptr<ZmqContext> context, int type);
    ~ZmqSocket() {zmq_close(socket);}
    void connect(const std::string& host);
    void setsockopt(int option, const void* optval, size_t optvallen);
    json recvJson();
    std::vector<ZmqMessage> recvMultipart();
    void send(const std::string& msg);
private:
    void* socket;
    std::shared_ptr<ZmqContext> context;
};

enum class State {reset, plat, alloc, connect};

class CollectionApp
{
public:
    CollectionApp(const std::string& managerHostname, int platform, const std::string& level);
    void run();
protected:
    virtual void handlePlat(const json& msg);
    virtual void handleAlloc(const json& msg);
    virtual void handleConnect(const json& msg) = 0;
    virtual void handleReset(const json& msg) = 0;
    void reply(const json& msg);
    size_t getId() const {return m_id;}
    void setState(State state) {m_state = state;}
private:
    State m_state;
    std::string m_level;
    std::unique_ptr<ZmqSocket> m_pushSocket;
    std::unique_ptr<ZmqSocket> m_subSocket;
    size_t m_id;
    std::unordered_map<std::string, std::function<void(json&)> > m_handleMap;
};

json createMsg(const std::string& key, const std::string& msg_id, size_t sender_id, json& body);
