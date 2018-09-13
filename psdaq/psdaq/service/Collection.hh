#ifndef COLLECTION_H
#define COLLECTION_H

#include <string>
#include <functional>
#include <unordered_map>
#include <zmq.h>
#include "json.hpp"
using json = nlohmann::json;

class ZmqMessage
{
public:
    ZmqMessage() {zmq_msg_init(&msg);};
    ~ZmqMessage() {zmq_msg_close(&msg);}
    ZmqMessage(ZmqMessage&& m) {msg = m.msg; zmq_msg_init(&m.msg);}
    void* data() {return zmq_msg_data(&msg);}
    size_t size() {return zmq_msg_size(&msg);}
    zmq_msg_t msg;
    ZmqMessage(const ZmqMessage&) = delete;
    void operator=(const ZmqMessage&) = delete;

};

class Collection
{
public:
    Collection(const std::string& manager_hostname, int platform, const std::string& level);
    ~Collection();
    void handle_plat(json& msg);
    void handle_alloc(json& msg);
    void handle_connect(json& msg);
    void handle_reset(json& msg);
    void connect();
    size_t id() {return m_id;}
    json cmstate;
private:
    std::string m_state;
    void* m_context;
    void* m_push;
    void* m_sub;
    std::string m_level;
    size_t m_id;
    std::unordered_map<std::string, std::function<void(json&)> > m_handle_request;
};

json create_msg(const std::string& key, const std::string& msg_id, size_t sender_id, json& body);
std::vector<ZmqMessage> recv_multipart(void* socket);
json recv_json(void* socket);
std::string get_infiniband_address();

#endif // COLLECTION_H
