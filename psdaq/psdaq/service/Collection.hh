#ifndef COLLECTION_H
#define COLLECTION_H

#include <string>
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
    Collection();
    ~Collection();
    json partition_info(const std::string& level);
    json connection_info(json& msg);
    std::vector<ZmqMessage> wait_for_msg(const std::string& key);
private:
    void* m_context;
    void* m_dealer;
    void* m_sub;
};

std::vector<ZmqMessage> recv_multipart(void* socket);
std::string get_infiniband_address();

#endif // COLLECTION_H
