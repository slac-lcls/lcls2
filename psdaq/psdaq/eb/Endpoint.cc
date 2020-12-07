#include "Endpoint.hh"

#include <rdma/fi_cm.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define FIVER FI_VERSION(1, 6)

#define ERR_MSG_LEN 256

#define ANY_ADDR "0.0.0.0"

#define CHECK_ERR(function, msg)  \
  _errno = function;              \
  if (_errno != FI_SUCCESS) {     \
    set_error(msg);               \
    return false;                 \
  }

#define CHECK_OBJ(function, obj)                \
  if (!function) {                              \
    _errno = (obj)->error_num();                \
    set_custom_error((obj)->error());           \
    return false;                               \
  }

#define CHECK_ERR_EX(function, msg, exclude)            \
  _errno = function;                                    \
  if ((_errno != FI_SUCCESS) && (_errno != -exclude)) { \
    set_error(msg);                                     \
    return false;                                       \
  }

#define CHECK(function)     \
  if (!function)            \
    return false;

#define CHECK_MR(buf, len, mr, cmd)                                                                                                                                 \
  if (mr) {                                                                                                                                                         \
    if (!mr->contains(buf, len)) {                                                                                                                                  \
      set_custom_error("%s: requested buffer starting at %p with len %zu is not within given memory region %p, len %zd", cmd, buf, len, mr->start(), mr->length()); \
      ssize_t rret = -FI_EINVAL;                                                                                                                                    \
      _errno = rret;                                                                                                                                                \
      return rret;                                                                                                                                                  \
    }                                                                                                                                                               \
  } else {                                                                                                                                                          \
    mr = _fabric->lookup_memory(buf, len);                                                                                                                          \
    if (!mr) {                                                                                                                                                      \
      set_custom_error("%s: requested buffer starting at %p with len %zu is not within a registered memory region", cmd, buf, len);                                 \
      ssize_t rret = -FI_EINVAL;                                                                                                                                    \
      _errno = rret;                                                                                                                                                \
     return rret;                                                                                                                                                   \
    }                                                                                                                                                               \
  }

#define CHECK_MR_IOVEC(iov, cmd)                                                                        \
  if (!iov->check_mr()) {                                                                               \
    if (!_fabric->lookup_memory_iovec(iov)) {                                                           \
      set_custom_error("%s: requested buffer in iovec is not within a registered memory region", cmd);  \
      ssize_t rret = -FI_EINVAL;                                                                        \
      _errno = rret;                                                                                    \
      return rret;                                                                                      \
    }                                                                                                   \
  }


using namespace Pds::Fabrics;

LocalAddress::LocalAddress() :
  _buf(0),
  _len(0),
  _mr(0)
{}

LocalAddress::LocalAddress(void* buf, size_t len, MemoryRegion* mr) :
  _buf(buf),
  _len(len),
  _mr(mr)
{}

void* LocalAddress::buf() const { return _buf; }

size_t LocalAddress::len() const { return _len; }

MemoryRegion* LocalAddress::mr() const { return _mr; }

const struct iovec* LocalAddress::iovec() const
{
  return reinterpret_cast<const struct iovec*>(this);
}

LocalIOVec::LocalIOVec(size_t count) :
  _mr_set(true),
  _count(0),
  _max(count),
  _iovs(new struct iovec[count]),
  _mr_desc(new void*[count])
{}

LocalIOVec::LocalIOVec(LocalAddress* local_addrs, size_t count) :
  _mr_set(true),
  _count(count),
  _max(count),
  _iovs(new struct iovec[count]),
  _mr_desc(new void*[count])
{
  if (local_addrs) {
    for (unsigned i=0; i<_count; i++) {
      _iovs[i] = *local_addrs[i].iovec();
      if (local_addrs[i].mr())
        _mr_desc[i] = local_addrs[i].mr()->desc();
      else
        _mr_set = false;
    }
  } else {
    _mr_set = false;
  }
}

LocalIOVec::LocalIOVec(const std::vector<LocalAddress*>& local_addrs) :
  _mr_set(true),
  _count(local_addrs.size()),
  _max(local_addrs.size()),
  _iovs(new struct iovec[local_addrs.size()]),
  _mr_desc(new void*[local_addrs.size()])
{
  for (unsigned i=0; i<_count; i++) {
    _iovs[i] = *local_addrs[i]->iovec();
    if (local_addrs[i]->mr())
      _mr_desc[i] = local_addrs[i]->mr()->desc();
    else
      _mr_set = false;
  }
}

LocalIOVec::~LocalIOVec()
{
  if (_iovs) {
    delete[] _iovs;
  }
  if (_mr_desc) {
    delete[] _mr_desc;
  }
}

void LocalIOVec::reset()
{
  _mr_set = true;
  _count  = 0;
}

void LocalIOVec::check_size(size_t count)
{
  if ((_count + count) > _max) {
    struct iovec* new_iovs = new struct iovec[_count + count];
    void** new_mr_desc = new void*[_count + count];
    for (unsigned i=0; i<_max; i++) {
      new_iovs[i] = _iovs[i];
      new_mr_desc[i] = _mr_desc[i];
    }

    delete[] _iovs;
    delete[] _mr_desc;
    _iovs = new_iovs;
    _mr_desc = new_mr_desc;
    _max = (_count + count);
  }
}

bool LocalIOVec::check_mr()
{
  return _mr_set;
}

void LocalIOVec::verify()
{
  _mr_set = true;
  for (unsigned i=0; i<_count; i++) {
    if (!_mr_desc[i]) {
      _mr_set=false;
      break;
    }
  }
}

size_t LocalIOVec::count() const { return _count; }

const struct iovec* LocalIOVec::iovecs() const { return _iovs; }

void** LocalIOVec::desc() const { return _mr_desc; }

void* LocalIOVec::allocate()
{
  check_size(1);

  return &_iovs[_count++];
}

bool LocalIOVec::add_iovec(LocalAddress* local_addr, size_t count)
{
  check_size(count);

  if (local_addr) {
    for (unsigned i=0; i<count; i++) {
      _iovs[_count] = *local_addr[i].iovec();
      if (local_addr[i].mr()) {
        _mr_desc[_count] = local_addr[i].mr()->desc();
      }
      _count++;
    }

    verify();

    return true;
  } else {
    return false;
  }
}

bool LocalIOVec::add_iovec(std::vector<LocalAddress*>& local_addr)
{
  check_size(local_addr.size());

  for (unsigned i=0; i<local_addr.size(); i++) {
    _iovs[_count] = *local_addr[i]->iovec();
    if (local_addr[i]->mr()) {
      _mr_desc[_count] = local_addr[i]->mr()->desc();
    }
    _count++;
  }

  verify();

  return true;
}

bool LocalIOVec::add_iovec(void* buf, size_t len, MemoryRegion* mr)
{
  check_size(1);

  _iovs[_count].iov_base = buf;
  _iovs[_count].iov_len = len;
  if (mr) {
    _mr_desc[_count] = mr->desc();
  }
  _count++;

  verify();

  return true;
}

bool LocalIOVec::set_count(size_t count)
{
  if (count >= _max) {
    return false;
  }

  _count = count;
  verify();

  return true;
}

bool LocalIOVec::set_iovec(unsigned index, LocalAddress* local_addr)
{
  if (index >= _max) {
    return false;
  }

  _iovs[index] = *local_addr->iovec();
  if (local_addr->mr()) {
    _mr_desc[index] = local_addr->mr()->desc();
    verify();
  } else {
    _mr_set = false;
  }

  return true;
}

bool LocalIOVec::set_iovec(unsigned index, void* buf, size_t len, MemoryRegion* mr)
{
  if (index >= _max) {
    return false;
  }

  _iovs[index].iov_base = buf;
  _iovs[index].iov_len = len;
  if (mr) {
    _mr_desc[index] = mr->desc();
    verify();
  } else {
    _mr_set = false;
  }

  return true;
}

bool LocalIOVec::set_iovec_mr(unsigned index, MemoryRegion* mr)
{
  if (index >= _max || !mr) {
    return false;
  }

  _mr_desc[index] = mr->desc();
  verify();

  return true;
}

RemoteAddress::RemoteAddress() :
  addr(0),
  extent(0),
  rkey(0)
{}

RemoteAddress::RemoteAddress(uint64_t rkey, uint64_t addr, size_t extent) :
  addr(addr),
  extent(extent),
  rkey(rkey)
{}

RemoteIOVec::RemoteIOVec(size_t count) :
  _count(0),
  _max(count),
  _rma_iovs(new struct fi_rma_iov[count])
{}

RemoteIOVec::RemoteIOVec(RemoteAddress* remote_addrs, size_t count) :
  _count(count),
  _max(count),
  _rma_iovs(new struct fi_rma_iov[count])
{
  if (remote_addrs) {
    for (unsigned i=0; i<_count; i++) {
      _rma_iovs[i] = *remote_addrs[i].rma_iov();
    }
  }
}

RemoteIOVec::RemoteIOVec(const std::vector<RemoteAddress*>& remote_addrs) :
  _count(remote_addrs.size()),
  _max(remote_addrs.size()),
  _rma_iovs(new struct fi_rma_iov[remote_addrs.size()])
{
  for (unsigned i=0; i<_count; i++) {
    _rma_iovs[i] = *remote_addrs[i]->rma_iov();
  }
}

RemoteIOVec::~RemoteIOVec()
{
  if (_rma_iovs) {
    delete[] _rma_iovs;
  }
}

void RemoteIOVec::check_size(size_t count)
{
  if ((_count + count) > _max) {
    _max = (_count + count);

    struct fi_rma_iov* new_rma_iovs = new struct fi_rma_iov[_max];
    for (unsigned i=0; i<_count; i++) {
      new_rma_iovs[i] = _rma_iovs[i];
    }

    delete[] _rma_iovs;
    _rma_iovs = new_rma_iovs;
  }
}

size_t RemoteIOVec::count() const { return _count; }

const struct fi_rma_iov* RemoteIOVec::iovecs() const { return _rma_iovs; }

bool RemoteIOVec::add_iovec(RemoteAddress* remote_addr, size_t count)
{
  check_size(count);

  if (remote_addr) {
    for (unsigned i=0; i<count; i++) {
      _rma_iovs[_count++] = *remote_addr[i].rma_iov();
    }
    return true;
  } else {
    return false;
  }
}

bool RemoteIOVec::add_iovec(std::vector<RemoteAddress*>& remote_addr)
{
  check_size(remote_addr.size());

  for (unsigned i=0; i<remote_addr.size(); i++) {
    _rma_iovs[_count++] = *remote_addr[i]->rma_iov();
  }

  return true;
}

bool RemoteIOVec::add_iovec(unsigned index, uint64_t rkey, uint64_t addr, size_t extent)
{
  check_size(1);

  _rma_iovs[_count].addr  = addr;
  _rma_iovs[_count].len   = extent;
  _rma_iovs[_count].key   = rkey;

  _count++;

  return true;
}

bool RemoteIOVec::set_iovec(unsigned index, RemoteAddress* remote_addr)
{
  if (index >= _count) {
    return false;
  }

  if (remote_addr) {
    _rma_iovs[index] = *remote_addr->rma_iov();
    return true;
  } else {
    return false;
  }
}

bool RemoteIOVec::set_iovec(unsigned index, uint64_t rkey, uint64_t addr, size_t extent)
{
  if (index >= _count) {
    return false;
  }

  _rma_iovs[index].addr  = addr;
  _rma_iovs[index].len   = extent;
  _rma_iovs[index].key   = rkey;

  return true;
}

const struct fi_rma_iov* RemoteAddress::rma_iov() const
{
  return reinterpret_cast<const struct fi_rma_iov*>(this);
}

RmaMessage::RmaMessage() :
  _loc_iov(0),
  _rem_iov(0),
  _msg(new struct fi_msg_rma)
{}

RmaMessage::RmaMessage(LocalIOVec* loc_iov, RemoteIOVec* rem_iov, void* context, uint64_t data) :
  _loc_iov(loc_iov),
  _rem_iov(rem_iov),
  _msg(new struct fi_msg_rma)
{
  if (_loc_iov) {
    _msg->msg_iov = loc_iov->iovecs();
    _msg->desc = loc_iov->desc();
    _msg->iov_count = loc_iov->count();
  }
  if (_rem_iov) {
    _msg->rma_iov = rem_iov->iovecs();
    _msg->rma_iov_count = rem_iov->count();
  }
  _msg->context = context;
  _msg->data = data;
}

RmaMessage::~RmaMessage()
{
  if (_msg) {
    delete _msg;
  }
}

LocalIOVec* RmaMessage::loc_iov() const { return _loc_iov; }

RemoteIOVec* RmaMessage::rem_iov() const { return _rem_iov; }

void RmaMessage::loc_iov(LocalIOVec* loc_iov)
{
  if (loc_iov) {
    _loc_iov = loc_iov;
    _msg->msg_iov = loc_iov->iovecs();
    _msg->desc = loc_iov->desc();
    _msg->iov_count = loc_iov->count();
  }
}

void RmaMessage::rem_iov(RemoteIOVec* rem_iov)
{
  if (rem_iov) {
    _rem_iov = rem_iov;
    _msg->rma_iov = rem_iov->iovecs();
    _msg->rma_iov_count = rem_iov->count();
  }
}

const struct iovec* RmaMessage::msg_iov() const { return _msg->msg_iov; }

void** RmaMessage::desc() const { return _msg->desc; }

size_t RmaMessage::iov_count() const { return _msg->iov_count; }

const struct fi_rma_iov* RmaMessage::rma_iov() const { return _msg->rma_iov; }

size_t RmaMessage::rma_iov_count() const { return _msg->rma_iov_count; }

void* RmaMessage::context() const { return _msg->context; }

void RmaMessage::context(void* context) { _msg->context = context; }

uint64_t RmaMessage::data() const { return _msg->data; }

void RmaMessage::data(uint64_t data) { _msg->data = data; }

const struct fi_msg_rma* RmaMessage::msg() const { return _msg; }

Message::Message() :
  _iov(0),
  _msg(new struct fi_msg)
{}

Message::Message(LocalIOVec* iov, fi_addr_t addr, void* context, uint64_t data) :
  _iov(iov),
  _msg(new struct fi_msg)
{
  if (_iov) {
    _msg->msg_iov = iov->iovecs();
    _msg->desc = iov->desc();
    _msg->iov_count = iov->count();
  }
  _msg->addr = addr;
  _msg->context = context;
  _msg->data = data;
}

Message::~Message()
{
  if (_msg) {
    delete _msg;
  }
}

LocalIOVec* Message::iov() const { return _iov; }

void Message::iov(LocalIOVec* iov)
{
  if (iov) {
    _iov = iov;
    _msg->msg_iov = iov->iovecs();
    _msg->desc = iov->desc();
    _msg->iov_count = iov->count();
  }
}

const struct iovec* Message::msg_iov() const { return _msg->msg_iov; }

void** Message::desc() const { return _msg->desc; }

size_t Message::iov_count() const { return _msg->iov_count; }

fi_addr_t Message::addr() const { return _msg->addr; }

void Message::addr(fi_addr_t addr) { _msg->addr = addr; }

void* Message::context() const { return _msg->context; }

void Message::context(void* context) { _msg->context = context; }

uint64_t Message::data() const { return _msg->data; }

void Message::data(uint64_t data) { _msg->data = data; }

const struct fi_msg* Message::msg() const { return _msg; }

MemoryRegion::MemoryRegion(struct fid_mr* mr, void* start, size_t len) :
  _mr(mr),
  _start(start),
  _len(len)
{}

MemoryRegion::~MemoryRegion()
{
  if (_mr) fi_close(&_mr->fid);
}

uint64_t MemoryRegion::rkey() const { return fi_mr_key(_mr); }

void* MemoryRegion::desc() const { return fi_mr_desc(_mr); }

struct fid_mr* MemoryRegion::fid() const { return _mr; }

void* MemoryRegion::start() const { return _start; }

size_t MemoryRegion::length() const { return _len; }

bool MemoryRegion::contains(const void* start, size_t len) const
{
  return (start >= _start) && ( ((char*) start + len) <= ((char*) _start + _len) );
}

ErrorHandler::ErrorHandler() :
  _errno(FI_SUCCESS),
  _error(new char[ERR_MSG_LEN])
{
  set_error("None");
}

ErrorHandler::~ErrorHandler()
{
  if (_error) delete[] _error;
}

int ErrorHandler::error_num() const { return _errno; };

const char* ErrorHandler::error() const { return _error; }

void ErrorHandler::clear_error()
{
  set_error("None");
  _errno = FI_SUCCESS;
}

void ErrorHandler::set_error(const char* error_desc)
{
  snprintf(_error, ERR_MSG_LEN, "%s: %s(%d)", error_desc, fi_strerror(-_errno), _errno);
}

void ErrorHandler::set_custom_error(const char* fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);
  vsnprintf(_error, ERR_MSG_LEN, fmt, argptr);
  va_end(argptr);
}


Info::Info() :
  _owner(true)
{
  _ready = initialize();
}

Info::Info(const std::map<std::string, std::string>& kwargs) :
  _owner(true),
  _ready(false)
{
  if (initialize()) {
    if (kwargs.find("ep_fabric") != kwargs.end()) {
      if (!hints->fabric_attr) {
        hints->fabric_attr = new struct fi_fabric_attr;
      }
      auto& kwa = const_cast<std::map<std::string, std::string>&>(kwargs); // Yuck!
      hints->fabric_attr->name = strdup(kwa["ep_fabric"].c_str());
    }
    if (kwargs.find("ep_domain") != kwargs.end()) {
      if (!hints->domain_attr) {
        hints->domain_attr = new struct fi_domain_attr;
      }
      auto& kwa = const_cast<std::map<std::string, std::string>&>(kwargs); // Yuck!
      hints->domain_attr->name = strdup(kwa["ep_domain"].c_str());
    }
    if (kwargs.find("ep_provider") != kwargs.end()) {
      if (!hints->fabric_attr) {
        hints->fabric_attr = new struct fi_fabric_attr;
      }
      auto& kwa = const_cast<std::map<std::string, std::string>&>(kwargs); // Yuck!
      hints->fabric_attr->prov_name = strdup(kwa["ep_provider"].c_str());
    }
  }
  _ready = true;
}

Info::~Info()
{
  shutdown();
}

void Info::take_hints() { _owner = false; }

bool Info::ready() const { return _ready; }

bool Info::initialize()
{
  hints = fi_allocinfo();
  if (!hints) {
    _errno = -FI_ENOMEM;
    set_error("fi_allocinfo");
    return false;
  }

  hints->addr_format = FI_SOCKADDR_IN;
  hints->ep_attr->type = FI_EP_MSG;
  hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  hints->caps = FI_MSG | FI_RMA;
  hints->mode = FI_LOCAL_MR | FI_RX_CQ_DATA;
  hints->domain_attr->cq_data_size = 4;  /* required minimum */

  return true;
}

void Info::shutdown()
{
  if (hints) {
    if (_owner)  fi_freeinfo(hints);
    hints = nullptr;
  }
  _ready = false;
}


Fabric::Fabric(const char* node, const char* service, uint64_t flags, Info* hints) :
  _up(false),
  _hints_owner(false),
  _info(0),
  _fabric(0),
  _domain(0)
{
  if (!hints) {
    Info info;
    if (!info.ready())  return;
    info.take_hints();
    _hints = info.hints;
    _hints_owner = true;
  } else {
    _hints = hints->hints;
  }

  _up = initialize(node, service, flags);
}

Fabric::~Fabric()
{
  shutdown();
}

MemoryRegion* Fabric::register_memory(void* start, size_t len)
{
  struct fid_mr* mr;
  if (_up) {
    _errno = fi_mr_reg(_domain, start, len,
                       FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV,
                       0, 0, 0, &mr, NULL);
    if (_errno == FI_SUCCESS) {
      MemoryRegion* mem = new MemoryRegion(mr, start, len);
      _mem_regions.push_back(mem);
      return mem;
    } else {
      set_error("fi_mr_reg");
    }
  }

  return NULL;
}

MemoryRegion* Fabric::register_memory(LocalAddress* laddr)
{
  struct fid_mr* mr;
  if (_up) {
    _errno = fi_mr_reg(_domain, laddr->buf(), laddr->len(),
                       FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV,
                       0, 0, 0, &mr, NULL);
    if (_errno == FI_SUCCESS) {
      MemoryRegion* mem = new MemoryRegion(mr, laddr->buf(), laddr->len());
      laddr->_mr = mem;
      _mem_regions.push_back(mem);
      return mem;
    } else {
      set_error("fi_mr_reg");
    }
  }

  return NULL;
}

bool Fabric::deregister_memory(MemoryRegion* mr)
{
  for (auto it = _mem_regions.begin(); it != _mem_regions.end(); ++it) {
    if (*it == mr) {
      _mem_regions.erase(it);
      delete mr;
      return true;
    }
  }
  return false;
}

MemoryRegion* Fabric::lookup_memory(const void* start, size_t len) const
{
  for (unsigned i=0; i<_mem_regions.size(); i++) {
    if (_mem_regions[i] && _mem_regions[i]->contains(start, len))
      return _mem_regions[i];
  }

  return NULL;
}

MemoryRegion* Fabric::lookup_memory(LocalAddress* laddr) const
{
  if (!laddr)
    return NULL;

  return lookup_memory(laddr->buf(), laddr->len());
}

bool Fabric::lookup_memory_iovec(LocalIOVec* iov) const
{
  void** descs = iov->desc();
  const struct iovec* iovecs = iov->iovecs();

  for (unsigned i=0; i<iov->count(); i++) {
    if(!descs[i]) {
      for (unsigned j=0; j<_mem_regions.size(); j++) {
        if (_mem_regions[j] && _mem_regions[j]->contains(iovecs[i].iov_base, iovecs[i].iov_len))
          iov->set_iovec_mr(i, _mem_regions[j]);
      }
    }
  }

  return iov->check_mr();
}

bool Fabric::up() const { return _up; }

bool Fabric::has_rma_event_support() const
{
  if (_info && (_info->caps & FI_RMA_EVENT)) {
    return true;
  } else {
    return false;
  }
}

const char* Fabric::domain_name() const
{
  if (_info) {
    return _info->domain_attr->name;
  } else {
    return NULL;
  }
}

const char* Fabric::fabric_name() const
{
  if (_info) {
    return _info->fabric_attr->name;
  } else {
    return NULL;
  }
}

const char* Fabric::provider() const
{
  if (_info) {
    return _info->fabric_attr->prov_name;
  } else {
    return NULL;
  }
}

uint32_t Fabric::version() const
{
  if (_info) {
    return _info->fabric_attr->prov_version;
  } else {
    return 0;
  }
}

struct addrinfo* Fabric::addrInfo() const { return _addrInfo; }

struct fi_info* Fabric::info() const { return _info; }

struct fid_fabric* Fabric::fabric() const { return _fabric; }

struct fid_domain* Fabric::domain() const { return _domain; }

bool Fabric::initialize(const char* node, const char* service, uint64_t flags)
{
  if (!_up) {
    if (!_hints) {
      _errno = -FI_ENODATA;
      set_error("Info::hints");
      return false;
    }

    struct addrinfo aihints;
    memset(&aihints, 0, sizeof aihints);
    aihints.ai_flags = AI_PASSIVE;
    int ret = getaddrinfo(node, service, &aihints, &_addrInfo);
    if (ret == EAI_SYSTEM) {
      set_custom_error("getaddrinfo for %s:%s: %s",
                       node, service, strerror(errno));
      _errno = -errno;
      return false;
    } else if (ret) {
      set_custom_error("getaddrinfo: %s", gai_strerror(ret));
      _errno = -FI_ENODATA;
      return false;
    }

    CHECK_ERR(fi_getinfo(FIVER, node ? node : ANY_ADDR, service, flags, _hints, &_info), "fi_getinfo");
    // sockets provider and maybe others ignore the hints so let's explicitly set the mr_mode bits.
    if (_info->domain_attr->mr_mode == FI_MR_UNSPEC) {
      _info->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    }
    CHECK_ERR(fi_fabric(_info->fabric_attr, &_fabric, NULL), "fi_fabric");
    CHECK_ERR(fi_domain(_fabric, _info, &_domain, NULL), "fi_domain");
  }

  return true;
}

void Fabric::shutdown()
{
  if (_up) {
    for (unsigned i=0; i<_mem_regions.size(); i++) {
      if(_mem_regions[i]) delete _mem_regions[i];
    }
    _mem_regions.clear();
    if (_domain) {
      fi_close(&_domain->fid);
      _domain = 0;
    }
    if (_fabric) {
      fi_close(&_fabric->fid);
      _fabric = 0;
    }
    if (_hints) {
      if (_hints_owner)  fi_freeinfo(_hints);
      _hints = 0;
    }
    if (_info) {
      fi_freeinfo(_info);
      _info = 0;
    }
  }
  _up = false;
}


EndpointBase::EndpointBase(const char* addr, const char* port, uint64_t flags, Info* hints) :
  _state(EP_INIT),
  _fab_owner(true),
  _eq_owner(true),
  _txcq_owner(true),
  _rxcq_owner(true),
  _fabric(new Fabric(addr, port, flags, hints)),
  _eq(0),
  _txcq(0),
  _rxcq(0)
{
  if (!initialize())
    _state = EP_CLOSED;
}

EndpointBase::EndpointBase(Fabric* fabric, EventQueue* eq, CompletionQueue* txcq, CompletionQueue* rxcq) :
  _state(EP_INIT),
  _fab_owner(false),
  _eq_owner(false),
  _txcq_owner(false),
  _rxcq_owner(false),
  _fabric(fabric),
  _eq(eq),
  _txcq(txcq),
  _rxcq(rxcq)
{
  if (!initialize())
    _state = EP_CLOSED;
}

EndpointBase::~EndpointBase()
{
  shutdown();
  if (_fab_owner) {
    delete _fabric;
    _fabric = 0;
  }
}

State EndpointBase::state() const { return _state; };

Fabric* EndpointBase::fabric() const { return _fabric; };

EventQueue* EndpointBase::eq() const { return _eq; }

CompletionQueue* EndpointBase::txcq() const { return _txcq; }

CompletionQueue* EndpointBase::rxcq() const { return _rxcq; }

bool EndpointBase::event(uint32_t* event, void* entry, bool* cm_entry)
{
  CHECK_OBJ(_eq->event(event, entry, cm_entry), _eq);

  return true;
}

bool EndpointBase::event_wait(uint32_t* event, void* entry, bool* cm_entry, int timeout)
{
  CHECK_OBJ(_eq->event_wait(event, entry, cm_entry, timeout), _eq);

  return true;
}

bool EndpointBase::event_error(struct fi_eq_err_entry *entry)
{
  CHECK_OBJ(_eq->event_error(entry), _eq);

  return true;
}

bool EndpointBase::handle_event(ssize_t event_ret, bool* cm_entry, const char* cmd)
{
  CHECK_OBJ(_eq->handle_event(event_ret, cm_entry, cmd), _eq);

  return true;
}

void EndpointBase::shutdown()
{
  if (_rxcq && _rxcq_owner) {
    delete _rxcq;
    if (_txcq == _rxcq)
      _txcq = 0;
    _rxcq = 0;
  }
  if (_txcq && _txcq_owner) {
    delete _txcq;
    _txcq = 0;
  }
  if (_eq && _eq_owner) {
    delete _eq;
    _eq = 0;
  }

  _state = EP_CLOSED;
}

bool EndpointBase::initialize()
{
  if (!_fabric->up()) {
    _errno = _fabric->error_num();
    set_custom_error(_fabric->error());
    return false;
  }

  struct fi_eq_attr eq_attr;

  memset(&eq_attr, 0, sizeof(eq_attr));
  eq_attr.size = 0;
  eq_attr.flags = 0;
  eq_attr.wait_obj = FI_WAIT_UNSPEC;

  struct fi_cq_attr cq_attr;

  memset(&cq_attr, 0, sizeof(cq_attr));
  cq_attr.size = 0;
  cq_attr.flags = 0;
  cq_attr.format = FI_CQ_FORMAT_DATA;
  cq_attr.wait_obj = FI_WAIT_UNSPEC;

  if (!_eq) {
    _eq = new EventQueue(_fabric, &eq_attr, NULL);
    if (!_eq)  return false;
    _eq_owner = true;
  }

  if (!_txcq) {
    _txcq = new CompletionQueue(_fabric, &cq_attr, NULL);
    if (!_txcq)  return false;
    _txcq_owner = true;
  }
  if (!_rxcq) {
    _rxcq = _txcq_owner ? _txcq : new CompletionQueue(_fabric, &cq_attr, NULL);
    if (!_rxcq)  return false;
    _rxcq_owner = !_txcq_owner;
  }

  _state = EP_UP;

  return true;
}


Endpoint::Endpoint(const char* addr, const char* port, uint64_t flags, Info* hints) :
  EndpointBase(addr, port, flags, hints),
  _counter(0),
  _ep(0)
{}

Endpoint::Endpoint(Fabric* fabric, EventQueue* eq, CompletionQueue* txcq, CompletionQueue* rxcq) :
  EndpointBase(fabric, eq, txcq, rxcq),
  _counter(0),
  _ep(0)
{}

Endpoint::~Endpoint()
{
  shutdown();
}

struct fid_ep* Endpoint::endpoint() const
{
  return _ep;
}

void Endpoint::shutdown()
{
  if (_ep) {
    fi_shutdown(_ep, 0);
    fi_close(&_ep->fid);
    _ep = 0;
  }

  // Application should call EndpointBase::shutdown() if wanted
  // This allows the state to be unwound to the point that connect()/accept()
  // can be called again without reallocating underlying objects

  _state = EP_INIT;
}

bool Endpoint::complete_connect(int timeout)
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  CHECK(event_wait(&event, &entry, &cm_entry, timeout));

  if (!cm_entry || event != FI_CONNECTED) {
    set_custom_error("unexpected event %u - expected FI_CONNECTED (%u)", event, FI_CONNECTED);
    _errno = -FI_ECONNREFUSED;
    return false;
  }

  _state = EP_CONNECTED;

  return true;
}

bool Endpoint::connect(int timeout, uint64_t txFlags, uint64_t rxFlags, void* context)
{
  if (check_connection_state() != FI_SUCCESS)
    return false;

  if (_txcq_owner && !txFlags)  txFlags  = FI_TRANSMIT;
  if (_rxcq_owner && !rxFlags)  rxFlags  = FI_RECV;
  if (_txcq == _rxcq)           txFlags |= rxFlags;

  CHECK_ERR(fi_endpoint(_fabric->domain(), _fabric->info(), &_ep, context), "fi_endpoint");
  CHECK_OBJ(_eq->bind(this), _eq);
  CHECK_OBJ(_txcq->bind(this, txFlags), _txcq);
  if (_txcq != _rxcq) {
    CHECK_OBJ(_rxcq->bind(this, rxFlags), _rxcq);
  }
  CHECK_ERR(fi_enable(_ep), "fi_enable");
  CHECK_ERR(fi_connect(_ep, _fabric->info()->dest_addr, NULL, 0), "fi_connect");

  return complete_connect(timeout);
}

bool Endpoint::accept(struct fi_info* remote_info, int timeout, uint64_t txFlags, uint64_t rxFlags, void* context)
{
  if (check_connection_state() != FI_SUCCESS)
    return false;

  if (_txcq_owner && !txFlags)  txFlags  = FI_TRANSMIT;
  if (_rxcq_owner && !rxFlags)  rxFlags  = FI_RECV;
  if (_txcq == _rxcq)           txFlags |= rxFlags;

  CHECK_ERR(fi_endpoint(_fabric->domain(), remote_info, &_ep, context), "fi_endpoint");
  CHECK_OBJ(_eq->bind(this), _eq);
  CHECK_OBJ(_txcq->bind(this, txFlags), _txcq);
  if (_txcq != _rxcq) {
    CHECK_OBJ(_rxcq->bind(this, rxFlags), _rxcq);
  }
  CHECK_ERR(fi_enable(_ep), "fi_enable");
  CHECK_ERR(fi_accept(_ep, NULL, 0), "fi_accept");

  return complete_connect(timeout);
}

ssize_t Endpoint::post_comp_data_recv(void* context)
{
  ssize_t rret = fi_recv(_ep, NULL, 0, NULL, 0, context);
  if ((rret != FI_SUCCESS) && (rret != -FI_EAGAIN)) {
    _errno = (int) rret;
    set_error("fi_recv");
  }

  return rret;
}

ssize_t Endpoint::recv_comp_data(void* context)
{
  // post a recieve buffer for remote completion
  return post_comp_data_recv(context);
}

ssize_t Endpoint::recv_comp_data_sync(CompletionQueue* cq, uint64_t* data)
{
  int context = _counter++;

  ssize_t rret = post_comp_data_recv(&context);
  if (rret == FI_SUCCESS) {
    return check_completion(cq, context, FI_REMOTE_CQ_DATA, data);
  }

  return rret;
}

ssize_t Endpoint::send(const void* buf, size_t len, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_send");

  ssize_t rret = fi_send(_ep, buf, len, mr->desc(), 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_send");
  }

  return rret;
}

ssize_t Endpoint::send_sync(const void* buf, size_t len, const MemoryRegion* mr)
{
  int context = _counter++;

  ssize_t rret = send(buf, len, &context, mr);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_SEND | FI_MSG);
  }

  return rret;
}

ssize_t Endpoint::send(const LocalAddress* laddr, void* context)
{
  return send(laddr->buf(), laddr->len(), context, laddr->mr());
}

ssize_t Endpoint::send_sync(const LocalAddress* laddr)
{
  return send_sync(laddr->buf(), laddr->len(), laddr->mr());
}

ssize_t Endpoint::recv(void* buf, size_t len, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_recv");

  ssize_t rret = fi_recv(_ep, buf, len, mr->desc(), 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_recv");
  }

  return rret;
}

ssize_t Endpoint::recv_sync(void* buf, size_t len, const MemoryRegion* mr)
{
  int context = _counter++;

  ssize_t rret = recv(buf, len, &context, mr);
  if (rret == FI_SUCCESS) {
    return check_completion(_rxcq, context, FI_RECV | FI_MSG);
  }

  return rret;
}

ssize_t Endpoint::recv(LocalAddress* laddr, void* context)
{
  return recv(laddr->buf(), laddr->len(), context, laddr->mr());
}

ssize_t Endpoint::recv_sync(LocalAddress* laddr)
{
  return recv_sync(laddr->buf(), laddr->len(), laddr->mr());
}

ssize_t Endpoint::read(void* buf, size_t len, const RemoteAddress* raddr, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_read");

  ssize_t rret = fi_read(_ep, buf, len, mr->desc(), 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_read");
  }

  return rret;
}

ssize_t Endpoint::read_sync(void* buf, size_t len, const RemoteAddress* raddr, const MemoryRegion* mr)
{
  int context = _counter++;

  ssize_t rret = read(buf, len, raddr, &context, mr);
  if (rret == FI_SUCCESS) {
    return check_completion(_rxcq, context, FI_READ | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::read(LocalAddress* laddr, const RemoteAddress* raddr, void* context)
{
  return read(laddr->buf(), laddr->len(), raddr, context, laddr->mr());
}

ssize_t Endpoint::read_sync(LocalAddress* laddr, const RemoteAddress* raddr)
{
  return read_sync(laddr->buf(), laddr->len(), raddr, laddr->mr());
}

ssize_t Endpoint::write(const void* buf, size_t len, const RemoteAddress* raddr, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_write");

  ssize_t rret = fi_write(_ep, buf, len, mr->desc(), 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_write");
  }

  return rret;
}

ssize_t Endpoint::write_sync(const void* buf, size_t len, const RemoteAddress* raddr, const MemoryRegion* mr)
{
  int context = _counter++;

  ssize_t rret = write(buf, len, raddr, &context, mr);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_WRITE | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::write(const LocalAddress* laddr, const RemoteAddress* raddr, void* context)
{
  return write(laddr->buf(), laddr->len(), raddr, context, laddr->mr());
}

ssize_t Endpoint::write_sync(const LocalAddress* laddr, const RemoteAddress* raddr)
{
  return write_sync(laddr->buf(), laddr->len(), raddr, laddr->mr());
}

ssize_t Endpoint::writedata(const void* buf, size_t len, const RemoteAddress* raddr, void* context, uint64_t data, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_writedata");

  ssize_t rret = fi_writedata(_ep, buf, len, mr->desc(), data, 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_writedata");
  }

  return rret;
}

ssize_t Endpoint::writedata_sync(const void* buf, size_t len, const RemoteAddress* raddr, uint64_t data, const MemoryRegion* mr)
{
  int context = _counter++;

  ssize_t rret = writedata(buf, len, raddr, &context, data, mr);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_WRITE | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::writedata(const LocalAddress* laddr, const RemoteAddress* raddr, void* context, uint64_t data)
{
  return writedata(laddr->buf(), laddr->len(), raddr, context, data, laddr->mr());
}

ssize_t Endpoint::writedata_sync(const LocalAddress* laddr, const RemoteAddress* raddr, uint64_t data)
{
  return writedata_sync(laddr->buf(), laddr->len(), raddr, data, laddr->mr());
}

ssize_t Endpoint::inject_writedata(const void* buf, size_t len, const RemoteAddress* raddr, uint64_t data)
{
  ssize_t rret = fi_inject_writedata(_ep, buf, len, data, 0, raddr->addr, raddr->rkey);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_inject_writedata");
  }

  return rret;
}

ssize_t Endpoint::inject_writedata(const LocalAddress* laddr, const RemoteAddress* raddr, uint64_t data)
{
  return inject_writedata(laddr->buf(), laddr->len(), raddr, data);
}

ssize_t Endpoint::recvv(LocalIOVec* iov, void* context)
{
  CHECK_MR_IOVEC(iov, "fi_recvv");

  ssize_t rret = fi_recvv(_ep, iov->iovecs(), iov->desc(), iov->count(), 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_recvv");
  }

  return rret;
}

ssize_t Endpoint::recvv_sync(LocalIOVec* iov)
{
  int context = _counter++;

  ssize_t rret = recvv(iov, &context);
  if (rret == FI_SUCCESS) {
    return check_completion(_rxcq, context, FI_RECV | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::sendv(LocalIOVec* iov, void* context)
{
  CHECK_MR_IOVEC(iov, "fi_sendv");

  ssize_t rret = fi_sendv(_ep, iov->iovecs(), iov->desc(), iov->count(), 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_sendv");
  }

  return rret;
}

ssize_t Endpoint::sendv_sync(LocalIOVec* iov)
{
  int context = _counter++;

  ssize_t rret = sendv(iov, &context);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_SEND | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::recvmsg(Message* msg, uint64_t flags)
{
  CHECK_MR_IOVEC(msg->iov(), "fi_recvmsg");

  ssize_t rret = fi_recvmsg(_ep, msg->msg(), flags);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_readmsg");
  }

  return rret;
}

ssize_t Endpoint::recvmsg_sync(Message* msg, uint64_t flags)
{
  int context = _counter++;
  msg->context(&context);

  ssize_t rret = recvmsg(msg, flags);
  if (rret == FI_SUCCESS) {
    return check_completion(_rxcq, context, FI_RECV | FI_MSG);
  }

  return rret;
}

ssize_t Endpoint::sendmsg(Message* msg, uint64_t flags)
{
  CHECK_MR_IOVEC(msg->iov(), "fi_sendmsg");

  ssize_t rret = fi_sendmsg(_ep, msg->msg(), flags);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_sendmsg");
  }

  return rret;
}

ssize_t Endpoint::sendmsg_sync(Message* msg, uint64_t flags)
{
  int context = _counter++;
  msg->context(&context);

  ssize_t rret = sendmsg(msg, flags);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_SEND | FI_MSG);
  }

  return rret;
}

ssize_t Endpoint::readv(LocalIOVec* iov, const RemoteAddress* raddr, void* context)
{
  CHECK_MR_IOVEC(iov, "fi_readv");

  ssize_t rret = fi_readv(_ep, iov->iovecs(), iov->desc(), iov->count(), 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_readv");
  }

  return rret;
}

ssize_t Endpoint::readv_sync(LocalIOVec* iov, const RemoteAddress* raddr)
{
  int context = _counter++;

  ssize_t rret = readv(iov, raddr, &context);
  if (rret == FI_SUCCESS) {
    return check_completion(_rxcq, context, FI_READ | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::writev(LocalIOVec* iov, const RemoteAddress* raddr, void* context)
{
  CHECK_MR_IOVEC(iov, "fi_writev");

  ssize_t rret = fi_writev(_ep, iov->iovecs(), iov->desc(), iov->count(), 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_writev");
  }

  return rret;
}

ssize_t Endpoint::writev_sync(LocalIOVec* iov, const RemoteAddress* raddr)
{
  int context = _counter++;

  ssize_t rret = readv(iov, raddr, &context);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_READ | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::readmsg(RmaMessage* msg, uint64_t flags)
{
  CHECK_MR_IOVEC(msg->loc_iov(), "fi_readmsg");

  ssize_t rret = fi_readmsg(_ep, msg->msg(), flags);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_readmsg");
  }

  return rret;
}

ssize_t Endpoint::readmsg_sync(RmaMessage* msg, uint64_t flags)
{
  int context = _counter++;
  msg->context(&context);

  ssize_t rret = readmsg(msg, flags);
  if (rret == FI_SUCCESS) {
    return check_completion(_rxcq, context, FI_READ | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::writemsg(RmaMessage* msg, uint64_t flags)
{
  CHECK_MR_IOVEC(msg->loc_iov(), "fi_writemsg");

  ssize_t rret = fi_writemsg(_ep, msg->msg(), flags);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_writemsg");
  }

  return rret;
}

ssize_t Endpoint::writemsg_sync(RmaMessage* msg, uint64_t flags)
{
  int context = _counter++;
  msg->context(&context);

  ssize_t rret = writemsg(msg, flags);
  if (rret == FI_SUCCESS) {
    return check_completion(_txcq, context, FI_WRITE | FI_RMA);
  }

  return rret;
}

ssize_t Endpoint::injectdata(const void* buf, size_t len, uint64_t data)
{
  ssize_t rret = fi_injectdata(_ep, buf, len, data, 0);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_injectdata");
  }

  return rret;
}

ssize_t Endpoint::injectdata(const LocalAddress* laddr, uint64_t data)
{
  return injectdata(laddr->buf(), laddr->len(), data);
}

ssize_t Endpoint::check_completion(CompletionQueue* cq, int context, unsigned flags, uint64_t* data)
{
  ssize_t rret = cq->check_completion(context, flags, data);
  if (rret != FI_SUCCESS) {
    _errno = (int) cq->error_num();
    set_custom_error(cq->error());
  }

  return rret;
}

ssize_t Endpoint::check_completion_noctx(CompletionQueue* cq, unsigned flags, uint64_t* data)
{
  ssize_t rret = cq->check_completion_noctx(flags, data);
  if (rret != FI_SUCCESS) {
    _errno = (int) cq->error_num();
    set_custom_error(cq->error());
  }

  return rret;
}

ssize_t Endpoint::check_connection_state()
{
  ssize_t rret = FI_SUCCESS;

  if (_state > EP_UP) {
    if (_state == EP_CONNECTED) {
      rret = -FI_EISCONN;
      set_error("Endpoint is already in a connected state");
    } else {
      rret = -FI_EOPNOTSUPP;
      set_error("Connecting is not possible in current state");
    }
  }
  else if (_state < EP_UP) {
    if (!EndpointBase::initialize()) {
      rret = -FI_EOPNOTSUPP;
      set_error("Connecting is not possible in current state");
    }
  }
  _errno = rret;

  return rret;
}

PassiveEndpoint::PassiveEndpoint(const char* addr, const char* port, uint64_t flags, Info* hints) :
  EndpointBase(addr, port, flags | FI_SOURCE, hints),
  _flags(flags),
  _pep(0)
{}

PassiveEndpoint::~PassiveEndpoint()
{
  shutdown();
  for (unsigned i=0; i<_endpoints.size(); i++) {
    if(_endpoints[i]) delete _endpoints[i];
  }
  _endpoints.clear();
}

void PassiveEndpoint::shutdown()
{
  for (unsigned i=0; i<_endpoints.size(); i++) {
    _endpoints[i]->shutdown();
  }
  if (_pep) {
    fi_close(&_pep->fid);
    _pep = 0;
  }

  // Application should call EndpointBase::shutdown() if wanted
  // This allows the state to be unwound to the point that listen()
  // can be called again without reallocating underlying objects

  _state = EP_INIT;
}

bool PassiveEndpoint::listen(int backlog)
{
  CHECK_ERR(fi_passive_ep(_fabric->fabric(), _fabric->info(), &_pep, NULL), "fi_passive_ep");
  CHECK_ERR(fi_pep_bind(_pep, &_eq->eq()->fid, 0), "fi_pep_bind(eq)");
  // Attempt to set the backlog parameter of the pep and ignore the failure if the provider doesn't support it.
  CHECK_ERR_EX(fi_control(&_pep->fid, FI_BACKLOG, &backlog), "fi_control(FI_BACKLOG)", FI_ENOSYS);
  CHECK_ERR(fi_listen(_pep), "fi_listen");

  _state = EP_LISTEN;

  return true;
}

bool PassiveEndpoint::listen(int backlog, uint16_t& port)
{
  struct addrinfo* ai = _fabric->addrInfo();

  CHECK_ERR(fi_passive_ep(_fabric->fabric(), _fabric->info(), &_pep, NULL), "fi_passive_ep");
  CHECK_ERR(fi_setname(&_pep->fid, ai->ai_addr, ai->ai_addrlen), "fi_setname");
  CHECK_ERR(fi_pep_bind(_pep, &_eq->eq()->fid, 0), "fi_pep_bind(eq)");
  // Attempt to set the backlog parameter of the pep and ignore the failure if the provider doesn't support it.
  CHECK_ERR_EX(fi_control(&_pep->fid, FI_BACKLOG, &backlog), "fi_control(FI_BACKLOG)", FI_ENOSYS);
  CHECK_ERR(fi_listen(_pep), "fi_listen");

  struct sockaddr_in bound_addr;
  size_t bound_addr_len = sizeof(bound_addr);
  CHECK_ERR(fi_getname(&_pep->fid, &bound_addr, &bound_addr_len), "fi_getname");
  port = ntohs(bound_addr.sin_port);

  _state = EP_LISTEN;

  return true;
}

Endpoint* PassiveEndpoint::accept(int timeout, EventQueue* eq, CompletionQueue* txcq, uint64_t txFlags, CompletionQueue* rxcq, uint64_t rxFlags, void* context)
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  if (_state != EP_LISTEN) {
    _errno = -FI_EOPNOTSUPP;
    set_error("Passive endpoint must be in a listening state to open connections");
    return NULL;
  }

  if (!event_wait(&event, &entry, &cm_entry, timeout))
    return NULL;

  if (!cm_entry || event != FI_CONNREQ) {
    set_custom_error("unexpected event %u - expected FI_CONNECTED (%u)", event, FI_CONNREQ);
    _errno = -FI_ECONNABORTED;
    return NULL;
  }

  Endpoint *endp = new Endpoint(_fabric, eq, txcq, rxcq);
  int tmo = -1;
  if (endp->accept(entry.info, tmo, txFlags, rxFlags, context)) {
    _endpoints.push_back(endp);
    return endp;
  } else {
    _errno = -FI_ECONNABORTED;
    set_error("active endpoint creation failed");
    fi_reject(_pep, entry.info->handle, NULL, 0);
    delete endp;
    return NULL;
  }
}

bool PassiveEndpoint::reject(int timeout)
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  if (_state != EP_LISTEN) {
    _errno = -FI_EOPNOTSUPP;
    set_error("Passive endpoint must be in a listening state to reject connections");
    return false;
  }

  CHECK(event_wait(&event, &entry, &cm_entry, timeout));

  if (!cm_entry || event != FI_CONNREQ) {
    set_custom_error("unexpected event %u - expected FI_CONNECTED (%u)", event, FI_CONNREQ);
    _errno = -FI_ECONNABORTED;
    return false;
  }

  CHECK_ERR(fi_reject(_pep, entry.info->handle, NULL, 0), "fi_reject");

  return true;
}

bool PassiveEndpoint::close(Endpoint* endpoint)
{
  if (endpoint) {
    for (std::vector<Endpoint*>::iterator it=_endpoints.begin(); it!=_endpoints.end(); ++it) {
      if (*it == endpoint) {
        _endpoints.erase(it);
        delete endpoint;
        break;
      }
    }
  }

  return true;
}


CompletionPoller::CompletionPoller(Fabric* fabric, nfds_t size_hint) :
  _up(false),
  _fabric(fabric),
  _nfd(0),
  _nfd_max(size_hint),
  _pfd(0),
  _pfid(0),
  _endps(0)
{
  _up = initialize();
}

CompletionPoller::~CompletionPoller()
{
  shutdown();
}

bool CompletionPoller::up() const { return _up; }

bool CompletionPoller::add(Endpoint* endp, CompletionQueue* cq)
{
  for (unsigned i=0; i<_nfd; i++) {
    if (_endps[i] == endp)
      return false;
  }

  check_size();

  CHECK_ERR(fi_control(&cq->cq()->fid, FI_GETWAIT, (void *) &_pfd[_nfd].fd), "fi_control");

  _pfd[_nfd].events   = POLLIN;
  _pfd[_nfd].revents  = 0;
  _pfid[_nfd]         = &cq->cq()->fid;
  _endps[_nfd]        = endp;
  _nfd++;

  return true;
}

bool CompletionPoller::del(Endpoint* endp)
{
  unsigned pos = 0;

  for (unsigned i=0; i<_nfd; i++) {
    if (_endps[i] == endp)
      break;
    pos++;
  }

  if (pos < _nfd) {
    for (; pos<(_nfd-1); pos++) {
      _pfd[pos] = _pfd[pos+1];
      _pfid[pos] = _pfid[pos+1];
      _endps[pos] = _endps[pos+1];
    }
    _nfd--;
    return true;
  } else {
    return false;
  }
}

int CompletionPoller::poll(int timeout)
{
  int ret = 0;

  ret = fi_trywait(_fabric->fabric(), _pfid, _nfd);
  if (ret == -FI_EAGAIN) {
    return ret;
  }
  if (ret == FI_SUCCESS) {
    ret = ::poll(_pfd, _nfd, timeout);
    if (ret >= 0) {
      clear_error();
    } else {
      _errno = ret;
      set_error("poll");
    }
    return ret;
  }
  _errno = ret;
  set_error("fi_trywait");
  return ret;
}

void CompletionPoller::check_size()
{
  if (_up && (_nfd >= _nfd_max)) {
    _nfd_max *= 2;

    struct pollfd* pfd_new = new pollfd[_nfd_max];
    struct fid** pfid_new = new struct fid*[_nfd_max];
    Endpoint** endp_new = new Endpoint*[_nfd_max];

    for (unsigned i=0; i<_nfd; i++) {
      pfd_new[i] = _pfd[i];
      pfid_new[i] = _pfid[i];
      endp_new[i] = _endps[i];
    }

    delete[] _pfd;
    delete[] _pfid;
    delete[] _endps;

    _pfd = pfd_new;
    _pfid = pfid_new;
    _endps = endp_new;
  }
}

bool CompletionPoller::initialize()
{
  if (!_fabric->up()) {
    _errno = _fabric->error_num();
    set_custom_error(_fabric->error());
    return false;
  }

  if (!_up) {
    _pfd = new pollfd[_nfd_max];
    _pfid = new struct fid*[_nfd_max];
    _endps = new Endpoint*[_nfd_max];
  }

  return true;
}

void CompletionPoller::shutdown()
{
  if (_up) {
    if (_pfd) {
      delete[] _pfd;
    }
    if (_pfid) {
      delete[] _pfid;
    }
    if (_endps) {
      delete[] _endps;
    }
  }
  _up = false;
}

CompletionQueue::CompletionQueue(Fabric* fabric, size_t size) :
  _up(false),
  _fabric(fabric),
  _cq(nullptr)
{
  struct fi_cq_attr cq_attr;

  memset(&cq_attr, 0, sizeof(cq_attr));
  cq_attr.size = size;
  cq_attr.flags = 0;
  cq_attr.format = FI_CQ_FORMAT_DATA;
  cq_attr.wait_obj = FI_WAIT_UNSPEC;

  _up = initialize(&cq_attr, NULL);
}

CompletionQueue::CompletionQueue(Fabric* fabric, struct fi_cq_attr* cq_attr, void* context) :
  _up(false),
  _fabric(fabric),
  _cq(nullptr)
{
  _up = initialize(cq_attr, context);
}

CompletionQueue::~CompletionQueue()
{
  shutdown();
}

struct fid_cq* CompletionQueue::cq() const { return _cq; }

ssize_t CompletionQueue::comp_error(struct fi_cq_err_entry* comp_err)
{
  ssize_t rret = fi_cq_readerr(_cq, comp_err, 0);
  if (rret < 0) {
    set_error("fi_cq_readerr");
    _errno = (int) rret;
  } else if (rret == 0) {
    set_custom_error("fi_cq_readerr: no errors to be read");
    rret = FI_SUCCESS;
  }

  return rret;
}

#pragma GCC diagnostic ignored "-Wunused-function"

static void comp_error_dump(struct fi_cq_err_entry* comp_err)
{
  printf ("void*    op_context    %p\n",     comp_err->op_context);
  printf ("uint64_t flags         %016lx\n", comp_err->flags);
  printf ("size_t   len           %zd\n",    comp_err->len);
  printf ("void*    buf           %p\n",     comp_err->buf);
  printf ("uint64_t data          %016lx\n", comp_err->data);
  printf ("uint64_t tag           %016lx\n", comp_err->tag);
  printf ("size_t   olen          %zd\n",    comp_err->olen);
  printf ("int      err           %d\n",     comp_err->err);
  printf ("int      prov_errno    %d\n",     comp_err->prov_errno);
  /* err_data is available until the next time the CQ is read */
  printf ("size_t   err_data_size %zd\n",    comp_err->err_data_size);
  printf ("void*    err_data      %p\n",     comp_err->err_data);
  if (comp_err->err_data_size) {
    uint32_t* ptr = (uint32_t*)comp_err->err_data;
    for (unsigned i = 0; i < comp_err->err_data_size; ++i)
      printf("  %08x", *ptr++);
    printf("\n");
  }
}

#pragma GCC diagnostic pop

ssize_t CompletionQueue::handle_comp(ssize_t comp_ret, struct fi_cq_data_entry* comp, const char* cmd)
{
  struct fi_cq_err_entry comp_err;
  //uint32_t err_data = 0xeeeeeeee;
  //memset(&comp_err, 0xee, sizeof(comp_err));
  //comp_err.buf      = nullptr;
  //comp_err.err_data = &err_data;

  if ((comp_ret < 0) && (comp_ret != -FI_EAGAIN)) {
    _errno = (int) comp_ret;
    if (comp_ret == -FI_EAVAIL) {
      if (comp_error(&comp_err) > 0) {
        char buf[ERR_MSG_LEN];
        fi_cq_strerror(_cq, comp_err.prov_errno, comp_err.err_data, buf, sizeof(buf));
        set_custom_error("%s: %s(%d)", cmd, buf, comp_err.prov_errno);
        //comp_error_dump(&comp_err);
      }
    } else {
      set_error(cmd);
    }
  }
  return comp_ret;
}

ssize_t CompletionQueue::comp(struct fi_cq_data_entry* comp, ssize_t max_count)
{
  return handle_comp(fi_cq_read(_cq, comp, max_count), comp, "fi_cq_read");
}

ssize_t CompletionQueue::comp_wait(struct fi_cq_data_entry* comp, ssize_t max_count, int timeout)
{
  return handle_comp(fi_cq_sread(_cq, comp, max_count, NULL, timeout), comp, "fi_cq_sread");
}

ssize_t CompletionQueue::check_completion(int context, unsigned flags, uint64_t* data)
{
  struct fi_cq_data_entry comp;
  //memset(&comp, 0xee, sizeof(comp));

  ssize_t rret = comp_wait(&comp, 1);
  if (rret == 1) {
    if ((comp.flags & flags) == flags) {
      if (comp.op_context) {
        if (*((int*) comp.op_context) == context) {
          if (data)
            *data = comp.data;
          //dump_cq_data_entry(comp); // Temporary
          return FI_SUCCESS;
        } else {
          set_custom_error("Wrong completion: comp.op_context value is %d, expected %d", *(int*)comp.op_context, context);
          dump_cq_data_entry(comp);
          return -EFAULT;
        }
      } else {
        set_custom_error("Wrong completion: comp.op_context is %p, expected %d", comp.op_context, context);
        dump_cq_data_entry(comp);
        return -EFAULT;
      }
    }
  }

  return rret ? rret : -FI_EAGAIN;     // Revisit case when comp_wait returns 0
}

ssize_t CompletionQueue::check_completion_noctx(unsigned flags, uint64_t* data)
{
  struct fi_cq_data_entry comp;

  ssize_t rret = comp_wait(&comp, 1);
  if (rret == 1) {
    if ((comp.flags & flags) == flags) {
      if (data)
        *data = comp.data;
      return FI_SUCCESS;
    }
  }

  return rret ? rret : -FI_EAGAIN;     // Revisit case when comp_wait returns 0
}

void CompletionQueue::dump_cq_data_entry(struct fi_cq_data_entry& comp)
{
  printf("op_context: %p",       comp.op_context);
  if (comp.op_context)
    printf(" -> 0x%08x",  *(int*)comp.op_context);
  printf("\n");
  printf("flags:      %016lx\n", comp.flags);
  printf("len:        %zd\n",    comp.len);
  printf("buf:        %p\n",     comp.buf);
  printf("data:       %016lx\n", comp.data);
}

bool CompletionQueue::up() const { return _up; }

bool CompletionQueue::bind(Endpoint* ep, uint64_t flags)
{
  CHECK_ERR(fi_ep_bind(ep->endpoint(), &_cq->fid, flags), "fi_ep_bind()");

  return true;
}

bool CompletionQueue::initialize(struct fi_cq_attr* cq_attr, void* context)
{
  if (!_fabric->up()) {
    _errno = _fabric->error_num();
    set_custom_error(_fabric->error());
    return false;
  }

  if (!_up) {
    CHECK_ERR(fi_cq_open(_fabric->domain(), cq_attr, &_cq, context), "fi_cq_open()");
  }

  return true;
}

void CompletionQueue::shutdown()
{
  if (_up) {
    if (_cq) {
      fi_close(&_cq->fid);
      _cq = nullptr;
    }
  }
  _up = false;
}

EventQueue::EventQueue(Fabric* fabric, size_t size) :
  _up(false),
  _fabric(fabric),
  _eq(nullptr)
{
  struct fi_eq_attr eq_attr;

  memset(&eq_attr, 0, sizeof(eq_attr));
  eq_attr.size = size;
  eq_attr.flags = 0;
  eq_attr.wait_obj = FI_WAIT_UNSPEC;

  _up = initialize(&eq_attr, NULL);
}

EventQueue::EventQueue(Fabric* fabric, struct fi_eq_attr* eq_attr, void* context) :
  _up(false),
  _fabric(fabric),
  _eq(nullptr)
{
  _up = initialize(eq_attr, context);
}

EventQueue::~EventQueue()
{
  shutdown();
}

struct fid_eq* EventQueue::eq() const { return _eq; }

bool EventQueue::event(uint32_t* event, void* entry, bool* cm_entry)
{
  return handle_event(fi_eq_read(_eq, event, entry, sizeof (struct fi_eq_cm_entry), 0), cm_entry, "fi_eq_read");
}

bool EventQueue::event_wait(uint32_t* event, void* entry, bool* cm_entry, int timeout)
{
  return handle_event(fi_eq_sread(_eq, event, entry, sizeof (struct fi_eq_cm_entry), timeout, 0), cm_entry, "fi_eq_sread");
}

bool EventQueue::event_error(struct fi_eq_err_entry *entry)
{
  memset(entry, 0, sizeof(*entry));     // Avoid valgrind's uninitialized memory commentary
  ssize_t rret = fi_eq_readerr(_eq, entry, 0);
  if (rret != sizeof (struct fi_eq_err_entry)) {
    if (rret < 0) {
      _errno = (int) rret;
      set_error("fi_eq_readerr");
    } else if (rret == 0) {
      set_custom_error("fi_eq_readerr: no errors to be read");
      _errno = FI_SUCCESS;
    } else {
      _errno = -FI_ETRUNC;
      set_error("fi_eq_readerr");
    }

    return false;
  }

  return true;
}

bool EventQueue::handle_event(ssize_t event_ret, bool* cm_entry, const char* cmd)
{
  struct fi_eq_err_entry entry;

  if (event_ret == sizeof (struct fi_eq_cm_entry)) {
    *cm_entry = true;
  } else if (event_ret == sizeof (struct fi_eq_entry)) {
    *cm_entry = false;
  } else {
    if (event_ret != 0) {
      if (event_ret < 0) {
        _errno = (int) event_ret;
        if (_errno == -FI_EAVAIL) {
          if (event_error(&entry)) {
            _errno = -entry.err;
          }
        }
        set_error(cmd);
      } else {
        _errno = -FI_ETRUNC;
        set_error(cmd);
      }
    } else {
      _errno = -FI_ENODATA;
      set_custom_error("no events seen within timeout by %s", cmd);
    }

    return false;
  }

  return true;
}

bool EventQueue::up() const { return _up; }

bool EventQueue::bind(Endpoint* ep)
{
  CHECK_ERR(fi_ep_bind(ep->endpoint(), &_eq->fid, 0), "fi_ep_bind()");

  return true;
}

bool EventQueue::initialize(struct fi_eq_attr* eq_attr, void* context)
{
  if (!_fabric->up()) {
    _errno = _fabric->error_num();
    set_custom_error(_fabric->error());
    return false;
  }

  if (!_up) {
    CHECK_ERR(fi_eq_open(_fabric->fabric(), eq_attr, &_eq, context), "fi_eq_open");
  }

  return true;
}

void EventQueue::shutdown()
{
  if (_up) {
    if (_eq) {
      fi_close(&_eq->fid);
      _eq = nullptr;
    }
  }
  _up = false;
}

#undef ERR_MSG_LEN
#undef ANY_ADDR
#undef CHECK_ERR
#undef CHECK_ERR_EX
#undef CHECK
#undef CHECK_MR
#undef CHECK_MR_IOVEC
