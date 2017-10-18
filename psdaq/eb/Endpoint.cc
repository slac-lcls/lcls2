#include "Endpoint.hh"

#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define FIVER FI_VERSION(1, 4)

#define ERR_MSG_LEN 256

#define CHECK_ERR(function, msg)  \
  _errno = function;              \
  if (_errno != FI_SUCCESS) {     \
    set_error(msg);               \
    return false;                 \
  }

#define CHECK(function)     \
  if (!function)            \
    return false;

#define CHECK_MR(buf, len, mr, cmd)                                                                                                 \
  if (!mr) {                                                                                                                        \
    mr = _fabric->lookup_memory(buf, len);                                                                                          \
    if (!mr) {                                                                                                                      \
      set_custom_error("%s: requested buffer starting at %p with len %lu is not within a registered memory region", cmd, buf, len); \
      _errno = FI_EINVAL;                                                                                                           \
      return false;                                                                                                                 \
    }                                                                                                                               \
  }


using namespace Pds::Fabrics;

RemoteAddress::RemoteAddress() :
  rkey(0),
  addr(0),
  extent(0)
{}

RemoteAddress::RemoteAddress(uint64_t rkey, uint64_t addr, size_t extent) :
  rkey(rkey),
  addr(addr),
  extent(extent)
{}

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

bool MemoryRegion::contains(void* start, size_t len) const
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


Fabric::Fabric(const char* node, const char* service, int flags) :
  _up(false),
  _hints(0),
  _info(0),
  _fabric(0),
  _domain(0)
{
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

MemoryRegion* Fabric::lookup_memory(void* start, size_t len) const
{
  for (unsigned i=0; i<_mem_regions.size(); i++) {
    if (_mem_regions[i] && _mem_regions[i]->contains(start, len))
      return _mem_regions[i];
  }

  return NULL;
}

bool Fabric::up() const { return _up; }

struct fi_info* Fabric::info() const { return _info; }

struct fid_fabric* Fabric::fabric() const { return _fabric; }

struct fid_domain* Fabric::domain() const { return _domain; }

bool Fabric::initialize(const char* node, const char* service, int flags)
{
  if (!_up) {
    _hints = fi_allocinfo();
    if (!_hints) {
      _errno = -FI_ENOMEM;
      set_error("fi_allocinfo");
      return false;
    }

    _hints->addr_format = FI_SOCKADDR_IN;
    _hints->ep_attr->type = FI_EP_MSG;
    _hints->domain_attr->mr_mode = FI_MR_BASIC;
    _hints->caps = FI_MSG | FI_RMA;
    _hints->mode = FI_CONTEXT | FI_LOCAL_MR | FI_RX_CQ_DATA;

    if (!node)
      flags |= FI_SOURCE;

    CHECK_ERR(fi_getinfo(FIVER, node, service, flags, _hints, &_info), "fi_getinfo");
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
      fi_freeinfo(_hints);
      _hints = 0;
    }
    if (_info) {
      fi_freeinfo(_info);
      _info = 0;
    }
  }
  _up = false;
}


EndpointBase::EndpointBase(const char* addr, const char* port, int flags) :
  _state(EP_INIT),
  _fab_owner(true),
  _fabric(new Fabric(addr, port, flags)),
  _eq(0),
  _cq(0)
{
  if (!initialize())
    _state = EP_CLOSED;
}

EndpointBase::EndpointBase(Fabric* fabric) :
  _state(EP_INIT),
  _fab_owner(false),
  _fabric(fabric),
  _eq(0),
  _cq(0)
{
  if (!initialize())
    _state = EP_CLOSED;
}

EndpointBase::~EndpointBase()
{
  shutdown();
  if (_fab_owner) {
    delete _fabric;
  }
}

State EndpointBase::state() const { return _state; };

Fabric* EndpointBase::fabric() const { return _fabric; };

struct fid_eq* EndpointBase::eq() const { return _eq; }

struct fid_cq* EndpointBase::cq() const { return _cq; }

bool EndpointBase::event(uint32_t* event, void* entry, bool* cm_entry)
{
  return handle_event(fi_eq_read(_eq, event, entry, sizeof (struct fi_eq_cm_entry), 0), cm_entry, "fi_eq_read");
}

bool EndpointBase::event_wait(uint32_t* event, void* entry, bool* cm_entry, int timeout)
{
  return handle_event(fi_eq_sread(_eq, event, entry, sizeof (struct fi_eq_cm_entry), timeout, 0), cm_entry, "fi_eq_sread");
}

bool EndpointBase::event_error(struct fi_eq_err_entry *entry)
{
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

bool EndpointBase::handle_event(ssize_t event_ret, bool* cm_entry, const char* cmd)
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

void EndpointBase::shutdown()
{
  if (_cq) {
    fi_close(&_cq->fid);
    _cq = 0;
  }
  if (_eq) {
    fi_close(&_eq->fid);
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

  struct fi_eq_attr eq_attr = {
    .size = 0,
    .flags = 0,
    .wait_obj = FI_WAIT_UNSPEC,
    .signaling_vector = 0,
    .wait_set = NULL,
  };

  struct fi_cq_attr cq_attr = {
    .size = 0,
    .flags = 0,
    .format = FI_CQ_FORMAT_DATA,
    .wait_obj = FI_WAIT_UNSPEC,
    .signaling_vector = 0,
    .wait_cond = FI_CQ_COND_NONE,
    .wait_set = NULL,
  };
  
  CHECK_ERR(fi_eq_open(_fabric->fabric(), &eq_attr, &_eq, NULL), "fi_eq_open");
  CHECK_ERR(fi_cq_open(_fabric->domain(), &cq_attr, &_cq, NULL), "fi_cq_open");

  _state = EP_UP;

  return true;
}


Endpoint::Endpoint(const char* addr, const char* port, int flags) :
  EndpointBase(addr, port, flags),
  _ep(0)
{}

Endpoint::Endpoint(Fabric* fabric) :
  EndpointBase(fabric),
  _ep(0)
{}

Endpoint::~Endpoint()
{
  shutdown();
}

void Endpoint::shutdown()
{
  if (_ep) {
    fi_shutdown(_ep, 0);
    fi_close(&_ep->fid);
    _ep = 0;
  }

  EndpointBase::shutdown();
}

bool Endpoint::connect()
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  if (!check_connection_state())
    return false;

  CHECK_ERR(fi_endpoint(_fabric->domain(), _fabric->info(), &_ep, NULL), "fi_endpoint");
  CHECK_ERR(fi_ep_bind(_ep, &_eq->fid, 0), "fi_ep_bind(eq)");
  CHECK_ERR(fi_ep_bind(_ep, &_cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK_ERR(fi_enable(_ep), "fi_enable");
  CHECK_ERR(fi_connect(_ep, _fabric->info()->dest_addr, NULL, 0), "fi_connect");

  CHECK(event_wait(&event, &entry, &cm_entry));

  if (!cm_entry || event != FI_CONNECTED) {
    set_custom_error("unexpected event %u - expected FI_CONNECTED (%u)", event, FI_CONNECTED);
    _errno = -FI_ECONNREFUSED;
    return false;
  }
  
  _state = EP_CONNECTED;

  return true;
}

bool Endpoint::accept(struct fi_info* remote_info)
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  if (!check_connection_state())
    return false;

  CHECK_ERR(fi_endpoint(_fabric->domain(), remote_info, &_ep, NULL), "fi_endpoint");
  CHECK_ERR(fi_ep_bind(_ep, &_eq->fid, 0), "fi_ep_bind(eq)");
  CHECK_ERR(fi_ep_bind(_ep, &_cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK_ERR(fi_accept(_ep, NULL, 0), "fi_accept");

  CHECK(event_wait(&event, &entry, &cm_entry));

  if (!cm_entry || event != FI_CONNECTED) {
    set_custom_error("unexpected event %u - expected FI_CONNECTED (%u)", event, FI_CONNECTED);
    _errno = -FI_ECONNREFUSED;
    return false;
  }

  _state = EP_CONNECTED;

  return true;
}

bool Endpoint::handle_comp(ssize_t comp_ret, struct fi_cq_data_entry* comp, int* comp_num, const char* cmd)
{
  struct fi_cq_err_entry comp_err;

  if (comp_ret < 0) {
    *comp_num = -1;
    _errno = (int) comp_ret;
    if (_errno == -FI_EAVAIL) {
      if (comp_error(&comp_err)) {
        fi_cq_strerror(_cq, comp_err.prov_errno, comp_err.err_data, _error, ERR_MSG_LEN);
      }
    } else {
      set_error(cmd);
    }
    return false;
  } else {
    *comp_num = (int) comp_ret;
    return true;
  }
}

bool Endpoint::comp(struct fi_cq_data_entry* comp, int* comp_num, ssize_t max_count)
{
  return handle_comp(fi_cq_read(_cq, comp, max_count), comp, comp_num, "fi_cq_read");
}

bool Endpoint::comp_wait(struct fi_cq_data_entry* comp, int* comp_num, ssize_t max_count, int timeout)
{
  return handle_comp(fi_cq_sread(_cq, comp, max_count, NULL, timeout), comp, comp_num, "fi_cq_sread");
}

bool Endpoint::comp_error(struct fi_cq_err_entry* comp_err)
{
  ssize_t rret = fi_cq_readerr(_cq, comp_err, 0);
  if (rret < 0) {
    _errno = (int) rret;
    set_error("fi_cq_readerr");
    return false;
  } else if (rret == 0) {
    set_custom_error("fi_cq_readerr: no errors to be read");
    _errno = FI_SUCCESS;
    return false;
  }

  return true;
}

bool Endpoint::recv_comp_data(void* context)
{
  ssize_t rret = fi_recv(_ep, NULL, 0, NULL, 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_recv");
    return false;
  }

  return true;
}

bool Endpoint::recv_comp_data_sync(uint64_t* data)
{
  int context = _counter++;

  if(recv_comp_data(&context)) {
    return check_completion(context, FI_REMOTE_CQ_DATA);
  }

  return false;
}

bool Endpoint::send(void* buf, size_t len, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_send");

  ssize_t rret = fi_send(_ep, buf, len, mr->desc(), 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_send");
    return false;
  }

  return true;
}

bool Endpoint::send_sync(void* buf, size_t len, const MemoryRegion* mr)
{
  int context = _counter++;

  if (send(buf, len, &context, mr)) {
    return check_completion(context, FI_SEND | FI_MSG);
  }

  return false;
}

bool Endpoint::recv(void* buf, size_t len, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_recv");

  ssize_t rret = fi_recv(_ep, buf, len, mr->desc(), 0, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_recv");
    return false;
  }

  return true;
}

bool Endpoint::recv_sync(void* buf, size_t len, const MemoryRegion* mr)
{
  int context = _counter++;

  if (recv(buf, len, &context, mr)) {
    return check_completion(context, FI_RECV | FI_MSG);
  }
  
  return false;
}

bool Endpoint::read(void* buf, size_t len, const RemoteAddress* raddr, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_read");

  ssize_t rret = fi_read(_ep, buf, len, mr->desc(), 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_read");
    return false;
  }

  return true;
}

bool Endpoint::read_sync(void* buf, size_t len, const RemoteAddress* raddr, const MemoryRegion* mr)
{
  int context = _counter++;

  if (read(buf, len, raddr, &context, mr)) {
    return check_completion(context, FI_READ | FI_RMA);
  }

  return false;
}

bool Endpoint::write(void* buf, size_t len, const RemoteAddress* raddr, void* context, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_write");

  ssize_t rret = fi_write(_ep, buf, len, mr->desc(), 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_write");
    return false;
  }

  return true;
}

bool Endpoint::write_sync(void* buf, size_t len, const RemoteAddress* raddr, const MemoryRegion* mr)
{
  int context = _counter++;

  if (write(buf, len, raddr, &context, mr)) {
    return check_completion(context, FI_WRITE | FI_RMA);
  }

  return false;
}

bool Endpoint::write_data(void* buf, size_t len, const RemoteAddress* raddr, void* context, uint64_t data, const MemoryRegion* mr)
{
  CHECK_MR(buf, len, mr, "fi_writedata");

  ssize_t rret = fi_writedata(_ep, buf, len, mr->desc(), data, 0, raddr->addr, raddr->rkey, context);
  if (rret != FI_SUCCESS) {
    _errno = (int) rret;
    set_error("fi_writedata");
    return false;
  }

  return true;
}

bool Endpoint::write_data_sync(void* buf, size_t len, const RemoteAddress* raddr, uint64_t data, const MemoryRegion* mr)
{
  int context = _counter++;

  if (write_data(buf, len, raddr, &context, data, mr)) {
    return check_completion(context, FI_WRITE | FI_RMA);
  }

  return false;
}

bool Endpoint::check_completion(int context, unsigned flags)
{
  int num_comp;
  struct fi_cq_data_entry comp;

  if (comp_wait(&comp, &num_comp, 1) && num_comp == 1) {
    if ((comp.flags & flags) == flags) {
      return (*((int*) comp.op_context) == context);
    }
  }

  return false;
}

bool Endpoint::check_connection_state()
{
  if (_state > EP_UP) {
    if (_state == EP_CONNECTED) {
      _errno = -FI_EISCONN;
      set_error("Endpoint is already in a connected state");
      _errno = -FI_EISCONN;
    } else {
      _errno = -FI_EOPNOTSUPP;
      set_error("Connecting is not possible in current state");
    }

    return false;
  }

  if (_state < EP_UP) {
    if (!EndpointBase::initialize()) {
      _errno = -FI_EOPNOTSUPP;
      set_error("Connecting is not possible in current state");
      return false;
    }
  }

  return true;
}

PassiveEndpoint::PassiveEndpoint(const char* addr, const char* port, int flags) :
  EndpointBase(addr, port, flags),
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

  EndpointBase::shutdown();
}

bool PassiveEndpoint::listen()
{
  CHECK_ERR(fi_passive_ep(_fabric->fabric(), _fabric->info(), &_pep, NULL), "fi_passive_ep");
  CHECK_ERR(fi_pep_bind(_pep, &_eq->fid, 0), "fi_pep_bind(eq)");
  CHECK_ERR(fi_listen(_pep), "fi_listen");

  _state = EP_LISTEN;

  return true;
}

Endpoint* PassiveEndpoint::accept()
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  if (_state != EP_LISTEN) {
    _errno = -FI_EOPNOTSUPP;
    set_error("Passive endpoint must be in a listening state to accept connections");
    return NULL;
  }

  if (!event_wait(&event, &entry, &cm_entry))
    return NULL;

  if (!cm_entry || event != FI_CONNREQ) {
    set_custom_error("unexpected event %u - expected FI_CONNECTED (%u)", event, FI_CONNREQ);
    _errno = -FI_ECONNABORTED;
    return NULL;
  }

  Endpoint *endp = new Endpoint(_fabric);
  if (endp->accept(entry.info)) {
    _endpoints.push_back(endp);
    return endp;
  } else {
    _errno = -FI_ECONNABORTED;
    set_error("active endpoint creation failed");
    delete endp;
    fi_reject(_pep, entry.info->handle, NULL, 0);
    return NULL;
  }
}

bool PassiveEndpoint::reject()
{
  bool cm_entry;
  struct fi_eq_cm_entry entry;
  uint32_t event;

  if (_state != EP_LISTEN) {
    _errno = -FI_EOPNOTSUPP;
    set_error("Passive endpoint must be in a listening state to reject connections");
    return false;
  }

  CHECK(event_wait(&event, &entry, &cm_entry));

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

bool CompletionPoller::add(Endpoint* endp)
{
  for (unsigned i=0; i<_nfd; i++) {
    if (_endps[i] == endp)
      return false;
  }

  check_size();

  CHECK_ERR(fi_control(&endp->cq()->fid, FI_GETWAIT, (void *) &_pfd[_nfd].fd), "fi_control");

  _pfd[_nfd].events   = POLLIN;
  _pfd[_nfd].revents  = 0;
  _pfid[_nfd]         = &endp->cq()->fid;
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

bool CompletionPoller::poll(int timeout)
{
  int npoll = 0;
  int ret = 0;

  ret = fi_trywait(_fabric->fabric(), _pfid, _nfd);
  if (ret == FI_SUCCESS) {
    npoll = ::poll(_pfd, _nfd, timeout);
    if (npoll < 0) {
      _errno = npoll;
      set_error("poll");
    } else if (npoll == 0) {
      _errno = -FI_EAGAIN;
      set_error("poll");
    } else {
      clear_error();
    }
    return (npoll > 0);
  } else if (ret == -FI_EAGAIN) {
    return true;
  } else {
    _errno = ret;
    set_error("fi_trywait");
    return false;
  }
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

#undef ERR_MSG_LEN
#undef CHECK_ERR
#undef CHECK
#undef CHECK_MR
