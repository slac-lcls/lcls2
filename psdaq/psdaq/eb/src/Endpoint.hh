/**
 * @file
 * @brief C++ wrappers around libfabric (OFI) objects: fabric and domain, memory regions, IO vectors, messages, event and completion queues, active and passive endpoints, and a completion poller.
 */
#ifndef Pds_Fabrics_Endpoint_hh
#define Pds_Fabrics_Endpoint_hh

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <poll.h>
#include <sys/uio.h>
#include <sys/socket.h>
#include <netdb.h>

/** Top-level namespace of the psdaq C++ code. */
namespace Pds {
  /** Thin C++ wrappers around the libfabric API. */
  namespace Fabrics {
    class CompletionQueue;
    class EventQueue;

    /** Life-cycle state of an EndpointBase, as assigned by the methods in Endpoint.cc. */
    enum State { EP_CLOSED, /**< Set by EndpointBase::shutdown() and when EndpointBase initialization fails. */ EP_INIT, /**< Initial state; also restored by Endpoint::shutdown() and PassiveEndpoint::shutdown(). */ EP_UP, /**< Set once EndpointBase::initialize() has the event and completion queues. */ EP_ENABLED, /**< Not assigned anywhere in Endpoint.cc. */ EP_LISTEN, /**< Set by PassiveEndpoint::listen(). */ EP_CONNECTED  /**< Set when a connect() or accept() handshake receives FI_CONNECTED. */ };

    /** Remote memory descriptor (address, length, key) for RMA calls; its fields are reinterpreted as a struct fi_rma_iov by rma_iov(). */
    class RemoteAddress {
    public:
      /** Construct with addr, extent and rkey all zero. */
      RemoteAddress();
      /** Construct from a remote key, a remote address and the extent (length) of the remote region. */
      RemoteAddress(uint64_t rkey, uint64_t addr, size_t extent);
      /** Return this object reinterpreted as a const struct fi_rma_iov pointer (relies on the addr, extent, rkey member order). */
      const struct fi_rma_iov* rma_iov() const;
    public:
      uint64_t addr;  ///< Remote address (the fi_rma_iov addr field).
      size_t extent;  ///< Length of the remote region (the fi_rma_iov len field).
      uint64_t rkey;  ///< Remote protection key (the fi_rma_iov key field).
    };

    /** Owns a registered libfabric memory region (fid_mr) together with its start address and length. */
    class MemoryRegion {
    public:
      /** Wrap an already registered fid_mr that covers len bytes from start; the destructor closes mr. */
      MemoryRegion(struct fid_mr* mr, void* start, size_t len);
      /** Close the fid_mr with fi_close() if it is non-null. */
      ~MemoryRegion();
      /** Return the remote key of the region, from fi_mr_key(). */
      uint64_t rkey() const;
      /** Return the local descriptor of the region, from fi_mr_desc(). */
      void* desc() const;
      /** Return the start address given at construction. */
      void* start() const;
      /** Return the length given at construction. */
      size_t length() const;
      /** Return the underlying fid_mr pointer. */
      struct fid_mr* fid() const;
      /** Return true if the range [start, start + len) lies entirely inside this region. */
      bool contains(const void* start, size_t len) const;
    private:
      struct fid_mr* _mr;
      void*          _start;
      size_t         _len;
    };

    /** Local buffer (pointer and length) with an optional MemoryRegion; its first two fields are reinterpreted as a struct iovec by iovec(). */
    class LocalAddress {
    public:
      /** Construct with a null buffer, zero length and no memory region. */
      LocalAddress();
      /** Construct from a buffer pointer, its length and an optional MemoryRegion. */
      LocalAddress(void* buf, size_t len, MemoryRegion* mr=NULL);
      /** Return the buffer pointer. */
      void* buf() const;
      /** Return the buffer length. */
      size_t len() const;
      /** Return the associated MemoryRegion, or NULL if none is set. */
      MemoryRegion* mr() const;
      /** Return this object reinterpreted as a const struct iovec pointer (relies on _buf and _len being the first members). */
      const struct iovec* iovec() const;
    public:
      void*         _buf;  ///< Buffer pointer (the iovec iov_base field).
      size_t        _len;  ///< Buffer length (the iovec iov_len field).
      MemoryRegion* _mr;  ///< Associated memory region or NULL; set by Fabric::register_memory(LocalAddress*).
    };

    /** Growable array of struct iovec entries plus one memory-region descriptor per entry, for vectored libfabric calls. */
    class LocalIOVec {
    public:
      /** Construct an empty vector (count() is 0) with capacity for count entries. */
      LocalIOVec(size_t count=1);
      /** Construct from an array of count LocalAddress objects, copying each iovec and, where a MemoryRegion is set, its descriptor. If local_addrs is NULL, count() is still count but the entries are not filled and check_mr() is false. */
      LocalIOVec(LocalAddress* local_addrs, size_t count);
      /** Construct from a vector of LocalAddress pointers, copying each iovec and, where a MemoryRegion is set, its descriptor. */
      LocalIOVec(const std::vector<LocalAddress*>& local_addrs);
      /** Free the iovec and descriptor arrays. */
      ~LocalIOVec();
      /** Set count() to 0 and set the all-descriptors-present flag; the capacity is kept. */
      void reset();
      /** Return the cached flag that says every entry has a memory-region descriptor. */
      bool check_mr();
      /** Return the number of entries in use. */
      size_t count() const;
      /** Return the iovec array. */
      const struct iovec* iovecs() const;
      /** Return the descriptor array (one entry per iovec). */
      void** desc() const;
      /** Grow the capacity by one if needed and return a pointer to the next unused iovec, incrementing count(). The descriptor of that entry is not set. */
      void* allocate();
      /** Append count entries from the LocalAddress array local_addr, copying descriptors where a MemoryRegion is set. Returns false if local_addr is NULL (the capacity may still have grown), otherwise true. */
      bool add_iovec(LocalAddress* local_addr, size_t count=1);
      /** Append one entry per LocalAddress in the vector, copying descriptors where a MemoryRegion is set; always returns true. */
      bool add_iovec(std::vector<LocalAddress*>& local_addr);
      /** Append one entry for buf and len, recording the descriptor of mr when mr is given; always returns true. */
      bool add_iovec(void* buf, size_t len, MemoryRegion* mr=NULL);
      /** Set count() to count; returns false without a change if count is not less than the current capacity. */
      bool set_count(size_t count);
      /** Overwrite entry index from local_addr, including its descriptor if it has a MemoryRegion (otherwise check_mr() becomes false). Returns false if index is not less than the capacity. */
      bool set_iovec(unsigned index, LocalAddress* local_addr);
      /** Overwrite entry index with buf and len and, if mr is given, its descriptor (otherwise check_mr() becomes false). Returns false if index is not less than the capacity. */
      bool set_iovec(unsigned index, void* buf, size_t len, MemoryRegion* mr=NULL);
      /** Set the descriptor of entry index from mr; returns false if index is not less than the capacity or mr is NULL. */
      bool set_iovec_mr(unsigned index, MemoryRegion* mr);
    private:
      void check_size(size_t count);
      void verify();
    private:
      bool          _mr_set;
      size_t        _count;
      size_t        _max;
      struct iovec* _iovs;
      void**        _mr_desc;
    };

    /** Growable array of struct fi_rma_iov entries that describe remote buffers for vectored RMA calls. */
    class RemoteIOVec {
    public:
      /** Construct an empty vector (count() is 0) with capacity for count entries. */
      RemoteIOVec(size_t count=1);
      /** Construct from an array of count RemoteAddress objects; if remote_addrs is NULL, count() is still count but the entries are not filled. */
      RemoteIOVec(RemoteAddress* remote_addrs, size_t count);
      /** Construct from a vector of RemoteAddress pointers. */
      RemoteIOVec(const std::vector<RemoteAddress*>& remote_addrs);
      /** Free the fi_rma_iov array. */
      ~RemoteIOVec();
      /** Return the number of entries in use. */
      size_t count() const;
      /** Return the fi_rma_iov array. */
      const struct fi_rma_iov* iovecs() const;
      /** Append count entries from the RemoteAddress array remote_addr. Returns false if remote_addr is NULL (the capacity may still have grown), otherwise true. */
      bool add_iovec(RemoteAddress* remote_addr, size_t count=1);
      /** Append one entry per RemoteAddress in the vector; always returns true. */
      bool add_iovec(std::vector<RemoteAddress*>& remote_addr);
      /** Append one entry built from rkey, addr and extent; always returns true. The index argument is not used. */
      bool add_iovec(unsigned index, uint64_t rkey, uint64_t addr, size_t extent);
      /** Overwrite entry index from remote_addr; returns false if index is not less than count() or remote_addr is NULL. */
      bool set_iovec(unsigned index, RemoteAddress* remote_addr);
      /** Overwrite entry index with rkey, addr and extent; returns false if index is not less than count(). */
      bool set_iovec(unsigned index, uint64_t rkey, uint64_t addr, size_t extent);
    private:
      void check_size(size_t count);
    private:
      size_t              _count;
      size_t              _max;
      struct fi_rma_iov*  _rma_iovs;
    };

    /** Owns a struct fi_msg_rma built from a LocalIOVec and a RemoteIOVec, for fi_readmsg() and fi_writemsg(). */
    class RmaMessage {
    public:
      /** Construct with no IO vectors; the fi_msg_rma is allocated but its fields are not initialized. */
      RmaMessage();
      /** Construct from the given IO vectors (each copied into the fi_msg_rma only if non-null), a context pointer and an immediate data value. */
      RmaMessage(LocalIOVec* loc_iov, RemoteIOVec* rem_iov, void* context, uint64_t data=0);
      /** Delete the owned fi_msg_rma; the IO vectors are not deleted. */
      ~RmaMessage();
      /** Return the local IO vector. */
      LocalIOVec* loc_iov() const;
      /** Return the remote IO vector. */
      RemoteIOVec* rem_iov() const;
      /** Set the local IO vector and copy its iovec array, descriptors and current count into the message; does nothing if loc_iov is NULL. */
      void loc_iov(LocalIOVec* loc_iov);
      /** Set the remote IO vector and copy its entry array and current count into the message; does nothing if rem_iov is NULL. */
      void rem_iov(RemoteIOVec* rem_iov);
      /** Return the iovec array pointer stored in the message. */
      const struct iovec* msg_iov() const;
      /** Return the iovec count stored in the message. */
      size_t iov_count() const;
      /** Return the descriptor array pointer stored in the message. */
      void** desc() const;
      /** Return the fi_rma_iov array pointer stored in the message. */
      const struct fi_rma_iov* rma_iov() const;
      /** Return the remote entry count stored in the message. */
      size_t rma_iov_count() const;
      /** Return the context pointer stored in the message. */
      void* context() const;
      /** Set the context pointer stored in the message. */
      void context(void* context);
      /** Return the immediate data value stored in the message. */
      uint64_t data() const;
      /** Set the immediate data value stored in the message. */
      void data(uint64_t data);
      /** Return the underlying struct fi_msg_rma. */
      const struct fi_msg_rma* msg() const;
    private:
      LocalIOVec*         _loc_iov;
      RemoteIOVec*        _rem_iov;
      struct fi_msg_rma*  _msg;
    };

    /** Owns a struct fi_msg built from a LocalIOVec, for fi_sendmsg() and fi_recvmsg(). */
    class Message {
    public:
      /** Construct with no IO vector; the fi_msg is allocated but its fields are not initialized. */
      Message();
      /** Construct from an IO vector (copied into the fi_msg only if non-null), a fabric address, a context pointer and an immediate data value. */
      Message(LocalIOVec* iov, fi_addr_t addr, void* context, uint64_t data=0);
      /** Delete the owned fi_msg; the IO vector is not deleted. */
      ~Message();
      /** Return the IO vector. */
      LocalIOVec* iov() const;
      /** Set the IO vector and copy its iovec array, descriptors and current count into the message; does nothing if iov is NULL. */
      void iov(LocalIOVec* iov);
      /** Return the iovec array pointer stored in the message. */
      const struct iovec* msg_iov() const;
      /** Return the descriptor array pointer stored in the message. */
      void** desc() const;
      /** Return the iovec count stored in the message. */
      size_t iov_count() const;
      /** Return the fabric address stored in the message. */
      fi_addr_t addr() const;
      /** Set the fabric address stored in the message. */
      void addr(fi_addr_t addr);
      /** Return the context pointer stored in the message. */
      void* context() const;
      /** Set the context pointer stored in the message. */
      void context(void* context);
      /** Return the immediate data value stored in the message. */
      uint64_t data() const;
      /** Set the immediate data value stored in the message. */
      void data(uint64_t data);
      /** Return the underlying struct fi_msg. */
      const struct fi_msg* msg() const;
    private:
      LocalIOVec*     _iov;
      struct fi_msg*  _msg;
    };

    /** Base class that keeps the last error number and a formatted error message (256-byte buffer). */
    class ErrorHandler {
    public:
      /** Allocate the message buffer and start with no error (FI_SUCCESS and a message built from the text None). */
      ErrorHandler();
      /** Free the message buffer. */
      virtual ~ErrorHandler();
      /** Return the last recorded error number (FI_SUCCESS if none). */
      int error_num() const;
      /** Return the last recorded error message. */
      const char* error() const;
      /** Reset the error number to FI_SUCCESS and the message to the no-error text. */
      void clear_error();
    protected:
      void set_custom_error(const char* fmt, ...);
      void set_error(const char* error_desc);
    protected:
      int   _errno;
      char* _error;
    };

    /** Owns a libfabric fi_info hints structure, preset by initialize() for FI_EP_MSG endpoints with FI_MSG and FI_RMA capabilities, FI_SOCKADDR_IN addresses and FI_RX_CQ_DATA mode. */
    class Info : public ErrorHandler {
    public:
      /** Allocate and preset the hints; ready() reports whether this succeeded. */
      Info();
      /** Allocate and preset the hints, then apply the optional kwargs entries ep_fabric, ep_domain and ep_provider as fabric name, domain name and provider name. Note that ready() is set true even if the allocation failed. */
      Info(const std::map<std::string, std::string>& kwargs);
      /** Free the hints with fi_freeinfo() if this object still owns them. */
      ~Info();
      /** Give up ownership of hints so that the destructor does not free them. */
      void take_hints();
      /** Return the ready flag set by the constructor. */
      bool ready() const;
    private:
      bool initialize();
      void shutdown();
    public:
      struct fi_info* hints;  ///< Hints for fi_getinfo(); owned by this object unless take_hints() was called.
    private:
      bool _owner;
      bool _ready;
    };

    /** Opens a libfabric fabric and domain for a node and service, and keeps the list of memory regions registered in that domain. */
    class Fabric : public ErrorHandler {
    public:
      /** Resolve node and service with getaddrinfo() and fi_getinfo(), then open the fabric and domain; up() reports success. If hints is null, a default Info is created and its hints are owned (and later freed) by this object. */
      Fabric(const char* node, const char* service, uint64_t flags=0, Info* hints=0);
      /** If up(), delete all registered memory regions, close the domain and fabric, and free the fi_info (and the hints if owned). */
      ~Fabric();
      /** Register len bytes at start for remote read/write, send and receive, then record and return the new MemoryRegion. Returns NULL if the fabric is not up or fi_mr_reg() fails (error recorded). */
      MemoryRegion* register_memory(void* start, size_t len);
      /** Register the buffer of laddr like register_memory(void*, size_t) and also store the new MemoryRegion in laddr; returns NULL on failure. */
      MemoryRegion* register_memory(LocalAddress* laddr);
      /** Remove mr from the list of registered regions and delete it (which closes it); returns false if mr is not in the list. */
      bool deregister_memory(MemoryRegion* mr);
      /** Print the registered memory regions (pointer, start, length) to stderr. */
      void list_memory() const;
      /** Return the first registered region that contains [start, start + len), or NULL. */
      MemoryRegion* lookup_memory(const void* start, size_t len) const;
      /** Return the first registered region that contains the buffer of laddr, or NULL (also when laddr is NULL). */
      MemoryRegion* lookup_memory(LocalAddress* laddr) const;
      /** For each entry of iov that has no descriptor, set one from a registered region containing it; return iov->check_mr(). */
      bool lookup_memory_iovec(LocalIOVec* iov) const;
      /** Return true if initialization succeeded and shutdown has not run. */
      bool up() const;
      /** Return true if the selected fi_info has the FI_RMA_EVENT capability. */
      bool has_rma_event_support() const;
      /** Return the domain name from the selected fi_info, or NULL if there is none. */
      const char* domain_name() const;
      /** Return the fabric name from the selected fi_info, or NULL if there is none. */
      const char* fabric_name() const;
      /** Return the provider name from the selected fi_info, or NULL if there is none. */
      const char* provider() const;
      /** Return the provider version from the selected fi_info, or 0 if there is none. */
      uint32_t version() const;
      /** Return the addrinfo list produced by getaddrinfo() during initialization. */
      struct addrinfo* addrInfo() const;
      /** Return the fi_info selected by fi_getinfo(). */
      struct fi_info* info() const;
      /** Return the opened fid_fabric. */
      struct fid_fabric* fabric() const;
      /** Return the opened fid_domain. */
      struct fid_domain* domain() const;
    private:
      bool initialize(const char* node, const char* service, uint64_t flags);
      void shutdown();
    private:
      bool                        _up;
      bool                        _hints_owner;
      struct addrinfo*            _addrInfo;
      struct fi_info*             _hints;
      struct fi_info*             _info;
      struct fid_fabric*          _fabric;
      struct fid_domain*          _domain;
      std::vector<MemoryRegion*>  _mem_regions;
    };

    /** Common base of active and passive endpoints: holds the Fabric, the event queue and the transmit and receive completion queues, creating any that are not supplied. */
    class EndpointBase : public ErrorHandler {
    protected:
      EndpointBase(const char* addr, const char* port, uint64_t flags=0, Info* hints=0);
      EndpointBase(Fabric* fabric, EventQueue* eq=0, CompletionQueue* txcq=0, CompletionQueue* rxcq=0);
      virtual ~EndpointBase();
    public:
      /** Return the current endpoint State. */
      State state() const;
      /** Return the Fabric used by this endpoint. */
      Fabric* fabric() const;
      /** Return the event queue. */
      EventQueue* eq() const;
      /** Return the transmit completion queue. */
      CompletionQueue* txcq() const;
      /** Return the receive completion queue. When none was supplied and the transmit queue was created internally, this is the same object as txcq(). */
      CompletionQueue* rxcq() const;
      /** Delete the event and completion queues that this object created and set the state to EP_CLOSED. */
      virtual void shutdown();
      /** Read one event from the event queue without blocking (EventQueue::event()); returns false, copying the queue's error, if no event was read. */
      bool event(uint32_t* event, void* entry, bool* cm_entry);
      /** Wait for one event on the event queue with the given timeout (EventQueue::event_wait()); returns false, copying the queue's error, if no event was read. */
      bool event_wait(uint32_t* event, void* entry, bool* cm_entry, int timeout=-1);
      /** Read one error entry from the event queue into entry; returns false, copying the queue's error, if none could be read. */
      bool event_error(struct fi_eq_err_entry *entry);
    protected:
      bool handle_event(ssize_t event_ret, bool* cm_entry, const char* cmd);
      bool initialize();
    protected:
      State            _state;
      const bool       _fab_owner;
      bool             _eq_owner;
      bool             _txcq_owner;
      bool             _rxcq_owner;
      Fabric*          _fabric;
      EventQueue*      _eq;
      CompletionQueue* _txcq;
      CompletionQueue* _rxcq;
    };
    /** Active (connected) libfabric endpoint with send, receive, RMA read and write calls and their blocking _sync variants. */
    class Endpoint : public EndpointBase {
    public:
      /** Create an endpoint that owns a new Fabric for addr and port and its own queues. */
      Endpoint(const char* addr, const char* port, uint64_t flags=0, Info* hints=0);
      /** Create an endpoint on an existing Fabric, using any supplied queues and creating the missing ones. */
      Endpoint(Fabric* fabric, EventQueue* eq=0, CompletionQueue* txcq=0, CompletionQueue* rxcq=0);
      /** Call shutdown() to close the libfabric endpoint; the base destructor then deletes owned queues and the fabric if owned. */
      ~Endpoint();
    public:
      /** Return the libfabric fid_ep, or NULL before connect() or accept() and after shutdown(). */
      struct fid_ep* endpoint() const;
      /** Shut down and close the fid_ep if open and set the state to EP_INIT; queues and fabric are kept so that connect() or accept() can be called again. */
      void shutdown();
      /** Create and enable the fid_ep, bind the event and completion queues, call fi_connect() to the destination address of the fabric info and wait (with timeout) for FI_CONNECTED. Zero txFlags and rxFlags default to FI_TRANSMIT and FI_RECV for internally created queues. Returns false on any failure (error recorded); on success the state is EP_CONNECTED. */
      bool connect(int timeout=-1, uint64_t txFlags=0, uint64_t rxFlags=0, void* context=NULL);
      /** Like connect(), but the fid_ep is created from remote_info (a connection request) and fi_accept() is called before waiting for FI_CONNECTED. */
      bool accept(struct fi_info* remote_info, int timeout=-1, uint64_t txFlags=0, uint64_t rxFlags=0, void* context=NULL);
      /* Asynchronous calls (raw buffer) */
      /** Post a zero-length receive (fi_recv) for remote completion data; returns the fi_recv() result (-FI_EAGAIN is not recorded as an error). */
      ssize_t recv_comp_data(void* context=NULL);
      /** Post fi_send() of len bytes from buf. If mr is NULL a registered region containing buf is looked up; -FI_EINVAL is returned if none is found or buf is outside mr. On -FI_EAGAIN the transmit queue is polled once without waiting and -FI_EAGAIN (or that poll's error) is returned. */
      ssize_t send(const void* buf, size_t len, void* context, const MemoryRegion* mr=NULL);
      /** Post fi_recv() of up to len bytes into buf, with the same memory-region lookup as send(); returns the fi_recv() result. */
      ssize_t recv(void* buf, size_t len, void* context, const MemoryRegion* mr=NULL);
      /** Post an RMA read (fi_read) of len bytes from raddr into buf, with the same memory-region lookup as send(); returns the fi_read() result. */
      ssize_t read(void* buf, size_t len, const RemoteAddress* raddr, void* context, const MemoryRegion* mr=NULL);
      /** Post an RMA write (fi_write) of len bytes from buf to raddr, with the same memory-region lookup and -FI_EAGAIN handling as send(). */
      ssize_t write(const void* buf, size_t len, const RemoteAddress* raddr, void* context, const MemoryRegion* mr=NULL);
      /** Post fi_writedata(), an RMA write of len bytes from buf to raddr that carries the immediate value data; memory-region lookup and -FI_EAGAIN handling as in send(). */
      ssize_t writedata(const void* buf, size_t len, const RemoteAddress* raddr, void* context, uint64_t data, const MemoryRegion* mr=NULL);
      /** Call fi_inject_writedata() for len bytes from buf to raddr with the immediate value data (no memory region, no context); -FI_EAGAIN handling as in send(). */
      ssize_t inject_writedata(const void* buf, size_t len, const RemoteAddress* raddr, uint64_t data);
      /** Call fi_injectdata() for len bytes from buf with the immediate value data; -FI_EAGAIN handling as in send(). */
      ssize_t injectdata(const void* buf, size_t len, uint64_t data);
      /* Asynchronous calls (LocalAddress wrapper) */
      /** Call send() with the buffer, length and memory region of laddr. */
      ssize_t send(const LocalAddress* laddr, void* context);
      /** Call recv() with the buffer, length and memory region of laddr. */
      ssize_t recv(LocalAddress* laddr, void* context);
      /** Call read() with the buffer, length and memory region of laddr. */
      ssize_t read(LocalAddress* laddr, const RemoteAddress* raddr, void* context);
      /** Call write() with the buffer, length and memory region of laddr. */
      ssize_t write(const LocalAddress* laddr, const RemoteAddress* raddr, void* context);
      /** Call writedata() with the buffer, length and memory region of laddr. */
      ssize_t writedata(const LocalAddress* laddr, const RemoteAddress* raddr, void* context, uint64_t data);
      /** Call inject_writedata() with the buffer and length of laddr. */
      ssize_t inject_writedata(const LocalAddress* laddr, const RemoteAddress* raddr, uint64_t data);
      /** Call injectdata() with the buffer and length of laddr. */
      ssize_t injectdata(const LocalAddress* laddr, uint64_t data);
      /* Vectored Asynchronous calls */
      /** Post fi_sendv() of the entries of iov. Missing descriptors are looked up among the registered regions of the fabric (-FI_EINVAL if any stay unresolved); -FI_EAGAIN handling as in send(). */
      ssize_t sendv(LocalIOVec* iov, void* context);
      /** Post fi_recvv() into the entries of iov, with the same descriptor lookup as sendv(); returns the fi_recvv() result. */
      ssize_t recvv(LocalIOVec* iov, void* context);
      /** Post fi_recvmsg() for msg with flags, after the descriptor lookup on the IO vector of msg; returns the fi_recvmsg() result. On failure the recorded error text names fi_readmsg. */
      ssize_t recvmsg(Message* msg, uint64_t flags=0);
      /** Post fi_sendmsg() for msg with flags, after the descriptor lookup on the IO vector of msg; -FI_EAGAIN handling as in send(). */
      ssize_t sendmsg(Message* msg, uint64_t flags=0);
      /** Post fi_readv() from raddr into the entries of iov, after the descriptor lookup; returns the fi_readv() result. */
      ssize_t readv(LocalIOVec* iov, const RemoteAddress* raddr, void* context);
      /** Post fi_writev() of the entries of iov to raddr, after the descriptor lookup; -FI_EAGAIN handling as in send(). */
      ssize_t writev(LocalIOVec* iov, const RemoteAddress* raddr, void* context);
      /** Post fi_readmsg() for msg with flags, after the descriptor lookup on the local IO vector of msg; returns the fi_readmsg() result. */
      ssize_t readmsg(RmaMessage* msg, uint64_t flags);
      /** Post fi_writemsg() for msg with flags, after the descriptor lookup on the local IO vector of msg; -FI_EAGAIN handling as in send(). */
      ssize_t writemsg(RmaMessage* msg, uint64_t flags);
      /* Synchronous calls (raw buffer) */
      /** Post a zero-length receive and wait (no timeout) on cq for a completion with FI_REMOTE_CQ_DATA and this call's context; its data is stored in data if data is non-null. Returns 0 on success or a negative error. */
      ssize_t recv_comp_data_sync(CompletionQueue* cq, uint64_t* data=NULL);
      /** Blocking send(): post it, then wait (no timeout) on the transmit queue for a completion with this call's context and FI_SEND | FI_MSG. Returns 0 on success or a negative error. */
      ssize_t send_sync(const void* buf, size_t len, const MemoryRegion* mr=NULL);
      /** Blocking recv(): post it, then wait (no timeout) on the receive queue for the matching completion (FI_RECV | FI_MSG). Returns 0 on success or a negative error. */
      ssize_t recv_sync(void* buf, size_t len, const MemoryRegion* mr=NULL);
      /** Blocking read(): post it, then wait (no timeout) on the receive completion queue for the matching completion (FI_READ | FI_RMA). Returns 0 on success or a negative error. */
      ssize_t read_sync(void* buf, size_t len, const RemoteAddress* raddr, const MemoryRegion* mr=NULL);
      /** Blocking write(): post it, then wait (no timeout) on the transmit queue for the matching completion (FI_WRITE | FI_RMA). Returns 0 on success or a negative error. */
      ssize_t write_sync(const void* buf, size_t len, const RemoteAddress* raddr, const MemoryRegion* mr=NULL);
      /** Blocking writedata(): post it, then wait (no timeout) on the transmit queue for the matching completion (FI_WRITE | FI_RMA). Returns 0 on success or a negative error. */
      ssize_t writedata_sync(const void* buf, size_t len, const RemoteAddress* raddr, uint64_t data, const MemoryRegion* mr=NULL);
      /* Synchronous calls (LocalAddress wrapper) */
      /** Call send_sync() with the buffer, length and memory region of laddr. */
      ssize_t send_sync(const LocalAddress* laddr);
      /** Call recv_sync() with the buffer, length and memory region of laddr. */
      ssize_t recv_sync(LocalAddress* laddr);
      /** Call read_sync() with the buffer, length and memory region of laddr. */
      ssize_t read_sync(LocalAddress* laddr, const RemoteAddress* raddr);
      /** Call write_sync() with the buffer, length and memory region of laddr. */
      ssize_t write_sync(const LocalAddress* laddr, const RemoteAddress* raddr);
      /** Call writedata_sync() with the buffer, length and memory region of laddr. */
      ssize_t writedata_sync(const LocalAddress* laddr, const RemoteAddress* raddr, uint64_t data);
      /* Vectored Synchronous calls */
      /** Blocking sendv(): wait (no timeout) on the transmit queue for the matching completion (FI_SEND | FI_RMA). */
      ssize_t sendv_sync(LocalIOVec* iov);
      /** Blocking recvv(): wait (no timeout) on the receive queue for the matching completion (FI_RECV | FI_RMA). */
      ssize_t recvv_sync(LocalIOVec* iov);
      /** Set the context of msg to a per-call value, call recvmsg() and wait (no timeout) on the receive queue for the matching completion (FI_RECV | FI_MSG). */
      ssize_t recvmsg_sync(Message* msg, uint64_t flags=0);
      /** Set the context of msg to a per-call value, call sendmsg() and wait (no timeout) on the transmit queue for the matching completion (FI_SEND | FI_MSG). */
      ssize_t sendmsg_sync(Message* msg, uint64_t flags=0);
      /** Blocking readv(): wait (no timeout) on the receive queue for the matching completion (FI_READ | FI_RMA). */
      ssize_t readv_sync(LocalIOVec* iov, const RemoteAddress* raddr);
      /** Meant as a blocking writev(), but the body calls readv() (not writev()) and then waits on the transmit queue for FI_WRITE | FI_RMA. */
      ssize_t writev_sync(LocalIOVec* iov, const RemoteAddress* raddr);
      /** Set the context of msg to a per-call value, call readmsg() and wait (no timeout) on the receive queue for the matching completion (FI_READ | FI_RMA). */
      ssize_t readmsg_sync(RmaMessage* msg, uint64_t flags=0);
      /** Set the context of msg to a per-call value, call writemsg() and wait (no timeout) on the transmit queue for the matching completion (FI_WRITE | FI_RMA). */
      ssize_t writemsg_sync(RmaMessage* msg, uint64_t flags=0);
    private:
      bool    complete_connect(int timeout);
      ssize_t post_comp_data_recv(void* context=NULL);
      ssize_t check_completion(CompletionQueue* cq, int context, unsigned flags, uint64_t* data=0, int timeout=-1);
      ssize_t check_completion_noctx(CompletionQueue* cq, unsigned flags, uint64_t* data=0, int timeout=-1);
      ssize_t check_connection_state();
    private:
      uint64_t        _counter;
      struct fid_ep*  _ep;
    };

    /** Listening libfabric endpoint that accepts or rejects connection requests and keeps the Endpoints it accepted. */
    class PassiveEndpoint : public EndpointBase {
    public:
      /** Create a passive endpoint that owns a new Fabric for addr and port (FI_SOURCE is added to flags). */
      PassiveEndpoint(const char* addr, const char* port, uint64_t flags=0, Info* hints=0);
      /** Call shutdown(), then delete all accepted endpoints. */
      ~PassiveEndpoint();
    public:
      /** Shut down every accepted endpoint, close the passive endpoint and set the state to EP_INIT; queues and fabric are kept. */
      void shutdown();
      /** Open the passive endpoint, bind it to the event queue, try to set the backlog (FI_ENOSYS is ignored) and call fi_listen(); the state becomes EP_LISTEN. Returns false on failure. */
      bool listen(int backlog=0);
      /** Like listen(int), but first binds the endpoint to the address from getaddrinfo() (fi_setname) and afterwards stores the bound port in port. */
      bool listen(int backlog, uint16_t& port);
      /** Wait (with timeout) for a connection request, create an Endpoint on this fabric with the given queues and accept it (that handshake waits without timeout). Returns the new Endpoint, which this object keeps and later deletes, or NULL on failure; a failed accept rejects the request. */
      Endpoint* accept(int timeout=-1, EventQueue* eq=0, CompletionQueue* txcq=0, uint64_t txFlags=0, CompletionQueue* rxcq=0, uint64_t rxFlags=0, void* context=NULL);
      /** Wait (with timeout) for a connection request and reject it with fi_reject(); returns false if not listening, if no request is read, or if the rejection fails. */
      bool reject(int timeout=-1);
      /** Remove endpoint from the accepted list and delete it if it is found; always returns true. */
      bool close(Endpoint* endpoint);
    private:
      int                     _flags;
      struct fid_pep*         _pep;
      std::vector<Endpoint*>  _endpoints;
    };

    /** Waits on the wait file descriptors of several completion queues with poll(). */
    class CompletionPoller : public ErrorHandler {
    public:
      /** Allocate arrays for size_hint entries; up() is false if the fabric is not up. */
      CompletionPoller(Fabric* fabric, nfds_t size_hint=1);
      /** Call shutdown() to free the arrays. */
      ~CompletionPoller();
      /** Return true if initialization succeeded and shutdown() has not been called. */
      bool up() const;
      /** Add the wait fd of cq (from fi_control FI_GETWAIT) for endp, growing the arrays if needed; returns false if endp is already present or fi_control() fails. */
      bool add(Endpoint* endp, CompletionQueue* cq);
      /** Remove the entry for endp; returns false if it is not present. */
      bool del(Endpoint* endp);
      /** Call fi_trywait() on all queues and, if it succeeds, poll() their fds with timeout. Returns -FI_EAGAIN if fi_trywait() returns it, otherwise the poll() result or the fi_trywait() error. */
      int  poll(int timeout=-1);
      /** Free the arrays if up() and mark the poller down. */
      void shutdown();
    private:
      bool initialize();
      void check_size();
    private:
      bool          _up;
      Fabric*       _fabric;
      nfds_t        _nfd;
      nfds_t        _nfd_max;
      pollfd*       _pfd;
      struct fid**  _pfid;
      Endpoint**    _endps;
    };

    /** Wraps a libfabric completion queue (fid_cq). */
    class CompletionQueue : public ErrorHandler {
    public:
      /** Open a completion queue of the given size with FI_CQ_FORMAT_DATA entries and an unspecified wait object; up() reports success. */
      CompletionQueue(Fabric* fabric, size_t size = 0);
      /** Open a completion queue with caller-supplied attributes and context; up() reports success. */
      CompletionQueue(Fabric* fabric, struct fi_cq_attr* cq_attr, void* context);
      /** Close the queue if it is up. */
      ~CompletionQueue();
      /** Return the underlying fid_cq. */
      struct fid_cq* cq() const;
      /** Read up to max_count entries without blocking (fi_cq_read()); returns the count or a negative error (-FI_EAGAIN is returned but not recorded as an error). */
      ssize_t comp(struct fi_cq_data_entry* comp, ssize_t max_count);
      /** Like comp(), but blocks in fi_cq_sread() with the given timeout. */
      ssize_t comp_wait(struct fi_cq_data_entry* comp, ssize_t max_count, int timeout=-1);
      /** Read one error entry into comp_err (fi_cq_readerr()); returns a positive count, FI_SUCCESS if there was nothing to read, or a negative error. */
      ssize_t comp_error(struct fi_cq_err_entry* comp_err);
      /** Return true if the queue was opened and not shut down. */
      bool up() const;
      /** Bind this queue to the fid_ep of ep with flags; returns false on failure. */
      bool bind(Endpoint* ep, uint64_t flags);
      /** Close the queue if it is up and mark it down. */
      void shutdown();
    public:
      /** Print the context pointer (and the int it points to), flags, length, buffer and data of a completion entry to stderr. */
      static void dump_cq_data_entry(const struct fi_cq_data_entry& comp);
      /** Print all fields of an error completion entry, including the provider error string and a hex dump of err_data, to stderr. */
      void comp_error_dump(const struct fi_cq_err_entry& comp_err) const;
    private:
      bool initialize(struct fi_cq_attr* cq_attr, void* context);
    private:
      friend Endpoint;
      ssize_t handle_comp(ssize_t comp_ret, struct fi_cq_data_entry* comp, const char* cmd);
      ssize_t check_completion(int context, unsigned flags, uint64_t* data=0, int timeout=-1);
      ssize_t check_completion_noctx(unsigned flags, uint64_t* data=0, int timeout=-1);
    private:
      bool           _up;
      Fabric*        _fabric;
      struct fid_cq* _cq;
    };

    /** Wraps a libfabric event queue (fid_eq). */
    class EventQueue : public ErrorHandler {
    public:
      /** Open an event queue of the given size with an unspecified wait object; up() reports success. */
      EventQueue(Fabric* fabric, size_t size = 0);
      /** Open an event queue with caller-supplied attributes and context; up() reports success. */
      EventQueue(Fabric* fabric, struct fi_eq_attr* eq_attr, void* context);
      /** Close the queue if it is up. */
      ~EventQueue();
      /** Return the underlying fid_eq. */
      struct fid_eq* eq() const;
      /** Read one event without blocking (fi_eq_read()) into event and entry; cm_entry is set true for an fi_eq_cm_entry sized event and false for an fi_eq_entry sized one. Returns false if nothing was read or on error (for -FI_EAVAIL the error number comes from the queued error entry). */
      bool event(uint32_t* event, void* entry, bool* cm_entry);
      /** Like event(), but blocks in fi_eq_sread() with the given timeout. */
      bool event_wait(uint32_t* event, void* entry, bool* cm_entry, int timeout=-1);
      /** Zero entry and read one error entry into it (fi_eq_readerr()); returns true only if a full fi_eq_err_entry was read. */
      bool event_error(struct fi_eq_err_entry *entry);
      /** Return true if the queue was opened and not shut down. */
      bool up() const;
      /** Bind this event queue to the fid_ep of ep; returns false on failure. */
      bool bind(Endpoint* ep);
      /** Close the queue if it is up and mark it down. */
      void shutdown();
    protected:
      friend EndpointBase;
      friend Endpoint;
      bool handle_event(ssize_t event_ret, bool* cm_entry, const char* cmd);
      bool initialize(struct fi_eq_attr* eq_attr, void* context);
    private:
      bool           _up;
      Fabric*        _fabric;
      struct fid_eq* _eq;
    };
  }
}

#endif
