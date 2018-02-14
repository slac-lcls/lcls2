/*
 * Much of the following code was taken from the pingpong.c libfabric example.
 */

/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014-2016, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos Nat. Security, LLC. All rights reserved.
 * Copyright (c) 2016 Cray Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "fiTransport.hh"

//#include <config.h>

// The following is needed to get PRIu16 and friends defined in c++ files
#define __STDC_FORMAT_MACROS

#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <netdb.h>
#include <poll.h>
#include <limits.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

#define	FT_FIVERSION FI_VERSION(1, 4)

#ifdef __APPLE__
#include "osx/osd.h"
#elif defined __FreeBSD__
#include "freebsd/osd.h"
#endif

#define _MSG_CHECK_PORT_OK "port ok"
#define _MSG_LEN_PORT 5
#define _MSG_CHECK_CNT_OK "cnt ok"
#define _MSG_LEN_CNT 10
#define _MSG_SYNC_Q "q"
#define _MSG_SYNC_A "a"

#define FT_PRINTERR(call, retv)                                         \
  fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__,       \
          __LINE__, (int)retv, fi_strerror((int) -retv))

#define FT_ERR(fmt, ...)                                        \
  fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__, \
          __LINE__, ##__VA_ARGS__)

static int ft_debug;

#define FT_DEBUG(fmt, ...)                              \
  do {                                                  \
    if (ft_debug) {                                     \
      fprintf(stderr, "[%s] %s:%-4d: " fmt, "debug",    \
              __FILE__, __LINE__, ##__VA_ARGS__);       \
    }                                                   \
  } while (0)

#define FT_CLOSE_FID(fd)                        \
  do {                                          \
    int ret;                                    \
    if ((fd)) {                                 \
      ret = fi_close(&(fd)->fid);               \
      if (ret)                                  \
        FT_ERR("fi_close (%d) fid %d", ret,     \
               (int)(fd)->fid.fclass);          \
      fd = NULL;                                \
    }                                           \
  } while (0)

#ifndef MAX
#define MAX(a, b)                                                       \
  ({                                                                    \
    typeof(a) _a = (a);                                                 \
    typeof(b) _b = (b);                                                 \
    _a > _b ? _a : _b;                                                  \
    })
#endif

#ifndef MIN
#define MIN(a, b)                                                       \
  ({                                                                    \
    typeof(a) _a = (a);                                                 \
    typeof(b) _b = (b);                                                 \
    _a < _b ? _a : _b;                                                  \
  })
#endif

using namespace Pds::Eb;


FiTransport::FiTransport(uint16_t         srcPort,
                         uint16_t         dstPort,
                         char*            dstAddr,
                         enum fi_ep_type  epType,
                         uint64_t         caps,
                         uint64_t         mode,
                         char*            domain,
                         char*            provider) :
  _hints(NULL),
  _fi(NULL),
  _fi_pep(NULL),
  _fabric(NULL),
  _domain(NULL),
  _eq(NULL),
  _av(NULL),
  _rxcq(NULL),
  _txcq(NULL),
  _mr(NULL),
  _pep(NULL),
  _ep(NULL),
  _tx_seq(0),
  _rx_seq(0),
  _tx_cq_cntr(0),
  _rx_cq_cntr(0),
  _timeout_sec(-1),
  _src_port(srcPort),
  _dst_port(dstPort),
  _dst_addr(dstAddr),
  _cnt_ack_msg(0),
  _ctrl_connfd(-1),
  _error(EXIT_SUCCESS)
{
  memset(&_no_mr,   0, sizeof(_no_mr));
  memset(&_tx_ctx,  0, sizeof(_tx_ctx));
  memset(&_rx_ctx,  0, sizeof(_rx_ctx));
  memset(&_av_attr, 0, sizeof(_av_attr));
  memset(&_eq_attr, 0, sizeof(_eq_attr));
  memset(&_cq_attr, 0, sizeof(_cq_attr));

  _eq_attr.wait_obj = FI_WAIT_UNSPEC;

  _hints = fi_allocinfo();
  if (!_hints)
  {
    _error = EXIT_FAILURE;
    return;
  }

  _hints->ep_attr->type          = epType;
  _hints->caps                   = caps;
  _hints->mode                   = mode;
  _hints->domain_attr->name      = domain;
  _hints->fabric_attr->prov_name = provider;
}

FiTransport::~FiTransport()
{
  if (_ep)
    fi_shutdown(_ep, 0);

  _free_res();
}

/*******************************************************************************
 *                                         Public interface
 ******************************************************************************/

int FiTransport::start(int maxMsgSize, void* buf, size_t size)
{
  int ret;

  if (_error)
    return _error;

  switch (_hints->ep_attr->type)
  {
    case FI_EP_DGRAM:
      if (maxMsgSize != 0)
        _hints->ep_attr->max_msg_size = maxMsgSize;
      ret = _setup_dgram(buf, size);
      break;
    case FI_EP_RDM:
      ret = _setup_rdm(buf, size);
      break;
    case FI_EP_MSG:
      ret = _setup_msg(buf, size);
      break;
    default:
      fprintf(stderr, "Endpoint unsupported: %d\n", _hints->ep_attr->type);
      ret = EXIT_FAILURE;
  }

  return ret;
}

ssize_t FiTransport::postTransmit(void* buf, size_t size)
{
  ssize_t ret;

  // Revisit: Move this choice to the caller's domain?
  if (size < _fi->tx_attr->inject_size)
    ret = _inject(_ep, buf, size);
  else
    ret = _tx(_ep, buf, size);

  return ret;
}

ssize_t FiTransport::postReceive(void* buf, size_t size)
{
  return _rx(_ep, buf, size);
}

int FiTransport::finalize(void* buf, size_t size)
{
  return _finalize(buf, size);
}

/*******************************************************************************
 *                                         Accessors
 ******************************************************************************/

void FiTransport::debug(int enable)
{
  ft_debug = enable;
}

void FiTransport::clearCounters()
{
  _cnt_ack_msg = 0;
}

const struct fi_info* FiTransport::fi() const
{
  return _fi;
}

int FiTransport::ctrlSync()             // Revisit
{
  return _ctrl_sync();
}

void FiTransport::dumpFabricInfo()
{
  FT_DEBUG("Running pingpong test with the %s endpoint through a %s provider\n",
           fi_tostr(&_fi->ep_attr->type, FI_TYPE_EP_TYPE),
           _fi->fabric_attr->prov_name);
  FT_DEBUG(" * Fabric Attributes:\n");
  FT_DEBUG("  - %-20s: %s\n", "name", _fi->fabric_attr->name);
  FT_DEBUG("  - %-20s: %s\n", "prov_name",
           _fi->fabric_attr->prov_name);
  FT_DEBUG("  - %-20s: %" PRIu32 "\n", "prov_version",
           _fi->fabric_attr->prov_version);
  FT_DEBUG(" * Domain Attributes:\n");
  FT_DEBUG("  - %-20s: %s\n", "name", _fi->domain_attr->name);
  FT_DEBUG("  - %-20s: %zu\n", "cq_cnt", _fi->domain_attr->cq_cnt);
  FT_DEBUG("  - %-20s: %zu\n", "cq_data_size",
           _fi->domain_attr->cq_data_size);
  FT_DEBUG("  - %-20s: %zu\n", "ep_cnt", _fi->domain_attr->ep_cnt);
  FT_DEBUG(" * Endpoint Attributes:\n");
  FT_DEBUG("  - %-20s: %s\n", "type",
           fi_tostr(&_fi->ep_attr->type, FI_TYPE_EP_TYPE));
  FT_DEBUG("  - %-20s: %" PRIu32 "\n", "protocol",
           _fi->ep_attr->protocol);
  FT_DEBUG("  - %-20s: %" PRIu32 "\n", "protocol_version",
           _fi->ep_attr->protocol_version);
  FT_DEBUG("  - %-20s: %zu\n", "max_msg_size",
           _fi->ep_attr->max_msg_size);
  FT_DEBUG("  - %-20s: %zu\n", "max_order_raw_size",
           _fi->ep_attr->max_order_raw_size);
}

/*******************************************************************************
 *                                         Utils
 ******************************************************************************/

static uint64_t gettime_us(void)
{
  struct timeval now;

  gettimeofday(&now, NULL);
  return now.tv_sec * 1000000 + now.tv_usec;
}

static long parse_ulong(char *str, long max)
{
  long ret;
  char *end;

  errno = 0;
  ret = strtol(str, &end, 10);
  if (*end != '\0' || errno != 0) {
    if (errno == 0)
      ret = -EINVAL;
    else
      ret = -errno;
    fprintf(stderr, "Error parsing \"%s\": %s\n", str,
            strerror(-ret));
    return ret;
  }

  if ((ret < 0) || (max > 0 && ret > max)) {
    ret = -ERANGE;
    fprintf(stderr, "Error parsing \"%s\": %s\n", str,
            strerror(-ret));
    return ret;
  }
  return ret;
}

/*******************************************************************************
 *                                         Control Messaging
 ******************************************************************************/

static int _getaddrinfo(char* name, uint16_t port, struct addrinfo** results)
{
  int ret;
  const char *err_msg;
  char port_s[6];

  struct addrinfo hints;
  hints.ai_family   = AF_INET;             /* IPv4 */
  hints.ai_socktype = SOCK_STREAM;         /* TCP socket */
  hints.ai_protocol = IPPROTO_TCP;         /* Any protocol */
  hints.ai_flags    = AI_NUMERICSERV;      /* numeric port is used */

  snprintf(port_s, 6, "%" PRIu16, port);

  ret = getaddrinfo(name, port_s, &hints, results);
  if (ret != 0)
  {
    err_msg = gai_strerror(ret);
    FT_ERR("getaddrinfo : %s", err_msg);
    ret = -EXIT_FAILURE;
    goto out;
  }
  ret = EXIT_SUCCESS;

 out:
  return ret;
}

int FiTransport::_ctrl_init_client()
{
  struct sockaddr_in in_addr = {0};
  struct addrinfo *results;
  struct addrinfo *rp;
  int errno_save;
  int ret;

  ret = _getaddrinfo(_dst_addr, _dst_port, &results);
  if (ret)
    return ret;

  if (!results)
  {
    FT_ERR("_getaddrinfo returned NULL list");
    return -EXIT_FAILURE;
  }

  for (rp = results; rp; rp = rp->ai_next)
  {
    _ctrl_connfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (_ctrl_connfd == -1)
    {
      errno_save = errno;
      continue;
    }

    if (_src_port != 0) // Revisit: Is this _src_port and not _dst_port correct?
    {                   //          _src_port is usually 0 on the client side
      in_addr.sin_family      = AF_INET;
      in_addr.sin_port        = htons(_src_port);
      in_addr.sin_addr.s_addr = htonl(INADDR_ANY);

      ret = bind(_ctrl_connfd, (struct sockaddr *)&in_addr, sizeof(in_addr));
      if (ret == -1)
      {
        errno_save = errno;
        close(_ctrl_connfd);
        continue;
      }
    }

    ret = connect(_ctrl_connfd, rp->ai_addr, rp->ai_addrlen);
    if (ret != -1)
      break;

    errno_save = errno;
    close(_ctrl_connfd);
  }

  if (!rp || ret == -1)
  {
    ret = -errno_save;
    _ctrl_connfd = -1;
    FT_ERR("Failed to connect: %s", strerror(errno_save));
  }
  else
  {
    FT_DEBUG("CLIENT: connected\n");
  }

  freeaddrinfo(results);

  return ret;
}

int FiTransport::_ctrl_init_server()
{
  struct sockaddr_in ctrl_addr = {0};
  int optval = 1;
  int listenfd;
  int ret;

  listenfd = socket(AF_INET, SOCK_STREAM, 0);
  if (listenfd == -1)
  {
    ret = -errno;
    FT_PRINTERR("socket", ret);
    return ret;
  }

  ret = setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
  if (ret == -1)
  {
    ret = -errno;
    FT_PRINTERR("setsockopt(SO_REUSEADDR)", ret);
    goto fail_close_socket;
  }

  ctrl_addr.sin_family      = AF_INET;
  ctrl_addr.sin_port        = htons(_src_port);
  ctrl_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  ret = bind(listenfd, (struct sockaddr *)&ctrl_addr, sizeof(ctrl_addr));
  if (ret == -1)
  {
    ret = -errno;
    FT_PRINTERR("bind", ret);
    goto fail_close_socket;
  }

  ret = listen(listenfd, 10);
  if (ret == -1)
  {
    ret = -errno;
    FT_PRINTERR("listen", ret);
    goto fail_close_socket;
  }

  FT_DEBUG("SERVER: waiting for connection\n");

  _ctrl_connfd = accept(listenfd, NULL, NULL);
  if (_ctrl_connfd == -1)
  {
    ret = -errno;
    FT_PRINTERR("accept", ret);
    goto fail_close_socket;
  }

  close(listenfd);

  FT_DEBUG("SERVER: connected\n");

  return ret;

 fail_close_socket:
  if (_ctrl_connfd != -1)
  {
    close(_ctrl_connfd);
    _ctrl_connfd = -1;
  }

  if (listenfd != -1)
    close(listenfd);

  return ret;
}

int FiTransport::_ctrl_init()
{
  const uint32_t default_ctrl = 47592;  // Revist: Good value?
  struct timeval tv =
  {
    .tv_sec = 5
  };
  int ret;

  FT_DEBUG("Initializing control messages\n");

  if (_dst_addr)
  {
    if (_dst_port == 0)
      _dst_port = default_ctrl;
    ret = _ctrl_init_client();
  }
  else
  {
    if (_src_port == 0)
      _src_port = default_ctrl;
    ret = _ctrl_init_server();
  }

  if (ret)
    return ret;

  ret = setsockopt(_ctrl_connfd, SOL_SOCKET, SO_RCVTIMEO, &tv,
                   sizeof(struct timeval));
  if (ret == -1)
  {
    ret = -errno;
    FT_PRINTERR("setsockopt(SO_RCVTIMEO)", ret);
    return ret;
  }

  FT_DEBUG("Control messages initialized\n");

  return ret;
}

int FiTransport::_ctrl_send(char *buf, size_t size)
{
  int ret, err;

  ret = send(_ctrl_connfd, buf, size, 0);
  if (ret < 0)
  {
    err = -errno;
    FT_PRINTERR("ctrl/send", err);
    return err;
  }
  if (ret == 0)
  {
    err = -ECONNABORTED;
    FT_ERR("ctrl/read: no data or remote connection closed");
    return err;
  }

  return ret;
}

int FiTransport::_ctrl_recv(char *buf, size_t size)
{
  int ret, err;

  do
  {
    FT_DEBUG("ctrl/read: receiving\n");
    ret = recv(_ctrl_connfd, buf, size, 0);
  }
  while (ret == -1 && errno == EAGAIN);
  if (ret < 0)
  {
    err = -errno;
    FT_PRINTERR("ctrl/read", err);
    return err;
  }
  if (ret == 0)
  {
    err = -ECONNABORTED;
    FT_ERR("ctrl/read: no data or remote connection closed");
    return err;
  }

  return ret;
}

int FiTransport::_send_name(struct fid *endpoint)
{
  char local_name[64];
  size_t addrlen;
  uint32_t len;
  int ret;

  FT_DEBUG("Fetching local address\n");

  addrlen = sizeof(local_name);
  ret = fi_getname(endpoint, local_name, &addrlen);
  if (ret)
  {
    FT_PRINTERR("fi_getname", ret);
    return ret;
  }

  if (addrlen > sizeof(local_name))
  {
    FT_DEBUG("Address exceeds control buffer length\n");
    return -EMSGSIZE;
  }

  FT_DEBUG("Sending name length\n");
  len = htonl(addrlen);
  ret = _ctrl_send((char *) &len, sizeof(len));
  if (ret < 0)
    return ret;

  FT_DEBUG("Sending name\n");
  ret = _ctrl_send(local_name, addrlen);
  FT_DEBUG("Sent name\n");

  return ret;
}

int FiTransport::_recv_name()
{
  uint32_t len;
  int ret;

  FT_DEBUG("Receiving name length\n");
  ret = _ctrl_recv((char *) &len, sizeof(len));
  if (ret < 0)
    return ret;

  len = ntohl(len);

  if (len > sizeof(_rem_name))
  {
    FT_DEBUG("Address length exceeds address storage\n");
    return -EMSGSIZE;
  }

  FT_DEBUG("Receiving name\n");
  ret = _ctrl_recv(_rem_name, len);
  if (ret < 0)
    return ret;
  FT_DEBUG("Received name\n");

  _hints->dest_addr = malloc(len);
  if (!_hints->dest_addr)
  {
    FT_DEBUG("Failed to allocate memory for destination address\n");
    return -ENOMEM;
  }

  /* fi_freeinfo will free the dest_addr field. */
  memcpy(_hints->dest_addr, _rem_name, len);
  _hints->dest_addrlen = len;

  return 0;
}

int FiTransport::_ctrl_finish()
{
  if (_ctrl_connfd != -1)
  {
    close(_ctrl_connfd);
    _ctrl_connfd = -1;
  }

  return 0;
}

int FiTransport::_ctrl_sync()
{
  int ret;

  FT_DEBUG("Syncing nodes\n");

  if (_dst_addr)
  {
    snprintf(_ctrl_buf, sizeof(_MSG_SYNC_Q), "%s", _MSG_SYNC_Q);

    FT_DEBUG("CLIENT: syncing\n");
    ret = _ctrl_send(_ctrl_buf, sizeof(_MSG_SYNC_Q));
    FT_DEBUG("CLIENT: after send / ret=%d\n", ret);
    if (ret < 0)
      return ret;
    if (ret < (int)sizeof(_MSG_SYNC_Q))
    {
      FT_ERR("CLIENT: bad length of sent data (len=%d/%zu)",
             ret, sizeof(_MSG_SYNC_Q));
      return -EBADMSG;
    }
    FT_DEBUG("CLIENT: syncing now\n");

    ret = _ctrl_recv(_ctrl_buf, sizeof(_MSG_SYNC_A));
    FT_DEBUG("CLIENT: after recv / ret=%d\n", ret);
    if (ret < 0)
      return ret;
    if (strcmp(_ctrl_buf, _MSG_SYNC_A))
    {
      _ctrl_buf[_CTRL_BUF_LEN] = '\0';
      FT_DEBUG("CLIENT: sync error while acking A: <%s> (len=%zu)\n",
               _ctrl_buf, strlen(_ctrl_buf));
      return -EBADMSG;
    }
    FT_DEBUG("CLIENT: synced\n");
  }
  else
  {
    FT_DEBUG("SERVER: syncing\n");
    ret = _ctrl_recv(_ctrl_buf, sizeof(_MSG_SYNC_Q));
    FT_DEBUG("SERVER: after recv / ret=%d\n", ret);
    if (ret < 0)
      return ret;
    if (strcmp(_ctrl_buf, _MSG_SYNC_Q))
    {
      _ctrl_buf[_CTRL_BUF_LEN] = '\0';
      FT_DEBUG("SERVER: sync error while acking Q: <%s> (len=%zu)\n",
               _ctrl_buf, strlen(_ctrl_buf));
      return -EBADMSG;
    }

    FT_DEBUG("SERVER: syncing now\n");
    snprintf(_ctrl_buf, sizeof(_MSG_SYNC_A), "%s", _MSG_SYNC_A);

    ret = _ctrl_send(_ctrl_buf, sizeof(_MSG_SYNC_A));
    FT_DEBUG("SERVER: after send / ret=%d\n", ret);
    if (ret < 0)
      return ret;
    if (ret < (int)sizeof(_MSG_SYNC_A))
    {
      FT_ERR("SERVER: bad length of sent data (len=%d/%zu)",
             ret, sizeof(_MSG_SYNC_A));
      return -EBADMSG;
    }
    FT_DEBUG("SERVER: synced\n");
  }

  FT_DEBUG("Nodes synced\n");

  return 0;
}

int FiTransport::ctrl_txrx_msg_count()  // Revisit
{
  int ret;

  FT_DEBUG("Exchanging ack count\n");

  if (_dst_addr)
  {
    memset(&_ctrl_buf, '\0', _MSG_LEN_CNT + 1);
    snprintf(_ctrl_buf, _MSG_LEN_CNT + 1, "%ld", _cnt_ack_msg);

    FT_DEBUG("CLIENT: sending count = <%s> (len=%zu)\n",
             _ctrl_buf, strlen(_ctrl_buf));
    ret = _ctrl_send(_ctrl_buf, _MSG_LEN_CNT);
    if (ret < 0)
      return ret;
    if (ret < _MSG_LEN_CNT)
    {
      FT_ERR("CLIENT: bad length of sent data (len=%d/%d)", ret, _MSG_LEN_CNT);
      return -EBADMSG;
    }
    FT_DEBUG("CLIENT: sent count\n");

    ret = _ctrl_recv(_ctrl_buf, sizeof(_MSG_CHECK_CNT_OK));
    if (ret < 0)
      return ret;
    if (ret < (int)sizeof(_MSG_CHECK_CNT_OK))
    {
      FT_ERR("CLIENT: bad length of received data (len=%d/%zu)",
             ret, sizeof(_MSG_CHECK_CNT_OK));
      return -EBADMSG;
    }

    if (strcmp(_ctrl_buf, _MSG_CHECK_CNT_OK))
    {
      FT_DEBUG("CLIENT: error while server acking the count: <%s> (len=%zu)\n",
               _ctrl_buf, strlen(_ctrl_buf));
      return ret;
    }
    FT_DEBUG("CLIENT: count acked by server\n");
  }
  else
  {
    memset(&_ctrl_buf, '\0', _MSG_LEN_CNT + 1);

    FT_DEBUG("SERVER: receiving count\n");
    ret = _ctrl_recv(_ctrl_buf, _MSG_LEN_CNT);
    if (ret < 0)
      return ret;
    if (ret < _MSG_LEN_CNT)
    {
      FT_ERR("SERVER: bad length of received data (len=%d/%d)",
             ret, _MSG_LEN_CNT);
      return -EBADMSG;
    }
    _cnt_ack_msg = parse_ulong(_ctrl_buf, -1);
    if (_cnt_ack_msg < 0)
      return ret;
    FT_DEBUG("SERVER: received count = <%ld> (len=%zu)\n",
             _cnt_ack_msg, strlen(_ctrl_buf));

    snprintf(_ctrl_buf, sizeof(_MSG_CHECK_CNT_OK), "%s", _MSG_CHECK_CNT_OK);
    ret = _ctrl_send(_ctrl_buf, sizeof(_MSG_CHECK_CNT_OK));
    if (ret < 0)
      return ret;
    if (ret < (int)sizeof(_MSG_CHECK_CNT_OK))
    {
      FT_ERR("SERVER: bad length of sent data (len=%d/%zu)",
             ret, sizeof(_MSG_CHECK_CNT_OK));
      return -EBADMSG;
    }
    FT_DEBUG("SERVER: acked count to client\n");
  }

  FT_DEBUG("Ack count exchanged\n");

  return 0;
}

/*******************************************************************************
 *                                         Error handling
 ******************************************************************************/

static void _eq_readerr(struct fid_eq *eq)
{
  struct fi_eq_err_entry eq_err;
  int rd;

  rd = fi_eq_readerr(eq, &eq_err, 0);
  if (rd != sizeof(eq_err))
  {
    FT_PRINTERR("fi_eq_readerr", rd);
  }
  else
  {
    FT_ERR("eq_readerr: %s",
           fi_eq_strerror(eq, eq_err.prov_errno, eq_err.err_data, NULL, 0));
  }
}

static void _process_eq_err(ssize_t rd, struct fid_eq *eq, const char *fn)
{
  if (rd == -FI_EAVAIL)
    _eq_readerr(eq);
  else
    FT_PRINTERR(fn, rd);
}

/*******************************************************************************
 *                                      Data Messaging
 ******************************************************************************/

static int _cq_readerr(struct fid_cq *cq)
{
  struct fi_cq_err_entry cq_err;
  int ret;

  ret = fi_cq_readerr(cq, &cq_err, 0);
  if (ret < 0)
  {
    FT_PRINTERR("fi_cq_readerr", ret);
  }
  else
  {
    FT_ERR("cq_readerr: %s",
           fi_cq_strerror(cq, cq_err.prov_errno, cq_err.err_data, NULL, 0));
    ret = -cq_err.err;
  }
  return ret;
}

static int _get_cq_comp(struct fid_cq *cq, uint64_t *cur, uint64_t total, unsigned timeout_sec)
{
  struct fi_cq_err_entry comp;
  uint64_t a = 0, b = 0;
  int ret = 0;

  if (timeout_sec >= 0)
    a = gettime_us();

  while (total - *cur > 0)
  {
    ret = fi_cq_read(cq, &comp, 1);
    if (ret > 0)
    {
      if (timeout_sec >= 0)
        a = gettime_us();

      (*cur)++;
    }
    else if (ret < 0 && ret != -FI_EAGAIN)
    {
      if (ret == -FI_EAVAIL)
      {
        ret = _cq_readerr(cq);
        (*cur)++;
      }
      else
      {
        FT_PRINTERR("_get_cq_comp", ret);
      }

      return ret;
    }
    else if (timeout_sec >= 0)
    {
      b = gettime_us();
      if ((b - a) / 1000000 > timeout_sec)
      {
        fprintf(stderr, "%ds timeout expired\n", timeout_sec);
        return -FI_ENODATA;
      }
    }
  }

  return 0;
}

int FiTransport::_get_rx_comp(uint64_t total)
{
  int ret = FI_SUCCESS;

  if (_rxcq)
  {
    ret = _get_cq_comp(_rxcq, &_rx_cq_cntr, total, _timeout_sec);
  }
  else
  {
    FT_ERR("Trying to get a RX completion when no RX CQ was opened");
    ret = -FI_EOTHER;
  }
  return ret;
}

int FiTransport::_get_tx_comp(uint64_t total)
{
  int ret;

  if (_txcq)
  {
    ret = _get_cq_comp(_txcq, &_tx_cq_cntr, total, -1);
  }
  else
  {
    FT_ERR("Trying to get a TX completion when no TX CQ was opened");
    ret = -FI_EOTHER;
  }
  return ret;
}

#define FT_POST(post_fn, comp_fn, seq, op_str, ...)                     \
  do {                                                                  \
    unsigned timeout_sec_save;                                          \
    int ret, rc;                                                        \
                                                                        \
    while (1)                                                           \
    {                                                                   \
      ret = post_fn(__VA_ARGS__);                                       \
      if (!ret)                                                         \
        break;                                                          \
                                                                        \
      if (ret != -FI_EAGAIN)                                            \
      {                                                                 \
        FT_PRINTERR(op_str, ret);                                       \
        return ret;                                                     \
      }                                                                 \
                                                                        \
      timeout_sec_save = _timeout_sec;                                  \
      _timeout_sec = 0;                                                 \
      rc = comp_fn(seq);                                                \
      _timeout_sec = timeout_sec_save;                                  \
      if (rc && rc != -FI_EAGAIN)                                       \
      {                                                                 \
        FT_ERR("Failed to get " op_str " completion");                  \
        return rc;                                                      \
      }                                                                 \
    }                                                                   \
    seq++;                                                              \
  } while (0)

ssize_t FiTransport::_post_tx(struct fid_ep*     ep,
                              void*              buf,
                              size_t             size,
                              struct fi_context* ctx)
{
  FT_POST(fi_send, _get_tx_comp, _tx_seq, "transmit", ep, buf, size,
          fi_mr_desc(_mr), _remote_fi_addr, ctx);
  return 0;
}

ssize_t FiTransport::_tx(struct fid_ep *ep, void* buf, size_t size)
{
  ssize_t ret;

  ret = _post_tx(ep, buf, size, &_tx_ctx);
  if (ret)
    return ret;

  ret = _get_tx_comp(_tx_seq);

  return ret;
}

ssize_t FiTransport::_post_inject(struct fid_ep *ep, void* buf, size_t size)
{
  FT_POST(fi_inject, _get_tx_comp, _tx_seq, "inject", ep, buf, size,
          _remote_fi_addr);
  _tx_cq_cntr++;
  return 0;
}

ssize_t FiTransport::_inject(struct fid_ep *ep, void* buf, size_t size)
{
  ssize_t ret;

  ret = _post_inject(ep, buf, size);

  return ret;
}

ssize_t FiTransport::_post_rx(struct fid_ep*     ep,
                              void*              buf,
                              size_t             size,
                              struct fi_context* ctx)
{
  FT_POST(fi_recv, _get_rx_comp, _rx_seq, "receive", ep, buf,
          MAX(size, _MAX_CTRL_MSG), fi_mr_desc(_mr), 0, ctx);
  return 0;
}

ssize_t FiTransport::_rx(struct fid_ep *ep, void* buf, size_t size)
{
  ssize_t ret;

  ret = _get_rx_comp(_rx_seq);
  if (ret)
    return ret;

  /* TODO: verify CQ data, if available */

  // Revisit: I don't think the following is true.  There is no _sync(), only
  //          _ctrl_sync(), which doesn't make use of _rx().

  /* Ignore the size arg. Post a buffer large enough to handle all message
   * sizes. _sync() makes use of _rx() and gets called in tests just
   * before message size is updated. The recvs posted are always for the
   * next incoming message.
   */
  ret = _post_rx(ep, buf, size, &_rx_ctx);
  if (!ret)
    _cnt_ack_msg++;

  return ret;
}

/*******************************************************************************
 *                                Initialization and allocations
 ******************************************************************************/

static uint64_t _init_cq_data(struct fi_info *info)
{
  if (info->domain_attr->cq_data_size >= sizeof(uint64_t))
  {
    return 0x0123456789abcdefULL;
  }
  else
  {
    return 0x0123456789abcdefULL &
      ((0x1ULL << (info->domain_attr->cq_data_size * 8)) - 1);
  }
}

int FiTransport::_alloc_msgs(void* buf, size_t size)
{
  int ret;

  FT_DEBUG("max_msg_size = %zd\n", _fi->ep_attr->max_msg_size);

  _remote_cq_data = _init_cq_data(_fi);

  if (_fi->mode & FI_LOCAL_MR)
  {
    FT_DEBUG("MR starting at %p, of size %zd with key %04x\n",
             buf, size, _MR_KEY);
    ret = fi_mr_reg(_domain, buf, size,
                    FI_SEND | FI_RECV, 0, _MR_KEY, 0, &_mr, NULL);
    if (ret)
    {
      FT_PRINTERR("fi_mr_reg", ret);
      return ret;
    }
  }
  else
  {
    _mr = &_no_mr;
  }

  return 0;
}

int FiTransport::_open_fabric_res()
{
  int ret;

  FT_DEBUG("Opening fabric resources: fabric, eq & domain\n");

  ret = fi_fabric(_fi->fabric_attr, &_fabric, NULL);
  if (ret) {
    FT_PRINTERR("fi_fabric", ret);
    return ret;
  }

  ret = fi_eq_open(_fabric, &_eq_attr, &_eq, NULL);
  if (ret) {
    FT_PRINTERR("fi_eq_open", ret);
    return ret;
  }

  ret = fi_domain(_fabric, _fi, &_domain, NULL);
  if (ret) {
    FT_PRINTERR("fi_domain", ret);
    return ret;
  }

  FT_DEBUG("Fabric resources opened\n");

  return 0;
}

int FiTransport::_alloc_active_res(struct fi_info *fi, void* buf, size_t size)
{
  int ret;

  ret = _alloc_msgs(buf, size);
  if (ret)
    return ret;

  if (_cq_attr.format == FI_CQ_FORMAT_UNSPEC)
    _cq_attr.format = FI_CQ_FORMAT_CONTEXT;

  _cq_attr.wait_obj = FI_WAIT_NONE;

  _cq_attr.size = fi->tx_attr->size;
  FT_DEBUG("fi->tx_attr->size = %zd\n", fi->tx_attr->size);
  ret = fi_cq_open(_domain, &_cq_attr, &_txcq, &_txcq);
  if (ret)
  {
    FT_PRINTERR("fi_cq_open", ret);
    return ret;
  }

  _cq_attr.size = fi->rx_attr->size;
  FT_DEBUG("fi->rx_attr->size = %zd\n", fi->rx_attr->size);
  ret = fi_cq_open(_domain, &_cq_attr, &_rxcq, &_rxcq);
  if (ret)
  {
    FT_PRINTERR("fi_cq_open", ret);
    return ret;
  }

  if (fi->ep_attr->type == FI_EP_RDM ||
      fi->ep_attr->type == FI_EP_DGRAM)
  {
    if (fi->domain_attr->av_type != FI_AV_UNSPEC)
      _av_attr.type = fi->domain_attr->av_type;

    ret = fi_av_open(_domain, &_av_attr, &_av, NULL);
    if (ret)
    {
      FT_PRINTERR("fi_av_open", ret);
      return ret;
    }
  }

  ret = fi_endpoint(_domain, fi, &_ep, NULL);
  if (ret) {
    FT_PRINTERR("fi_endpoint", ret);
    return ret;
  }

  return 0;
}

int FiTransport::_getinfo(struct fi_info *hints, struct fi_info **info)
{
  uint64_t flags = 0;
  int ret;

  if (!hints->ep_attr->type)
    hints->ep_attr->type = FI_EP_DGRAM;

  ret = fi_getinfo(FT_FIVERSION, NULL, NULL, flags, hints, info);
  if (ret)
  {
    FT_PRINTERR("fi_getinfo", ret);
    return ret;
  }
  return 0;
}

#define FT_EP_BIND(ep, fd, flags)                                       \
  do                                                                    \
  {                                                                     \
    int ret;                                                            \
    if ((fd))                                                           \
    {                                                                   \
      ret = fi_ep_bind((ep), &(fd)->fid, (flags));                      \
      if (ret)                                                          \
      {                                                                 \
        FT_PRINTERR("fi_ep_bind", ret);                                 \
        return ret;                                                     \
      }                                                                 \
    }                                                                   \
  } while (0)

int FiTransport::_init_ep(void* buf, size_t size)
{
  int ret;

  FT_DEBUG("Initializing endpoint\n");

  if (_fi->ep_attr->type == FI_EP_MSG)
    FT_EP_BIND(_ep, _eq, 0);
  FT_EP_BIND(_ep, _av, 0);
  FT_EP_BIND(_ep, _txcq, FI_TRANSMIT);
  FT_EP_BIND(_ep, _rxcq, FI_RECV);

  ret = fi_enable(_ep);
  if (ret) {
    FT_PRINTERR("fi_enable", ret);
    return ret;
  }

  // Revisit: This should move to the application and not presume that there is
  // a buffer at the beginning of the memory region that should be posted
  FT_DEBUG("post_rx: buf = %p, size = %zd\n", buf, MAX(size, _MAX_CTRL_MSG));
  ret = _post_rx(_ep, buf, MAX(size, _MAX_CTRL_MSG), &_rx_ctx);
  if (ret)
    return ret;

  FT_DEBUG("Endpoint initialized\n");

  return 0;
}

static int _av_insert(struct fid_av *av, void *addr, size_t count,
                      fi_addr_t *fi_addr, uint64_t flags, void *context)
{
  int ret;

  FT_DEBUG("Connection-less endpoint: inserting new address in vector\n");

  ret = fi_av_insert(av, addr, count, fi_addr, flags, context);
  if (ret < 0)
  {
    FT_PRINTERR("fi_av_insert", ret);
    return ret;
  } else if ((unsigned)ret != count)
  {
    FT_ERR("fi_av_insert: number of addresses inserted = %d;"
           " number of addresses given = %zd\n",
           ret, count);
    return -EXIT_FAILURE;
  }

  FT_DEBUG("Connection-less endpoint: new address inserted in vector\n");

  return 0;
}

int FiTransport::_exchange_names_connected()
{
  int ret;

  FT_DEBUG("Connection-based endpoint: setting up connection\n");

  ret = _ctrl_sync();
  if (ret)
    return ret;

  if (_dst_addr)
  {
    ret = _recv_name();
    if (ret < 0)
      return ret;

    ret = _getinfo(_hints, &_fi);
    if (ret)
      return ret;
  }
  else
  {
    ret = _send_name(&_pep->fid);
    if (ret < 0)
      return ret;
  }

  return 0;
}

int FiTransport::_start_server()
{
  int ret;

  FT_DEBUG("Connected endpoint: starting server\n");

  ret = _getinfo(_hints, &_fi_pep);
  if (ret)
    return ret;

  ret = fi_fabric(_fi_pep->fabric_attr, &_fabric, NULL);
  if (ret)
  {
    FT_PRINTERR("fi_fabric", ret);
    return ret;
  }

  ret = fi_eq_open(_fabric, &_eq_attr, &_eq, NULL);
  if (ret)
  {
    FT_PRINTERR("fi_eq_open", ret);
    return ret;
  }

  ret = fi_passive_ep(_fabric, _fi_pep, &_pep, NULL);
  if (ret)
  {
    FT_PRINTERR("fi_passive_ep", ret);
    return ret;
  }

  ret = fi_pep_bind(_pep, &(_eq->fid), 0);
  if (ret)
  {
    FT_PRINTERR("fi_pep_bind", ret);
    return ret;
  }

  ret = fi_listen(_pep);
  if (ret)
  {
    FT_PRINTERR("fi_listen", ret);
    return ret;
  }

  FT_DEBUG("Connected endpoint: server started\n");

  return 0;
}

int FiTransport::_server_connect(void* buf, size_t size)
{
  struct fi_eq_cm_entry entry;
  uint32_t event;
  ssize_t rd;
  int ret;

  FT_DEBUG("Connected endpoint: connecting server\n");

  ret = _exchange_names_connected();
  if (ret)
    goto err;

  ret = _ctrl_sync();
  if (ret)
    goto err;

  /* Listen */
  rd = fi_eq_sread(_eq, &event, &entry, sizeof(entry), -1, 0);
  if (rd != sizeof(entry))
  {
    _process_eq_err(rd, _eq, "fi_eq_sread");
    return (int)rd;
  }

  _fi = entry.info;
  if (event != FI_CONNREQ)
  {
    fprintf(stderr, "Unexpected CM event %d\n", event);
    ret = -FI_EOTHER;
    goto err;
  }

  ret = fi_domain(_fabric, _fi, &_domain, NULL);
  if (ret)
  {
    FT_PRINTERR("fi_domain", ret);
    goto err;
  }

  ret = _alloc_active_res(_fi, buf, size);
  if (ret)
    goto err;

  ret = _init_ep(buf, size);
  if (ret)
    goto err;

  FT_DEBUG("accepting\n");

  ret = fi_accept(_ep, NULL, 0);
  if (ret)
  {
    FT_PRINTERR("fi_accept", ret);
    goto err;
  }

  ret = _ctrl_sync();
  if (ret)
    goto err;

  /* Accept */
  rd = fi_eq_sread(_eq, &event, &entry, sizeof(entry), -1, 0);
  if (rd != sizeof(entry))
  {
    _process_eq_err(rd, _eq, "fi_eq_sread");
    ret = (int)rd;
    goto err;
  }

  if (event != FI_CONNECTED || entry.fid != &(_ep->fid))
  {
    fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
            event, entry.fid, _ep);
    ret = -FI_EOTHER;
    goto err;
  }

  FT_DEBUG("Connected endpoint: server connected\n");

  return 0;

 err:
  fi_reject(_pep, _fi->handle, NULL, 0);
  return ret;
}

int FiTransport::_client_connect(void* buf, size_t size)
{
  struct fi_eq_cm_entry entry;
  uint32_t event;
  ssize_t rd;
  int ret;

  ret = _exchange_names_connected();
  if (ret)
    return ret;

  /* Check that the remote is still up */
  ret = _ctrl_sync();
  if (ret)
    return ret;

  ret = _open_fabric_res();
  if (ret)
    return ret;

  ret = _alloc_active_res(_fi, buf, size);
  if (ret)
    return ret;

  ret = _init_ep(buf, size);
  if (ret)
    return ret;

  ret = fi_connect(_ep, _fi->dest_addr, NULL, 0); // Revisit: Was _rem_name, which was packed into _hints
  if (ret)                                        // and _fi in _exchange_names_connected() above
  {
    FT_PRINTERR("fi_connect", ret);
    return ret;
  }

  ret = _ctrl_sync();
  if (ret)
    return ret;

  /* Connect */
  rd = fi_eq_sread(_eq, &event, &entry, sizeof(entry), -1, 0);
  if (rd != sizeof(entry))
  {
    _process_eq_err(rd, _eq, "fi_eq_sread");
    ret = (int)rd;
    return ret;
  }

  if (event != FI_CONNECTED || entry.fid != &(_ep->fid))
  {
    fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
            event, entry.fid, _ep);
    ret = -FI_EOTHER;
    return ret;
  }

  FT_DEBUG("Connected endpoint: client connected\n");

  return 0;
}

int FiTransport::_init_fabric(void* buf, size_t size)
{
  int ret;

  ret = _ctrl_init();
  if (ret)
    return ret;

  FT_DEBUG("Initializing fabric\n");

  FT_DEBUG("Connection-less endpoint: initializing address vector\n");

  if (_dst_addr)
  {
    ret = _recv_name();
    if (ret < 0)
      return ret;

    ret = _getinfo(_hints, &_fi);
    if (ret)
      return ret;

    ret = _open_fabric_res();
    if (ret)
      return ret;

    ret = _alloc_active_res(_fi, buf, size);
    if (ret)
      return ret;

    ret = _init_ep(buf, size);
    if (ret)
      return ret;

    ret = _send_name(&_ep->fid);
  }
  else
  {
    FT_DEBUG("SERVER: getinfo\n");
    ret = _getinfo(_hints, &_fi);
    if (ret)
      return ret;

    FT_DEBUG("SERVER: open fabric resources\n");
    ret = _open_fabric_res();
    if (ret)
      return ret;

    FT_DEBUG("SERVER: allocate active resource\n");
    ret = _alloc_active_res(_fi, buf, size);
    if (ret)
      return ret;

    FT_DEBUG("SERVER: initialize endpoint\n");
    ret = _init_ep(buf, size);
    if (ret)
      return ret;

    ret = _send_name(&_ep->fid);
    if (ret < 0)
      return ret;

    ret = _recv_name();
  }

  if (ret < 0)
    return ret;

  ret = _av_insert(_av, _rem_name, 1, &_remote_fi_addr, 0, NULL);
  if (ret)
    return ret;
  FT_DEBUG("Connection-less endpoint: address vector initialized\n");

  FT_DEBUG("Fabric Initialized\n");

  return 0;
}

/*******************************************************************************
 *                                Deallocations and Final
 ******************************************************************************/

void FiTransport::_free_res()
{
  FT_CLOSE_FID(_ep);
  FT_CLOSE_FID(_pep);
  if (_mr != &_no_mr)
    FT_CLOSE_FID(_mr);
  FT_CLOSE_FID(_txcq);
  FT_CLOSE_FID(_rxcq);
  FT_CLOSE_FID(_av);
  FT_CLOSE_FID(_eq);
  FT_CLOSE_FID(_domain);
  FT_CLOSE_FID(_fabric);

  if (_fi_pep)
  {
    fi_freeinfo(_fi_pep);
    _fi_pep = NULL;
  }
  if (_fi)
  {
    fi_freeinfo(_fi);
    _fi = NULL;
  }
  if (_hints)
  {
    fi_freeinfo(_hints);
    _hints = NULL;
  }
}

int FiTransport::_finalize(void* buf, size_t size)
{
  struct iovec iov;
  int ret;
  struct fi_context ctx;
  struct fi_msg msg;

  FT_DEBUG("Terminating\n");

  strncpy((char*)buf, "fin", size);
  iov.iov_base = buf;
  iov.iov_len  = strlen((char*)buf);

  memset(&msg, 0, sizeof(msg));
  msg.msg_iov   = &iov;
  msg.iov_count = 1;
  msg.addr      = _remote_fi_addr;
  msg.context   = &ctx;

  ret = fi_sendmsg(_ep, &msg, FI_INJECT | FI_TRANSMIT_COMPLETE);
  if (ret)
  {
    FT_PRINTERR("transmit", ret);
    return ret;
  }

  ret = _get_tx_comp(++_tx_seq);
  if (ret)
    return ret;

  ret = _get_rx_comp(_rx_seq);
  if (ret)
    return ret;

  ret = _ctrl_finish();
  if (ret)
    return ret;

  FT_DEBUG("Terminated\n");

  return 0;
}

int FiTransport::_setup_dgram(void* buf, size_t size)
{
  int ret;

  FT_DEBUG("Selected endpoint: DGRAM\n");

  ret = _init_fabric(buf, size);
  if (ret)
    return ret;

  // Revisit: This should move to the application

  /* Post an extra receive to avoid lacking a posted receive in the
   * finalize.
   */
  ret = fi_recv(_ep, buf, size, fi_mr_desc(_mr), 0, &_rx_ctx);
  if (ret)
    return ret;

  return ret;
}

int FiTransport::_setup_rdm(void* buf, size_t size)
{
  int ret;

  FT_DEBUG("Selected endpoint: RDM\n");

  ret = _init_fabric(buf, size);
  if (ret)
    return ret;

  return ret;
}

int FiTransport::_setup_msg(void* buf, size_t size)
{
  int ret;

  FT_DEBUG("Selected endpoint: MSG\n");

  ret = _ctrl_init();
  if (ret)
    return ret;

  if (!_dst_addr)
  {
    ret = _start_server();
    if (ret)
      return ret;
  }

  if (_dst_addr)
  {
    ret = _client_connect(buf, size);
    FT_DEBUG("CLIENT: client_connect=%s\n", ret ? "KO" : "OK");
  }
  else
  {
    ret = _server_connect(buf, size);
    FT_DEBUG("SERVER: server_connect=%s\n", ret ? "KO" : "OK");
  }

  if (ret)
    return ret;

  ret = _ctrl_sync();
  if (ret)
    return ret;

  return ret;
}
