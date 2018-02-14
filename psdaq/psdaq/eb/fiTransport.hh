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

#ifndef Eb_EbClient_hh
#define Eb_EbClient_hh

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

namespace Pds {
  namespace Eb {

    class FiTransport
    {
    public:
      static void debug(int enable);
    public:
      FiTransport(uint16_t        srcPort,
                  uint16_t        dstPort,
                  char*           dstAddr,
                  enum fi_ep_type epType,
                  uint64_t        caps,
                  uint64_t        mode,
                  char*           domain,
                  char*           provider);
      ~FiTransport();
    public:
      int     start(int maxMsgSize, void* buffer, size_t size);
      ssize_t postTransmit(void* buf, size_t size);
      ssize_t postReceive(void* buf, size_t size);
      int     finalize(void* buf, size_t size);
    public:
      void    clearCounters();
      const struct fi_info* fi() const;
      void    dumpFabricInfo();
      int     ctrlSync();               // Revisit
      int     ctrl_txrx_msg_count();    // Revisit
    private:
      int     _ctrl_init_client();
      int     _ctrl_init_server();
      int     _ctrl_init();
      int     _ctrl_send(char *buf, size_t size);
      int     _ctrl_recv(char *buf, size_t size);
      int     _send_name(struct fid *endpoint);
      int     _recv_name();
      int     _ctrl_finish();
      int     _ctrl_sync();
      int     _get_rx_comp(uint64_t total);
      int     _get_tx_comp(uint64_t total);
      ssize_t _post_tx(struct fid_ep *ep, void* buf, size_t size, struct fi_context *ctx);
      ssize_t _tx(struct fid_ep *ep, void* buf, size_t size);
      ssize_t _post_inject(struct fid_ep *ep, void* buf, size_t size);
      ssize_t _inject(struct fid_ep *ep, void* buf, size_t size);
      ssize_t _post_rx(struct fid_ep* ep, void* buf, size_t size, struct fi_context* ctx);
      ssize_t _rx(struct fid_ep *ep, void* buf, size_t size);
      int     _alloc_msgs(void* buf, size_t size);
      int     _open_fabric_res();
      int     _alloc_active_res(struct fi_info *fi, void* buf, size_t size);
      int     _getinfo(struct fi_info *hints, struct fi_info **info);
      int     _init_ep(void* buf, size_t size);
      int     _exchange_names_connected();
      int     _start_server();
      int     _server_connect(void* buf, size_t size);
      int     _client_connect(void* buf, size_t size);
      int     _init_fabric(void* buf, size_t size);
      void    _free_res();
      int     _finalize(void* buf, size_t size);
      int     _setup_dgram(void* buf, size_t size);
      int     _setup_rdm(void* buf, size_t size);
      int     _setup_msg(void* buf, size_t size);
    private:
      enum { _MAX_CTRL_MSG = 64 };
      enum { _CTRL_BUF_LEN = 64 };
      enum { _MR_KEY       = 0xC0DE };  // Revisit: Belongs in .cc file?
    private:
      struct fi_info*    _hints;
      struct fi_info*    _fi;
      struct fi_info*    _fi_pep;
      struct fid_fabric* _fabric;
      struct fid_domain* _domain;
      struct fid_eq*     _eq;
      struct fid_av*     _av;
      struct fid_cq*     _rxcq;
      struct fid_cq*     _txcq;
      struct fid_mr*     _mr;
      struct fid_pep*    _pep;
      struct fid_ep*     _ep;

      struct fid_mr      _no_mr;
      struct fi_context  _tx_ctx;
      struct fi_context  _rx_ctx;
      uint64_t           _remote_cq_data;

      uint64_t           _tx_seq;
      uint64_t           _rx_seq;
      uint64_t           _tx_cq_cntr;
      uint64_t           _rx_cq_cntr;

      fi_addr_t          _remote_fi_addr;

      unsigned           _timeout_sec;

      struct fi_av_attr  _av_attr;
      struct fi_eq_attr  _eq_attr;
      struct fi_cq_attr  _cq_attr;

      uint16_t           _src_port;
      uint16_t           _dst_port;
      char*              _dst_addr;

      long               _cnt_ack_msg;

      int                _ctrl_connfd;
      char               _ctrl_buf[_CTRL_BUF_LEN + 1];
      char               _rem_name[_MAX_CTRL_MSG];

      int                _error;
    };
  };
};

#endif
