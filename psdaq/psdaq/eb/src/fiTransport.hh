/**
 * @file
 * @brief FiTransport, a point-to-point libfabric transport adapted from the libfabric pingpong example (see the license block below).
 */
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

    /** Point-to-point libfabric transport for DGRAM, RDM or MSG endpoints, adapted from the libfabric pingpong example per the file comment. In MSG mode a separate socket control connection is used to exchange synchronization messages. */
    class FiTransport
    {
    public:
      /** Enable (non-zero) or disable the FT_DEBUG messages; the flag is file-static, so it affects all instances. */
      static void debug(int enable);
    public:
      /** Store the ports and destination address and allocate libfabric hints filled from epType, caps, mode, mr_mode, domain and provider. If the allocation fails, start() returns EXIT_FAILURE. */
      FiTransport(uint16_t        srcPort,
                  uint16_t        dstPort,
                  char*           dstAddr,
                  enum fi_ep_type epType,
                  uint64_t        caps,
                  uint64_t        mode,
                  uint64_t        mr_mode,
                  char*           domain,
                  char*           provider);
      /** Shut down the endpoint if one is open and free all libfabric resources. */
      ~FiTransport();
    public:
      /** Set up the transport for the endpoint type given in the hints: DGRAM (applying maxMsgSize if non-zero and posting one extra receive into buffer), RDM, or MSG (control connection, server or client connect, then a sync). Returns 0 or an error code (EXIT_FAILURE for an unsupported type or a failed constructor). */
      int     start(int maxMsgSize, void* buffer, size_t size);
      /** Send size bytes from buf with fi_inject() if size is below the provider's inject size, otherwise with fi_send() followed by a wait for its transmit completion. Posts that return -FI_EAGAIN are retried. Returns 0 or an error. */
      ssize_t postTransmit(void* buf, size_t size);
      /** Wait for the completion of the previously posted receive, then post the next receive into buf (length is the larger of size and 64) and count it as an acknowledged message. Returns 0 or an error. */
      ssize_t postReceive(void* buf, size_t size);
      /** Copy the string fin into buf and send it with fi_sendmsg(FI_INJECT | FI_TRANSMIT_COMPLETE), wait for its transmit completion and for one receive completion, then close the control connection. Returns 0 or an error. */
      int     finalize(void* buf, size_t size);
    public:
      /** Reset the acknowledged-message counter. */
      void    clearCounters();
      /** Return the fi_info selected for the connection. */
      const struct fi_info* fi() const;
      /** Print fabric, domain and endpoint attributes of the selected fi_info with FT_DEBUG (printed only when debug output is enabled). */
      void    dumpFabricInfo();
      /** Exchange the sync question and answer strings over the control connection (the client sends and waits for the answer, the server waits and answers). Returns 0, a negative errno, or -EBADMSG. */
      int     ctrlSync();               // Revisit
      /** Exchange the acknowledged-message count over the control connection: the client sends its count and expects an ok reply; the server receives and stores the count and replies. Returns 0, a negative error, or in some mismatch cases the received length. */
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
