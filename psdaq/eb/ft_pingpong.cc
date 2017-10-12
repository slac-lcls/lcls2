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
using namespace Pds::Eb;

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

#include <rdma/fi_errno.h>

enum precision
{
  NANO  = 1,
  MICRO = 1000,
  MILLI = 1000000,
};

enum
{
  PP_OPT_ACTIVE      = 1 << 0,
  PP_OPT_ITER        = 1 << 1,
  PP_OPT_SIZE        = 1 << 2,
  PP_OPT_VERIFY_DATA = 1 << 3,
};

struct pp_opts
{
  uint16_t        src_port;
  uint16_t        dst_port;
  char*           dst_addr;
  int             iterations;
  size_t          transfer_size;
  int             sizes_enabled;
  int             options;
  char*           domain;
  char*           provider;
  enum fi_ep_type ep_type;
};

#define PP_SIZE_MAX_POWER_TWO 22
#define PP_MAX_DATA_MSG                                                        \
	((1 << PP_SIZE_MAX_POWER_TWO) + (1 << (PP_SIZE_MAX_POWER_TWO - 1)))

#define PP_STR_LEN 32

#define INTEG_SEED 7
#define PP_ENABLE_ALL (~0)
#define PP_DEFAULT_SIZE (1 << 0)

#define PP_PRINTERR(call, retv)                                                \
	fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__,        \
		__LINE__, (int)retv, fi_strerror((int) -retv))

#define PP_ERR(fmt, ...)                                                       \
	fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__,          \
		__LINE__, ##__VA_ARGS__)

int pp_debug;

#define PP_DEBUG(fmt, ...)                                                     \
	do {                                                                   \
		if (pp_debug) {                                                \
			fprintf(stderr, "[%s] %s:%-4d: " fmt, "debug",         \
				__FILE__, __LINE__, ##__VA_ARGS__);            \
		}                                                              \
	} while (0)

struct ct_pingpong
{
  void*  buf;
  void*  tx_buf;
  void*  rx_buf[2];                    // Toggle between 2 receive buffers
  size_t buf_size;
  size_t tx_size;
  size_t rx_size;

  uint64_t start, end;

  struct pp_opts opts;

  long cnt_ack_msg;

  FiTransport* ft;
};

static const char integ_alphabet[] =
	"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
/* Size does not include trailing new line */
static const int integ_alphabet_length =
	(sizeof(integ_alphabet) / sizeof(*integ_alphabet)) - 1;


/*******************************************************************************
 *                                         Utils
 ******************************************************************************/

uint64_t pp_gettime_us(void)
{
  struct timeval now;

  gettimeofday(&now, NULL);
  return now.tv_sec * 1000000 + now.tv_usec;
}

long parse_ulong(char *str, long max)
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

int size_to_count(int size)
{
  if (size >= (1 << 20))
    return 100;
  else if (size >= (1 << 16))
    return 1000;
  else
    return 10000;
}

void pp_banner_options(struct ct_pingpong *ct)
{
  char size_msg[50];
  char iter_msg[50];
  struct pp_opts opts = ct->opts;

  if (opts.sizes_enabled == PP_ENABLE_ALL)
    snprintf(size_msg, 50, "%s", "All sizes");
  else if (opts.options & PP_OPT_SIZE)
    snprintf(size_msg, 50, "selected size = %zu", opts.transfer_size);

  if (opts.options & PP_OPT_ITER)
    snprintf(iter_msg, 50, "selected iterations: %d", opts.iterations);
  else {
    opts.iterations = size_to_count(opts.transfer_size);
    snprintf(iter_msg, 50, "default iterations: %d", opts.iterations);
  }

  PP_DEBUG(" * PingPong options:\n");
  PP_DEBUG("  - %-20s: [%" PRIu16 "]\n", "src_port", opts.src_port);
  PP_DEBUG("  - %-20s: [%s]\n", "dst_addr", ((opts.dst_addr == NULL) ||
                                             (opts.dst_addr[0] == '\0')) ? "None" : opts.dst_addr);
  PP_DEBUG("  - %-20s: [%" PRIu16 "]\n", "dst_port", opts.dst_port);
  PP_DEBUG("  - %-20s: %s\n", "sizes_enabled", size_msg);
  PP_DEBUG("  - %-20s: %s\n", "iterations", iter_msg);
  if (ct->opts.provider)
    PP_DEBUG("  - %-20s: %s\n", "provider", ct->opts.provider);
  if (ct->opts.domain)
    PP_DEBUG("  - %-20s: %s\n", "domain", ct->opts.domain);
}

/*******************************************************************************
 *                                         Options
 ******************************************************************************/

static inline void pp_start(struct ct_pingpong *ct)
{
  PP_DEBUG("Starting test chrono\n");
  ct->opts.options |= PP_OPT_ACTIVE;
  ct->start = pp_gettime_us();
}

static inline void pp_stop(struct ct_pingpong *ct)
{
  ct->end = pp_gettime_us();
  ct->opts.options &= ~PP_OPT_ACTIVE;
  PP_DEBUG("Stopped test chrono\n");
}

static inline int pp_check_opts(struct ct_pingpong *ct, uint64_t flags)
{
  return (ct->opts.options & flags) == flags;
}

/*******************************************************************************
 *                                         Data Verification
 ******************************************************************************/

void pp_fill_buf(void *buf, int size)
{
  char *msg_buf;
  int msg_index;
  static unsigned int iter;
  int i;

  msg_index = ((iter++) * INTEG_SEED) % integ_alphabet_length;
  msg_buf = (char *)buf;
  for (i = 0; i < size; i++) {
    PP_DEBUG("index=%d msg_index=%d\n", i, msg_index);
    msg_buf[i] = integ_alphabet[msg_index++];
    if (msg_index >= integ_alphabet_length)
      msg_index = 0;
  }
}

int pp_check_buf(void *buf, int size)
{
  char *recv_data;
  char c;
  static unsigned int iter;
  int msg_index;
  int i;

  PP_DEBUG("Verifying buffer content\n");

  msg_index = ((iter++) * INTEG_SEED) % integ_alphabet_length;
  recv_data = (char *)buf;

  for (i = 0; i < size; i++) {
    c = integ_alphabet[msg_index++];
    if (msg_index >= integ_alphabet_length)
      msg_index = 0;
    if (c != recv_data[i]) {
      PP_DEBUG("index=%d msg_index=%d expected=%d got=%d\n",
               i, msg_index, c, recv_data[i]);
      break;
    }
  }
  if (i != size) {
    PP_DEBUG("Finished veryfing buffer: content is corrupted\n");
    printf("Error at iteration=%d size=%d byte=%d\n", iter, size,
           i);
    return 1;
  }

  PP_DEBUG("Buffer verified\n");

  return 0;
}

/*******************************************************************************
 *                                         Test sizes
 ******************************************************************************/

int generate_test_sizes(struct pp_opts *opts, size_t tx_size, size_t **sizes_)
{
  size_t defaults[6] = {64, 256, 1024, 4096, 65536, 1048576};
  size_t power_of_two;
  size_t half_up;
  int n = 0;
  unsigned i;
  size_t *sizes = NULL;

  PP_DEBUG("Generating test sizes\n");

  sizes = (size_t*)calloc(64, sizeof(*sizes));
  if (sizes == NULL)
    return 0;
  *sizes_ = sizes;

  if (opts->options & PP_OPT_SIZE) {
    if (opts->transfer_size > tx_size)
      return 0;

    sizes[0] = opts->transfer_size;
    n = 1;
  } else if (opts->sizes_enabled != PP_ENABLE_ALL) {
    for (i = 0; i < (sizeof(defaults) / sizeof(defaults[0])); i++) {
      if (defaults[i] > tx_size)
        break;

      sizes[i] = defaults[i];
      n++;
    }
  } else {
    for (i = 0;; i++) {
      power_of_two = (i == 0) ? 0 : (1 << i);
      half_up =
        (i == 0) ? 1 : power_of_two + (power_of_two / 2);

      if (power_of_two > tx_size)
        break;

      sizes[i * 2] = power_of_two;
      n++;

      if (half_up > tx_size)
        break;

      sizes[(i * 2) + 1] = half_up;
      n++;
    }
  }

  PP_DEBUG("Generated %d test sizes\n", n);

  return n;
}

/*******************************************************************************
 *                                    Performance output
 ******************************************************************************/

/* str must be an allocated buffer of PP_STR_LEN bytes */
char *size_str(char *str, uint64_t size)
{
  uint64_t base, fraction = 0;
  char mag;

  memset(str, '\0', PP_STR_LEN);

  if (size >= (1 << 30)) {
    base = 1 << 30;
    mag = 'g';
  } else if (size >= (1 << 20)) {
    base = 1 << 20;
    mag = 'm';
  } else if (size >= (1 << 10)) {
    base = 1 << 10;
    mag = 'k';
  } else {
    base = 1;
    mag = '\0';
  }

  if (size / base < 10)
    fraction = (size % base) * 10 / base;

  if (fraction)
    snprintf(str, PP_STR_LEN, "%" PRIu64 ".%" PRIu64 "%c",
             size / base, fraction, mag);
  else
    snprintf(str, PP_STR_LEN, "%" PRIu64 "%c", size / base, mag);

  return str;
}

/* str must be an allocated buffer of PP_STR_LEN bytes */
char *cnt_str(char *str, size_t size, uint64_t cnt)
{
  if (cnt >= 1000000000)
    snprintf(str, size, "%" PRIu64 "b", cnt / 1000000000);
  else if (cnt >= 1000000)
    snprintf(str, size, "%" PRIu64 "m", cnt / 1000000);
  else if (cnt >= 1000)
    snprintf(str, size, "%" PRIu64 "k", cnt / 1000);
  else
    snprintf(str, size, "%" PRIu64, cnt);

  return str;
}

void show_perf(char *name, int tsize, int sent, int acked,
	       uint64_t start, uint64_t end, int xfers_per_iter)
{
  static int header = 1;
  char str[PP_STR_LEN];
  int64_t elapsed = end - start;
  uint64_t bytes = (uint64_t)sent * tsize * xfers_per_iter;
  float usec_per_xfer;

  if (sent == 0)
    return;

  if (name) {
    if (header) {
      printf("%-50s%-8s%-8s%-9s%-8s%8s %10s%13s%13s\n",
             "name", "bytes", "#sent", "#ack", "total",
             "time", "MB/sec", "usec/xfer", "Mxfers/sec");
      header = 0;
    }

    printf("%-50s", name);
  } else {
    if (header) {
      printf("%-8s%-8s%-9s%-8s%8s %10s%13s%13s\n", "bytes",
             "#sent", "#ack", "total", "time", "MB/sec",
             "usec/xfer", "Mxfers/sec");
      header = 0;
    }
  }

  printf("%-8s", size_str(str, tsize));
  printf("%-8s", cnt_str(str, sizeof(str), sent));

  if (sent == acked)
    printf("=%-8s", cnt_str(str, sizeof(str), acked));
  else if (sent < acked)
    printf("-%-8s", cnt_str(str, sizeof(str), acked - sent));
  else
    printf("+%-8s", cnt_str(str, sizeof(str), sent - acked));

  printf("%-8s", size_str(str, bytes));

  usec_per_xfer = ((float)elapsed / sent / xfers_per_iter);
  printf("%8.2fs%10.2f%11.2f%11.2f\n", elapsed / 1000000.0,
         bytes / (1.0 * elapsed), usec_per_xfer, 1.0 / usec_per_xfer);
}

/*******************************************************************************
 *                                Initialization and allocations
 ******************************************************************************/

void init_test(struct ct_pingpong *ct, struct pp_opts *opts)
{
  char sstr[PP_STR_LEN];

  size_str(sstr, opts->transfer_size);
  if (!(opts->options & PP_OPT_ITER))
    opts->iterations = size_to_count(opts->transfer_size);


  ct->cnt_ack_msg = 0;
  ct->ft->clearCounters();
}

int pp_alloc_bufs(struct ct_pingpong *ct)
{
  int ret;
  long alignment = 1;

  ct->tx_size = ct->opts.options & PP_OPT_SIZE ? ct->opts.transfer_size
                                               : PP_MAX_DATA_MSG;
  //if (ct->tx_size > ct->ft->fi()->ep_attr->max_msg_size) // Revisit: No ft yet
  //  ct->tx_size = ct->ft->fi()->ep_attr->max_msg_size;
  PP_DEBUG("transfer_size = %zd, MAX_DATA_MSG = %d, tx_size = %zd\n",
           ct->opts.transfer_size, PP_MAX_DATA_MSG, ct->tx_size);
  ct->rx_size = ct->tx_size;
  ct->buf_size = ct->tx_size + 2 * ct->rx_size;

  alignment = sysconf(_SC_PAGESIZE);
  if (alignment < 0) {
    ret = -errno;
    PP_PRINTERR("sysconf", ret);
    return ret;
  }
  /* Extra alignment for the second part of the buffer */
  ct->buf_size += alignment;

  ret = posix_memalign(&(ct->buf), (size_t)alignment, ct->buf_size);
  if (ret)
  {
    PP_PRINTERR("posix_memalign", ret);
    return ret;
  }
  memset(ct->buf, 0, ct->buf_size);
  ct->rx_buf[0] = ct->buf;
  ct->rx_buf[1] = (char *)ct->buf +     ct->rx_size;
  ct->tx_buf    = (char *)ct->buf + 2 * ct->rx_size;
  ct->tx_buf    = (void *)(((uintptr_t)ct->tx_buf + alignment - 1) & ~(alignment - 1));

  PP_DEBUG("bufs allocated from %p to %p, size %zd\n",
           ct->buf, (char*)ct->buf + ct->buf_size, ct->buf_size);
  PP_DEBUG("rx[0] = %p - %p, size = %zd\n", ct->rx_buf[0], ct->rx_buf[1], ct->rx_size);
  PP_DEBUG("rx[1] = %p - %p, size = %zd\n", ct->rx_buf[1], ct->tx_buf,    ct->rx_size);
  PP_DEBUG("tx[1] = %p - %p, size = %zd\n", ct->tx_buf,    (char*)ct->tx_buf + ct->tx_size, ct->tx_size);

  return 0;
}

/*******************************************************************************
 *                                Deallocations and Final
 ******************************************************************************/

void pp_free_res(struct ct_pingpong *ct)
{
  PP_DEBUG("Freeing resources of test suite\n");

  if (ct->buf)
  {
    free(ct->buf);
    ct->buf = ct->rx_buf[0] = ct->rx_buf[1] = ct->tx_buf = NULL;
    ct->buf_size = ct->rx_size = ct->tx_size = 0;
  }

  PP_DEBUG("Resources of test suite freed\n");
}

/*******************************************************************************
 *                                CLI: Usage and Options parsing
 ******************************************************************************/

void pp_pingpong_usage(char *name, char *desc)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
  fprintf(stderr, "  %s [OPTIONS] <srv_addr>\tconnect to server\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s\n", "-B <src_port>",
          "source control port number (server: 47592, client: auto)");
  fprintf(stderr, " %-20s %s\n", "-P <dst_port>",
          "destination control port number (client: 47592)");

  fprintf(stderr, " %-20s %s\n", "-d <domain>", "domain name");
  fprintf(stderr, " %-20s %s\n", "-p <provider>",
          "specific provider name eg sockets, verbs");
  fprintf(stderr, " %-20s %s\n", "-e <ep_type>",
          "endpoint type: msg|rdm|dgram (dgram)");

  fprintf(stderr, " %-20s %s\n", "-I <number>",
          "number of iterations (1000)");
  fprintf(stderr, " %-20s %s\n", "-S <size>",
          "specific transfer size or 'all' (all)");

  fprintf(stderr, " %-20s %s\n", "-c", "enables data_integrity checks");

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
  fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output");
}

void pp_parse_opts(struct ct_pingpong *ct, int op, char *optarg)
{
  switch (op)
  {
    /* Domain */
    case 'd':
      ct->opts.domain = strdup(optarg);
      break;

      /* Provider */
    case 'p':
      /* The provider name will be checked during the fabric
       * initialization.
       */
      ct->opts.provider = strdup(optarg);
      break;

      /* Endpoint */
    case 'e':
      if (!strncasecmp("msg", optarg, 3) && (strlen(optarg) == 3))
      {
        ct->opts.ep_type = FI_EP_MSG;
      }
      else if (!strncasecmp("rdm", optarg, 3) && (strlen(optarg) == 3))
      {
        ct->opts.ep_type = FI_EP_RDM;
      }
      else if (!strncasecmp("dgram", optarg, 5) && (strlen(optarg) == 5))
      {
        ct->opts.ep_type = FI_EP_DGRAM;
      }
      else
      {
        fprintf(stderr, "Unknown endpoint : %s\n", optarg);
        exit(EXIT_FAILURE);
      }
      break;

      /* Iterations */
    case 'I':
      ct->opts.options |= PP_OPT_ITER;
      ct->opts.iterations = (int)parse_ulong(optarg, INT_MAX);
      if (ct->opts.iterations < 0)
        ct->opts.iterations = 0;
      break;

      /* Message Size */
    case 'S':
      if (!strncasecmp("all", optarg, 3) && (strlen(optarg) == 3))
      {
        ct->opts.sizes_enabled = PP_ENABLE_ALL;
      }
      else
      {
        ct->opts.options |= PP_OPT_SIZE;
        ct->opts.transfer_size = (int)parse_ulong(optarg, INT_MAX);
      }
      break;

      /* Check data */
    case 'c':
      ct->opts.options |= PP_OPT_VERIFY_DATA;
      break;

      /* Source Port */
    case 'B':
      ct->opts.src_port = parse_ulong(optarg, USHRT_MAX);
      break;

      /* Destination Port */
    case 'P':
      ct->opts.dst_port = parse_ulong(optarg, USHRT_MAX);
      break;

      /* Debug */
    case 'v':
      pp_debug = 1;
      break;
    default:
      /* let getopt handle unknown opts*/
      break;
  }
}

/*******************************************************************************
 *      PingPong core and implemenations for endpoints
 ******************************************************************************/


int pingpong(struct ct_pingpong *ct)
{
  static int j = 0;
  int ret, i;
  FiTransport* ft = ct->ft;

  ret = ft->ctrlSync();                 // Revisit: Seems not to be required
  if (ret)
    return ret;

  pp_start(ct);
  if (ct->opts.dst_addr)
  {
    for (i = 0; i < ct->opts.iterations; j++, i++)
    {
      if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE))
        pp_fill_buf((char *)ct->tx_buf, ct->opts.transfer_size);

      //PP_DEBUG("CLIENT: postTx: %d, buf    = %p, size = %zd\n", i,
      //         ct->tx_buf, ct->opts.transfer_size);
      ret = ft->postTransmit(ct->tx_buf, ct->opts.transfer_size);
      if (ret)
        return ret;

      //PP_DEBUG("CLIENT: postRx: %d, buf[%d] = %p, size = %zd\n", i, (j & 1) ^ 1,
      //         ct->rx_buf[(j & 1) ^ 1], ct->rx_size);
      ret = ft->postReceive(ct->rx_buf[(j & 1) ^ 1], ct->rx_size);
      if (ret)
        return ret;
      else
        ct->cnt_ack_msg++;

      if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE))
      {
        ret = pp_check_buf((char *)ct->rx_buf[j & 1], ct->opts.transfer_size);
        if (ret)
          return ret;
      }
    }
  }
  else
  {
    for (i = 0; i < ct->opts.iterations; j++, i++)
    {
      //PP_DEBUG("SERVER: postRx: %d, buf[%d] = %p, size = %zd\n", i, (j & 1) ^ 1,
      //         ct->rx_buf[(j & 1) ^ 1], ct->rx_size);
      ret = ft->postReceive(ct->rx_buf[(j & 1) ^ 1], ct->rx_size);
      if (ret)
        return ret;
      else
        ct->cnt_ack_msg++;

      if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE))
      {
        ret = pp_check_buf((char *)ct->rx_buf[j & 1], ct->opts.transfer_size);
        if (ret)
          return ret;
      }

      if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE))
        pp_fill_buf((char *)ct->tx_buf, ct->opts.transfer_size);

      //PP_DEBUG("SERVER: postTx: %d, buf    = %p, size = %zd\n", i,
      //         ct->tx_buf, ct->opts.transfer_size);
      ret = ft->postTransmit(ct->tx_buf, ct->opts.transfer_size);
      if (ret)
        return ret;
    }
  }
  pp_stop(ct);

  ret = ft->ctrl_txrx_msg_count();      // Revisit: Seems not to be required
  if (ret)
    return ret;

  PP_DEBUG("Results:\n");
  show_perf(NULL, ct->opts.transfer_size, ct->opts.iterations,
            ct->cnt_ack_msg, ct->start, ct->end, 2);

  return 0;
}

int run_suite_pingpong(struct ct_pingpong *ct)
{
  int i, sizes_cnt;
  int ret = 0;
  size_t *sizes = NULL;

  ret = pp_alloc_bufs(ct);
  if (ret)
    return ret;

  FiTransport::debug(pp_debug);

  FiTransport ft(ct->opts.src_port,
                 ct->opts.dst_port,
                 ct->opts.dst_addr,
                 ct->opts.ep_type,
                 FI_MSG,
                 FI_CONTEXT | FI_LOCAL_MR,
                 ct->opts.domain,
                 ct->opts.provider);

  ret = ft.start(( (ct->opts.ep_type == FI_EP_DGRAM) &&
                  !(ct->opts.options & PP_OPT_SIZE)) ? 0 : ct->opts.transfer_size,
                 ct->buf,
                 ct->buf_size);
  if (ret)
    return ret;
  ct->ft = &ft;

  if (ct->tx_size > ft.fi()->ep_attr->max_msg_size)
  {
    PP_ERR("Endpoint can't support our message size %zd, max is %zd",
           ct->tx_size, ft.fi()->ep_attr->max_msg_size);
    return EXIT_FAILURE;
  }

  // Revisit: Post a receive on the initial rx_buf

  ft.dumpFabricInfo();

  sizes_cnt = generate_test_sizes(&ct->opts, ct->tx_size, &sizes);

  PP_DEBUG("Count of sizes to test: %d\n", sizes_cnt);

  for (i = 0; i < sizes_cnt; i++) {
    ct->opts.transfer_size = sizes[i];
    init_test(ct, &(ct->opts));
    ret = pingpong(ct);
    if (ret)
      goto out;
  }

  ret = ft.finalize(ct->tx_buf, ct->tx_size);

out:
  free(sizes);
  return ret;
}

int main(int argc, char **argv)
{
  int ret, op;

  ret = EXIT_SUCCESS;

  struct ct_pingpong ct;
  memset(&ct, 0, sizeof(ct));
  ct.opts.iterations    = 1000;
  ct.opts.transfer_size = 1024;
  ct.opts.sizes_enabled = PP_DEFAULT_SIZE;

  while ((op = getopt(argc, argv, "hvd:p:e:I:S:B:P:c")) != -1)
  {
    switch (op)
    {
      default:
        pp_parse_opts(&ct, op, optarg);
        break;
      case '?':
      case 'h':
        pp_pingpong_usage(argv[0], (char*)"Ping pong client and server");
        return EXIT_FAILURE;
    }
  }

  if (optind < argc)
    ct.opts.dst_addr = argv[optind];
  else
    ct.opts.dst_addr = NULL;

  pp_banner_options(&ct);

  switch (ct.opts.ep_type)
  {
    case FI_EP_DGRAM:
    case FI_EP_RDM:
    case FI_EP_MSG:
      ret = run_suite_pingpong(&ct);
      break;
    default:
      fprintf(stderr, "Endpoint unsupported: %d\n", ct.opts.ep_type);
      ret = EXIT_FAILURE;
  }

  pp_free_res(&ct);
  return -ret;
}
