#include <thread>
#include <cstdio>
#include <chrono>
#include <atomic>
#include <fstream>
#include <limits.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#include <zmq.h>

typedef std::chrono::microseconds us_t;

static const char*    dflt_device   = "mlx4_0";
static const unsigned dflt_ib_port  = 1;
static const unsigned dflt_period   = 1; // Seconds
static const char*    dflt_zmq_adx  = "tcp://psdev7b:55559";
static const char*    dflt_zmq_adx2 = "tcp://psdev7b:55565"; // Yuck.
static const char*    xx_counters[] = { "excessive_buffer_overrun_errors",
                                        "link_downed",
                                        "link_error_recovery",
                                        "local_link_integrity_errors",
                                        "multicast_rcv_packets",
                                        "multicast_xmit_packets",
                                        "port_rcv_constraint_errors",
                                        "port_rcv_data",
                                        "port_rcv_errors",
                                        "port_rcv_packets",
                                        "port_rcv_remote_physical_errors",
                                        "port_rcv_switch_relay_errors",
                                        "port_xmit_constraint_errors",
                                        "port_xmit_data",
                                        "port_xmit_discards",
                                        "port_xmit_packets",
                                        "port_xmit_wait",
                                        "symbol_error",
                                        "unicast_rcv_packets",
                                        "unicast_xmit_packets",
                                        "VL15_dropped" };
static const char*    hw_counters[] = { "lifespan",
                                        "num_cqovf",
                                        "rq_num_dup",
                                        "rq_num_lle",
                                        "rq_num_lpe",
                                        "rq_num_lqpoe",
                                        "rq_num_oos",
                                        "rq_num_rae",
                                        "rq_num_rire",
                                        "rq_num_rnr",
                                        "rq_num_udsdprd",
                                        "rq_num_wrfe",
                                        "sq_num_bre",
                                        "sq_num_lle",
                                        "sq_num_lpe",
                                        "sq_num_lqpoe",
                                        "sq_num_mwbe",
                                        "sq_num_oos",
                                        "sq_num_rae",
                                        "sq_num_rire",
                                        "sq_num_rnr",
                                        "sq_num_roe",
                                        "sq_num_rree",
                                        "sq_num_to",
                                        "sq_num_tree",
                                        "sq_num_wrfe" };

static std::atomic<bool> lrunning(true);
static unsigned          lverbose(0);


static long readIbCounter(const char* path,
                          const char* counter)
{
  char filePath[PATH_MAX];
  snprintf(filePath, sizeof(filePath), "%s/%s", path, counter);
  std::ifstream in(filePath);
  std::string line;
  std::getline(in, line);
  if (line.length() == 0)  return -1;
  return stol(line);
}

void process(const char*  path,
             unsigned     period,
             const char*  zmqAddr,
             const char** counters,
             size_t       numCtrs)
{
  void* context = zmq_ctx_new();
  void* socket  = zmq_socket(context, ZMQ_PUB);
  zmq_connect(socket, zmqAddr);

  char buffer[4096];
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  long*  previous = new long[numCtrs];
  for (unsigned i = 0; i < numCtrs; ++i)
    previous[i] = readIbCounter(path, counters[i]);

  auto now = std::chrono::steady_clock::now();

  while (lrunning)
  {
    std::this_thread::sleep_for(std::chrono::seconds(period));

    auto epoch = std::chrono::duration_cast<std::chrono::duration<int64_t>>(now.time_since_epoch()).count();
    int  size  = snprintf(buffer, sizeof(buffer), R"(["%s",{"time": [%ld])", hostname, epoch);
    auto then  = now;

    for (unsigned i = 0; i < numCtrs; ++i)
    {
      now            = std::chrono::steady_clock::now();
      auto   counter = readIbCounter(path, counters[i]);
      auto   dC      = counter - previous[i];
      auto   dT      = std::chrono::duration_cast<us_t>(now - then).count();
      double rate    = double(dC) / double(dT) * 1.e6;

      size += snprintf(&buffer[size], sizeof(buffer) - size, R"(, "%s": [%f])", counters[i], rate);

      previous[i] = counter;
    }

    size += snprintf(&buffer[size], sizeof(buffer) - size, "}]");
    if (lverbose)
    {
      printf("size = %d, buf = '%s'\n", size, buffer);
    }

    zmq_send(socket, buffer, size, 0);
  }

  delete [] previous;
}

void sigHandler( int signal )
{
  lrunning = false;
}

static void usage(char *name, char *desc)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n", "-D <device>",
          "Infiniband device",                 dflt_device);
  fprintf(stderr, " %-20s %s (default: %d)\n", "-P <port>",
          "Infiniband device port number",     dflt_ib_port);
  fprintf(stderr, " %-20s %s (default: %d)\n", "-m <seconds>",
          "Monitoring cycle period",           dflt_period);
  fprintf(stderr, " %-20s %s (default: %s)\n", "-Z <address>",
          "ZMQ server address",                dflt_zmq_adx);
  fprintf(stderr, " %-20s %s (default: %s)\n", "-c",
          "Handle IB port counters",           "hw_counters");

  fprintf(stderr, " %-20s %s\n",               "-h",
          "Display this help output");
  fprintf(stderr, " %-20s %s\n",               "-v",
          "Enable debugging output (repeat for increased detail)");
}

int main(int argc, char **argv)
{
  int         op;
  const char* device = dflt_device;
  unsigned    ibPort = dflt_ib_port;
  unsigned    period = dflt_period;
  const char* zmqAdx = dflt_zmq_adx;
  bool        hw     = true;

  while ((op = getopt(argc, argv, "h?vD:P:m:Z:c")) != -1)
  {
    switch (op)
    {
      case 'D':  device = optarg;         break;
      case 'P':  ibPort = atoi(optarg);   break;
      case 'm':  period = atoi(optarg);   break;
      case 'Z':  zmqAdx = optarg;         break;
      case 'c':  hw     = false;          break;
      case 'v':  ++lverbose;              break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Monitor Infiniband counters");
        return 1;
    }
  }

  char path[PATH_MAX];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/%s",
           device, ibPort, hw ? "hw_counters" : "counters");

  if (hw && (zmqAdx == dflt_zmq_adx))  zmqAdx = dflt_zmq_adx2; // Yuck.

  ::signal(SIGINT, sigHandler);

  size_t       numCtrs  = hw ? (sizeof(hw_counters) / sizeof(hw_counters[0]))
                             : (sizeof(xx_counters) / sizeof(xx_counters[0]));
  const char** counters = hw ? hw_counters : xx_counters;

  std::thread monThread(process,
                        std::ref(path),
                        std::ref(period),
                        std::ref(zmqAdx),
                        std::ref(counters),
                        std::ref(numCtrs));

  monThread.join();

  return 0;
}
