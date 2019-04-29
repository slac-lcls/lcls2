#include "StatsMonitor.hh"

#include <zmq.h>

#include <unistd.h>                     // gethostname()
#include <limits.h>
#include <chrono>

using namespace Pds::Eb;

typedef std::chrono::microseconds us_t;


StatsMonitor::StatsMonitor(const char*        hostname,
                           unsigned           basePort,
                           unsigned           partition,
                           unsigned           period,
                           unsigned           verbose) :
  _partition(partition),
  _period   (period),
  _verbose  (verbose),
  _enabled  (false),
  _running  (false),
  _then     (std::chrono::steady_clock::now()),
  _task     (nullptr)
{
  unsigned port = basePort; // + 2 * partition; // *2: 1 for forwarder.py
  snprintf(_addr, sizeof(_addr), "tcp://%s:%u", hostname, port);
  printf("Publishing statistics to %s\n", _addr);
}

StatsMonitor::~StatsMonitor()
{
  if (_task)  delete _task;
}

void StatsMonitor::startup()
{
  _running = true;
  _task    = new std::thread([&] { _routine(); });
}

void StatsMonitor::shutdown()
{
  printf("\nShutting down StatsMonitor...\n");

  _running = false;

  if (_task)  _task->join();
}

void StatsMonitor::metric(const std::string& name,
                          const uint64_t&    scalar,
                          Mode               mode)
{
  _names.push_back(name);
  _scalars.push_back(scalar);
  _modes.push_back(mode);

  _previous.push_back(scalar);
}

void StatsMonitor::_routine()
{
  void* context = zmq_ctx_new();
  void* socket  = zmq_socket(context, ZMQ_PUB);
  zmq_connect(socket, _addr);

  char buffer[4096];
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  auto now   = std::chrono::steady_clock::now();
  auto start = std::chrono::duration_cast<std::chrono::duration<int64_t> >(now.time_since_epoch()).count();

  while (_running)
  {
    bool enabled = _enabled; // One more scan when _enabled goes false while sleeping

    std::this_thread::sleep_for(std::chrono::seconds(_period));

    if (enabled)  update(socket, buffer, sizeof(buffer), hostname);
  }

  now = std::chrono::steady_clock::now();
  auto end = std::chrono::duration_cast<std::chrono::duration<int64_t> >(now.time_since_epoch()).count();

  printf("StatsMon exiting after %ld seconds\n", end - start);

  for (unsigned i = 0; i < _scalars.size(); ++i)
  {
    int size = snprintf(buffer, sizeof(buffer), "%s,host=%s,partition=%d %f",
                        _names[i].c_str(), hostname, _partition, 0.0);
    zmq_send(socket, buffer, size, 0);
  }
}

void StatsMonitor::update(void*        socket,
                          char*        buffer,
                          const size_t bufSize,
                          const char*  hostname)
{
  //auto epoch = std::chrono::duration_cast<std::chrono::duration<int64_t> >(now.time_since_epoch()).count();
  //int  size  = snprintf(buffer, bufSize, R"(["%s",{"time": [%ld])", hostname, epoch);
  auto now   = _then;

  for (unsigned i = 0; i < _scalars.size(); ++i)
  {
    now             = std::chrono::steady_clock::now();
    uint64_t scalar = _scalars[i];
    double   value;

    switch (_modes[i])
    {
      case SCALAR:
      {
        //size += snprintf(&buffer[size], bufSize - size, R"(, "%s": [%ld])", _names[i].c_str(), scalar);

        value = scalar;

        break;
      }
      case RATE:
      {
        auto   dC   = (scalar >= _previous[i]) ? scalar - _previous[i] : 0;
        auto   dT   = std::chrono::duration_cast<us_t>(now - _then).count();
        double rate = double(dC) / double(dT) * 1.0e6; // Hz

        //printf("%s: N %016lx, dN %7ld, rate %7.02f KHz\n", _names[i].c_str(), scalar, dC, rate);

        //size += snprintf(&buffer[size], bufSize - size, R"(, "%s": [%.1f])", _names[i].c_str(), rate);

        value = rate;

        _previous[i] = scalar;
        break;
      }
      case CHANGE:
      {
        auto dC = scalar - _previous[i];

        //size += snprintf(&buffer[size], bufSize - size, R"(, "%s": [%ld])", _names[i].c_str(), dC);

        value = dC;

        _previous[i] = scalar;
        break;
      }
    }

    int size = snprintf(buffer, bufSize, "%s,host=%s,partition=%d %f",
                        _names[i].c_str(), hostname, _partition, value);
    if (_verbose)  printf("%s\n", buffer);

    zmq_send(socket, buffer, size, 0);
  }

  //if (_scalars.size() > 0)
  //{
  //  size += snprintf(&buffer[size], bufSize - size, "}]");
  //
  //  if (_verbose)  printf("%s\n", buffer);
  //
  //  zmq_send(socket, buffer, size, 0);
  //}

  _then = now;
}
