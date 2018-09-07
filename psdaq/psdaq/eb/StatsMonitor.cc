#include "psdaq/eb/StatsMonitor.hh"

#include "psdaq/eb/utilities.hh"

#include <zmq.h>

#include <unistd.h>                     // gethostname()
#include <limits.h>
#include <chrono>

using namespace Pds::Eb;

typedef std::chrono::microseconds us_t;


StatsMonitor::StatsMonitor(const std::string& addr,
                           const std::string& partition,
                           unsigned           period,
                           unsigned           verbose) :
  _addr     (addr),
  _partition(partition),
  _period   (period),
  _verbose  (verbose),
  _running  (true),
  _task     (new std::thread([&] { routine(); }))
{
}

StatsMonitor::~StatsMonitor()
{
  delete _task;
}

void StatsMonitor::shutdown()
{
  _running = false;

  if (_task)  _task->join();
}

void StatsMonitor::registerIt(const std::string& name,
                              const uint64_t&    scalar,
                              Mode               mode)
{
  _names.push_back(name);
  _scalars.push_back(scalar);
  _modes.push_back(mode);

  _previous.push_back(scalar);
}

void StatsMonitor::routine()
{
  pinThread(pthread_self(), 1); //lcore2);

  void* context = zmq_ctx_new();
  void* socket  = zmq_socket(context, ZMQ_PUB);
  zmq_connect(socket, _addr.c_str());

  char buffer[4096];
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  auto now   = std::chrono::steady_clock::now();
  auto start = std::chrono::duration_cast<std::chrono::duration<int64_t>>(now.time_since_epoch()).count();

  while (_running)
  {
    std::this_thread::sleep_for(std::chrono::seconds(_period));

    auto epoch = std::chrono::duration_cast<std::chrono::duration<int64_t>>(now.time_since_epoch()).count();
    int  size  = snprintf(buffer, sizeof(buffer), R"(["%s",{"time": [%ld])", hostname, epoch);
    auto then  = now;

    for (unsigned i = 0; i < _scalars.size(); ++i)
    {
      uint64_t scalar = _scalars[i];
      now             = std::chrono::steady_clock::now();

      switch (_modes[i])
      {
        case SCALAR:
        {
          size += snprintf(&buffer[size], sizeof(buffer) - size, R"(, "%s": [%ld])", _names[i].c_str(), scalar);
          break;
        }
        case RATE:
        {
          auto   dT   = std::chrono::duration_cast<us_t>(now - then).count();
          auto   dC   = scalar - _previous[i];
          double rate = double(dC) / double(dT) * 1.0e6; // Hz

          //printf("%s: N %016lx, dN %7ld, rate %7.02f KHz\n", _names[i].c_str(), scalar, dC, rate);

          size += snprintf(&buffer[size], sizeof(buffer) - size, R"(, "%s": [%.1f])", _names[i].c_str(), rate);

          _previous[i] = scalar;
          break;
        }
        case CHANGE:
        {
          auto dC = scalar - _previous[i];

          size += snprintf(&buffer[size], sizeof(buffer) - size, R"(, "%s": [%ld])", _names[i].c_str(), dC);

          _previous[i] = scalar;
          break;
        }
      }
    }

    if (_scalars.size() > 0)
    {
      size += snprintf(&buffer[size], sizeof(buffer) - size, "}]");

      if (_verbose)  printf("%s\n", buffer);

      zmq_send(socket, buffer, size, 0);
    }
  }

  now = std::chrono::steady_clock::now();
  auto end = std::chrono::duration_cast<std::chrono::duration<int64_t>>(now.time_since_epoch()).count();

  printf("StatsMon exiting after %ld seconds\n", end - start);
}
