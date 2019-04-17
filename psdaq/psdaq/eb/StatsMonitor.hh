#ifndef Pds_Eb_StatsMonitor_hh
#define Pds_Eb_StatsMonitor_hh

#include <stdint.h>
#include <string>
#include <vector>
#include <atomic>
#include <thread>


namespace Pds {
  namespace Eb {

    class StatsMonitor
    {
    public:
      enum Mode { SCALAR, RATE, CHANGE };
    public:
      StatsMonitor(const char* hostname,
                   unsigned    basePort,
                   unsigned    partition,
                   unsigned    period,
                   unsigned    verbose);
      ~StatsMonitor();
    public:
      void enable()   { _enabled = true;  }
      void disable()  { _enabled = false; }
      void startup();
      void shutdown();
      void metric(const std::string& name,
                  const uint64_t&    scalar,
                  Mode               mode);
    public:
      void update(void*        socket,
                  char*        buffer,
                  const size_t bufSize,
                  const char*  hostname);
    private:
      void _routine();
    private:
      std::vector<std::reference_wrapper<const uint64_t> > _scalars;
      std::vector<uint64_t>                                _previous;
      std::vector<std::string>                             _names;
      std::vector<Mode>                                    _modes;
    private:
      char                                                 _addr[128];
      const unsigned                                       _partition;
      const unsigned                                       _period;
      const unsigned                                       _verbose;
      std::atomic<bool>                                    _enabled;
      std::atomic<bool>                                    _running;
      std::chrono::steady_clock::time_point                _then;
      std::thread*                                         _task;
    };
  };
};

#endif
