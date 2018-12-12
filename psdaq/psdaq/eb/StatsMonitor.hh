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
      StatsMonitor(const char*        hostname,
                   unsigned           basePort,
                   unsigned           platform,
                   const std::string& partition,
                   unsigned           period,
                   unsigned           verbose);
      ~StatsMonitor();
    public:
      void shutdown();
      void registerIt(const std::string& name,
                      const uint64_t&    scalar,
                      Mode               mode);
    public:
      void routine();
    private:
      std::vector<std::reference_wrapper<const uint64_t>> _scalars;
      std::vector<uint64_t>    _previous;
      std::vector<std::string> _names;
      std::vector<Mode>        _modes;
    private:
      char                     _addr[128];
      const std::string        _partition;
      const unsigned           _period;
      const unsigned           _verbose;
      std::atomic<bool>        _running;
      std::thread*             _task;
    };
  };
};

#endif
