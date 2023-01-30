#include <getopt.h>
#include <unistd.h>

#include "psdaq/epicstools/PvMonitorBase.hh"


namespace Drp {

class PvaMonitor : public Pds_Epics::PvMonitorBase
{
public:
    PvaMonitor(const std::string& channelName, const std::string& provider) :
      Pds_Epics::PvMonitorBase(channelName, provider, provider == "pva"
                                                    ? "field(value,timeStamp,dimension)"
                                                    : "field(value,timeStamp)"),
      m_provider              (provider),
      m_ready                 (false),
      m_verbose               (true)
    {
    }
public:
    void onConnect()    override;
    void onDisconnect() override;
    void updated()      override;
public:
    bool ready();
private:
    const std::string&  m_provider;
    bool                m_ready;
    bool                m_verbose;
};

bool PvaMonitor::ready()
{
    return m_ready;
}

void PvaMonitor::onConnect()
{
    printf("%s connected\n", name().c_str());

    if (m_verbose) {
        printf("+++\n");
        printStructure();
        printf("---\n");
    }
}

void PvaMonitor::onDisconnect()
{
    printf("%s disconnected\n", name().c_str());
}

void PvaMonitor::updated()
{
    m_ready = true;

    if (m_verbose) {
        int64_t seconds;
        int32_t nanoseconds;
        getTimestamp(seconds, nanoseconds);

        printf("Updated: time %9ld.%09d, ", seconds, nanoseconds);

        pvd::ScalarType type;
        size_t          nelem;
        size_t          rank;
        getParams(type, nelem, rank);
        printf("type %d, nelem, %zd, rank %zd\n", type, nelem, rank);
    }
}

} // namespace Drp


int main(int argc, char* argv[])
{
  // Provider is "pva" (default) or "ca"
  std::string pv;                     // [<provider>/]<PV name>
  if (optind < argc)
    pv = argv[optind];
  else {
    printf("A PV ([<provider>/]<PV name>) is mandatory\n");
    return 1;
  }

  std::string provider = "pva";
  auto pos = pv.find("/", 0);
  if (pos != std::string::npos) {
    provider = pv.substr(0, pos);
    pv       = pv.substr(pos+1);
  }

  auto pvMon = std::make_shared<Drp::PvaMonitor>(pv, provider);

  sleep(1);

  if (!pvMon->ready()) {
    printf("Failed to connect with %s\n", pvMon->name().c_str());
    return 1;
  }

  sleep(5);

  return 0;
}
