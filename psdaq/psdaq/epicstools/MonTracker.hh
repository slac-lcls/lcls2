#ifndef Pds_MonTracker_hh
#define Pds_MonTracker_hh

#include <string>
#include <deque>
#include <future>

#include <epicsEvent.h>
#include <epicsMutex.h>
#include <pv/thread.h>
#include <pva/client.h>

#include "PVMonitorCb.hh"

namespace pvd = epics::pvData;

namespace Pds_Epics {

struct Worker
{
  virtual ~Worker() {}
  virtual void process(const pvac::MonitorEvent& event) = 0;
};

// simple work queue with thread.
// moves monitor queue handling off of PVA thread(s)
class WorkQueue : public epicsThreadRunable
{
  typedef std::tr1::shared_ptr<Worker> value_type;
  typedef std::tr1::weak_ptr<Worker> weak_type;
  typedef std::deque<std::pair<weak_type, pvac::MonitorEvent> > queue_t;
public:
  WorkQueue();
  ~WorkQueue();

  void close();
  void push(const weak_type& cb, const pvac::MonitorEvent& evt);

  virtual void run() OVERRIDE FINAL;
private:
  epicsMutex _mutex;
  // work queue holds only weak_ptr
  // so jobs must be kept alive seperately
  queue_t _queue;
  epicsEvent _event;
  bool _running;
  pvd::Thread _worker;
};

class MonTracker : public pvac::ClientChannel::ConnectCallback,
                   public pvac::ClientChannel::GetCallback,
                   public pvac::ClientChannel::MonitorCallback,
                   public Worker,
                   public std::tr1::enable_shared_from_this<MonTracker>,
                   public PVMonitorCb
{
public:
  static void close();
public:
  POINTER_DEFINITIONS(MonTracker);
  MonTracker(const std::string& name) :_name(name) {}
  virtual ~MonTracker() { _mon.cancel(); }

  enum ProviderType { CA, PVA };

  const std::string& name() const { return _name; }
  bool connected() const { return _connected; }

  virtual void getDone (const pvac::GetEvent &evt) OVERRIDE FINAL;
  virtual void connectEvent (const pvac::ConnectEvent &evt) OVERRIDE FINAL;
  virtual void monitorEvent(const pvac::MonitorEvent& evt) OVERRIDE FINAL;
  virtual void process(const pvac::MonitorEvent& evt) OVERRIDE FINAL;
protected:
  bool getComplete(ProviderType providerType, const std::string& request = "field()");
  void disconnect();
private:
  static WorkQueue _monwork;            // Static to start only one WorkQueue
  std::promise<pvd::PVStructure::const_shared_pointer> _promise;
protected:
  const std::string _name;
  bool _connected;
  pvd::PVStructure::const_shared_pointer _strct;
  pvd::PVStructure::shared_pointer _pvRequest;
  static pvac::ClientProvider _caProvider;            // Only one for the process
  static pvac::ClientProvider _pvaProvider;           // Only one for the process
  pvac::ClientChannel _channel;
  pvac::Operation _op;
  pvac::Monitor _mon;
};

};

#endif
