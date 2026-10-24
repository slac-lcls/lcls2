/**
 * @file
 * @brief MonTracker, a PV monitor whose events are handled on a shared worker thread, with its WorkQueue and Worker interface.
 */
#ifndef Pds_MonTracker_hh
#define Pds_MonTracker_hh

#include <string>
#include <deque>

#include "epicsEvent.h"
#include "epicsMutex.h"
#include "pv/thread.h"
#include "pva/client.h"

#include "PVMonitorCb.hh"

namespace pvd = epics::pvData;

namespace Pds_Epics {

/** Interface for objects whose monitor events are processed on the WorkQueue thread. */
struct Worker
{
  /** Does nothing; empty virtual destructor. */
  virtual ~Worker() {}
  /** Pure virtual: handle one monitor event. */
  virtual void process(const pvac::MonitorEvent& event) = 0;
};

// simple work queue with thread.
// moves monitor queue handling off of PVA thread(s)
/** Queue and thread that move monitor event handling off the pvAccess threads (per the code comment). It holds only weak pointers, so workers must be kept alive elsewhere. */
class WorkQueue : public epicsThreadRunable
{
  typedef std::tr1::shared_ptr<Worker> value_type;
  typedef std::tr1::weak_ptr<Worker> weak_type;
  typedef std::deque<std::pair<weak_type, pvac::MonitorEvent> > queue_t;
public:
  /** Start the worker thread, named Monitor handler. */
  WorkQueue();
  /** Call close(). */
  ~WorkQueue();

  /** Stop accepting events, wake the thread and wait for it to exit. */
  void close();
  /** Queue evt for worker cb and wake the thread if the queue was empty; ignored after close(). */
  void push(const weak_type& cb, const pvac::MonitorEvent& evt);

  /** Thread body: until close(), take each queued event and call process() on its worker if the worker still exists, printing any std::exception it throws. */
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

/** Monitor of one PV whose events are handled on the shared WorkQueue thread. The first data update after connecting calls onConnect(), later ones call updated(), and a disconnect calls onDisconnect(). */
class MonTracker : public pvac::ClientChannel::MonitorCallback,
                   public Worker,
                   public std::tr1::enable_shared_from_this<MonTracker>,
                   public PVMonitorCb
{
public:
  /** Close the shared WorkQueue. */
  static void close();
public:
  /** Expands the pvData POINTER_DEFINITIONS macro for this class; doxygen lists it as a function. */
  POINTER_DEFINITIONS(MonTracker);
  /** Connect to name through pva and start a monitor with request (default field()). */
  MonTracker(const std::string& name,
             const std::string& request = "field()");
  /** Connect to name through ca if provider is ca, otherwise pva, and start a monitor with request. */
  MonTracker(const std::string& provider,
             const std::string& name,
             const std::string& request = "field()");
  /** Cancel the monitor. */
  virtual ~MonTracker();

  /** Return the channel name. */
  const std::string name() const { return _channel.name(); }
  /** Return true from the first data update until a disconnect. */
  bool connected() const { return _connected; }

  /** Monitor callback on a pvAccess thread: queue the event on the WorkQueue for process(). Cancel events, and events that arrive while the object is being destroyed, are dropped. */
  virtual void monitorEvent(const pvac::MonitorEvent& evt) OVERRIDE FINAL;
  /** Handle a monitor event on the WorkQueue thread. Failures and cancels are printed; a disconnect clears connected() and calls onDisconnect(); for data, up to 2 updates are taken per call (the first after connecting calls onConnect(), others updated()) and the event is queued again if 2 were taken. */
  virtual void process(const pvac::MonitorEvent& evt) OVERRIDE FINAL;
protected:
  // These 2 are obsolete but remain defined in case somebody is calling them
  void disconnect() {};
  void reconnect() {};
private:
  static WorkQueue _monwork;            // Static to start only one WorkQueue
protected:
  pvd::PVStructure::const_shared_pointer _strct;
  pvd::PVStructure::shared_pointer _request;
  pvac::ClientChannel _channel;
  pvac::Monitor _mon;
  bool _connected;
};

};

#endif
