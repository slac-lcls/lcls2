#include "MonTracker.hh"
#include "EpicsProviders.hh"

#include <epicsGuard.h>

#include <pv/pvData.h>
#include <pv/createRequest.h>

#include <exception>

using namespace Pds_Epics;

typedef epicsGuard<epicsMutex> Guard;
typedef epicsGuardRelease<epicsMutex> UnGuard;

WorkQueue::WorkQueue() :
  _running(true),
  _worker(pvd::Thread::Config()
          .name("Monitor handler")
          .autostart(true)
          .run(this))
{
}

WorkQueue::~WorkQueue()
{
  close();
}

void WorkQueue::close()
{
  {
    Guard G(_mutex);
    _running = false;
  }
  _event.signal();
  _worker.exitWait();
}

void WorkQueue::push(const weak_type& cb, const pvac::MonitorEvent& evt)
{
  bool wake;
  {
    Guard G(_mutex);
    if(!_running) return; // silently refuse to queue during/after close()
    wake = _queue.empty();
    _queue.push_back(std::make_pair(cb, evt));
  }
  if(wake)
    _event.signal();
}

void WorkQueue::run()
{
  Guard G(_mutex);
  while(_running) {
    if(_queue.empty()) {
      UnGuard U(G);
      _event.wait();
    } else {
      queue_t::value_type ent(_queue.front());
      value_type cb(ent.first.lock());
      _queue.pop_front();
      if(!cb) continue;
      try {
        UnGuard U(G);
        cb->process(ent.second);
      }catch(std::exception& e){
        std::cout<<"Error in monitor handler : "<<e.what()<<"\n";
      }
    }
  }
}

Pds_Epics::WorkQueue MonTracker::_monwork;

MonTracker::MonTracker(const std::string& name,
                       const std::string& request) :
  _request(pvd::createRequest(request)),
  _channel(EpicsProviders::pva().connect(name)),
  _mon(_channel.monitor(this, _request)),
  _connected(false)
{
}

MonTracker::MonTracker(const std::string& provider,
                       const std::string& name,
                       const std::string& request) :
  _request(pvd::createRequest(request)),
  _channel(provider == "ca" ?
           EpicsProviders::ca ().connect(name) :
           EpicsProviders::pva().connect(name)),
  _mon(_channel.monitor(this, _request)),
  _connected(false)
{
}

MonTracker::~MonTracker()
{
  _mon.cancel();
}

void MonTracker::close()
{
  _monwork.close();
}

void MonTracker::monitorEvent(const pvac::MonitorEvent& evt)
{
  // shared_from_this() will fail as Cancel is delivered in our dtor.
  if(evt.event==pvac::MonitorEvent::Cancel) return;
  // running on internal provider worker thread
  // minimize work here.
  // TODO: bound queue size
  try {
    _monwork.push(shared_from_this(), evt);
  } catch(std::bad_weak_ptr&) {
    // There's a race between update events and the dtor being called that can
    // cause these (?).  Since it occurs only at dtor time, it can be ignored.
    //printf("*** Caught bad weak ptr exception\n");
  } catch(std::exception& error) {
    std::cerr<<"MonTracker::monitorEvent: Error: "<<error.what()<<"\n";
  }
}

void MonTracker::process(const pvac::MonitorEvent& evt)
{
  // running on our worker thread
  switch(evt.event) {
    case pvac::MonitorEvent::Fail:
      std::cout<<"Error "<<name()<<" "<<evt.message<<"\n";
      break;
    case pvac::MonitorEvent::Cancel:
      std::cout<<"Cancel "<<name()<<"\n";
      break;
    case pvac::MonitorEvent::Disconnect:
      std::cout<<"Disconnect "<<name()<<"\n";
      _connected = false;
      onDisconnect();
      break;
    case pvac::MonitorEvent::Data:
    {
      unsigned n;
      for(n=0; n<2 && _mon.poll(); n++) {
        //pvd::PVField::const_shared_pointer fld(_mon.root->getSubField("value"));
        //if(!fld)
        //  fld = _mon.root;
        //std::cout<<"Event "<<name()<<" "<<fld
        //         <<" Changed:"<<_mon.changed
        //         <<" overrun:"<<_mon.overrun<<"\n";
        _strct = _mon.root;
        if (!_connected) {
          _connected = true;
          onConnect();
          continue;                  // Call updated() only when PV is updating
        }
        updated();
      }
      if(n==2) {
        // too many updates, re-queue to balance with others
        _monwork.push(shared_from_this(), evt);
      //} else if(n==0) {
      //  std::cerr<<"Spurious Data event "<<name()<<"\n";
      }
      break;
    }
  }
}
