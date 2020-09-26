#include "MonTracker.hh"

#include <epicsGuard.h>

#include <pv/pvData.h>
#include <pv/configuration.h>
#include <pv/caProvider.h>

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
pvac::ClientProvider MonTracker::_caProvider;
pvac::ClientProvider MonTracker::_pvaProvider;

int MonTracker::prepare()
{
  // Do the following only once per process instance
  static bool prepared = false;
  if (!prepared) {
    prepared = true;

    // explicitly select configuration from process environment
    pva::Configuration::shared_pointer configuration(pva::ConfigurationBuilder()
                                                     .push_env()
                                                     .build());
    // "pva" provider automatically in registry
    // add "ca" provider to registry
    epics::pvAccess::ca::CAClientFactory::start();

    _caProvider  = pvac::ClientProvider("ca",  configuration);
    _pvaProvider = pvac::ClientProvider("pva", configuration);
  }
  return 0;
}

void MonTracker::shutdown()
{
  _caProvider.disconnect();
  _pvaProvider.disconnect();
}

void MonTracker::close()
{
  _monwork.close();
}

void MonTracker::getDone(const pvac::GetEvent &evt)
{
  if (evt.event == pvac::GetEvent::Fail) {
    std::cerr << "Error getting the value of PV " << name() << " " << evt.message << "\n";
  } else if (evt.event == pvac::GetEvent::Success)  {
    _promise.set_value(evt.value);
    _mon = _channel.monitor(this, _pvRequest);
  } else {
    std::cerr << "Cancelled getting the value of PV " << name() << " " << evt.event << "\n";
  }
}

void MonTracker::connectEvent(const pvac::ConnectEvent &evt)
{
  _connected = evt.connected;
  if (_connected) {
    _op = _channel.get(this);
  }
}

void MonTracker::monitorEvent(const pvac::MonitorEvent& evt)
{
  // shared_from_this() will fail as Cancel is delivered in our dtor.
  if(evt.event==pvac::MonitorEvent::Cancel) return;
  // running on internal provider worker thread
  // minimize work here.
  // TODO: bound queue size
  _monwork.push(shared_from_this(), evt);
}

bool MonTracker::getComplete(ProviderType providerType, const std::string& request)
{
  if (_strct != NULL) { return true; }

  // build "pvRequest" which by default asks for all fields
  try {
    _pvRequest = pvd::createRequest(request);
  } catch(std::exception& e){
    std::cerr << "Failed to parse request string '" << request << "': " << e.what() << "\n";
    return 1;
  }

  switch (providerType) {
    case CA:   _channel = _caProvider.connect(_name);   break;
    case PVA:  _channel = _pvaProvider.connect(_name);  break;
  }
  _channel.addConnectListener(this);

  std::future<pvd::PVStructure::const_shared_pointer> ft = _promise.get_future();
  std::future_status status = ft.wait_for(std::chrono::seconds(30));
  if (status == std::future_status::ready) {
    _strct = ft.get();
    // Sending the onConnect message after the get; most users expect the data to be available on connect.
    onConnect();
    return true;
  } else {
    std::cerr << "Timeout getting the value of PV " << name() << "\n";
    return false;
  }
}

void MonTracker::disconnect()
{
  if (_connected) {
    _channel.removeConnectListener(this);
    _op.cancel();
    _connected = false;
  }
}

void MonTracker::process(const pvac::MonitorEvent& evt)
{
  // running on our worker thread
  switch(evt.event) {
    case pvac::MonitorEvent::Fail:
      std::cout<<"Error "<<_name<<" "<<evt.message<<"\n";
      break;
    case pvac::MonitorEvent::Cancel:
      std::cout<<"Cancel "<<_name<<"\n";
      break;
    case pvac::MonitorEvent::Disconnect:
      std::cout<<"Disconnect "<<_name<<"\n";
      onDisconnect();
      break;
    case pvac::MonitorEvent::Data:
    {
      unsigned n;
      for(n=0; n<2 && _mon.poll(); n++) {
        //pvd::PVField::const_shared_pointer fld(_mon.root->getSubField("value"));
        //if(!fld)
        //  fld = _mon.root;
        //std::cout<<"Event "<<_name<<" "<<fld
        //         <<" Changed:"<<_mon.changed
        //         <<" overrun:"<<_mon.overrun<<"\n";
        _strct = _mon.root;
        updated();
      }
      if(n==2) {
        // too many updates, re-queue to balance with others
        _monwork.push(shared_from_this(), evt);
      } else if(n==0) {
        std::cerr<<"Spurious Data event "<<_name<<"\n";
      }
      break;
    }
  }
}
