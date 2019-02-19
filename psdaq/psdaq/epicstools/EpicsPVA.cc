//--------------------------------------------------------------------------
//
// File and Version Information:
// 	$Id: EpicsPVA.cc 6181 2016-07-07 01:06:10Z tookey@SLAC.STANFORD.EDU $
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
//
//------------------------------------------------------------------------

#include "psdaq/epicstools/EpicsPVA.hh"

#include <string.h>
#include <stdlib.h>

#include <stdio.h>

namespace Pds_Epics {
    static pvac::ClientProvider provider("pva");

    EpicsPVA::EpicsPVA(const char *channelName, const int maxElements) :
    _channel(provider.connect(channelName)), _monitorCB(NULL)
    {
        _channel.addConnectListener(this);
    }
    EpicsPVA::EpicsPVA(const char *channelName, PVMonitorCb* monitor, const int maxElements) :
    _channel(provider.connect(channelName)),
    _monitorCB(monitor)
    {
        _channel.addConnectListener(this);
    }

    EpicsPVA::~EpicsPVA() {
      _channel.removeConnectListener(this);
      _op.cancel();
    }

    void EpicsPVA::getDone (const pvac::GetEvent &evt) {
      if (evt.event == pvac::GetEvent::Fail) {
          std::cerr << "Error getting the value of PV " << name() << " " << evt.message << "\n";
      } else if (evt.event == pvac::GetEvent::Success)  {
          _promise.set_value(evt.value);
          if(_monitorCB) {
              _pvmon = _channel.monitor(this);
          }
      } else {
          std::cerr << "Cancelled getting the value of PV " << name() << " " << evt.event << "\n";
      }
    }

    void EpicsPVA::connectEvent (const pvac::ConnectEvent &evt) {
      _connected = evt.connected;
      if(_connected) {
          _op = _channel.get(this);
      }
    }

    void EpicsPVA::monitorEvent (const pvac::MonitorEvent &evt) {
        switch(evt.event) {
            case pvac::MonitorEvent::Fail:
            std::cerr << "Error " << name() << " " <<evt.message<<"\n";
            break;
            case pvac::MonitorEvent::Cancel:
            std::cerr << "Cancel " << name() << "\n";
            break;
            case pvac::MonitorEvent::Disconnect:
            std::cerr << "Disconnect " << name() << "\n";
            break;
            case pvac::MonitorEvent::Data:
            {
                while(_pvmon.poll()) {
                    _strct = _pvmon.root;
                    if(_monitorCB != NULL) { _monitorCB->updated(); }
                }
                break;
            }
        }
    }

    bool EpicsPVA::getComplete() {
        if (_strct != NULL) { return true; }
        std::future<pvd::PVStructure::const_shared_pointer> ft = _promise.get_future();
        std::future_status status = ft.wait_for(std::chrono::seconds(30));
        if (status == std::future_status::ready) {
            _strct = ft.get();
            // Sending the onConnect message after the get; most users expect the data to be available on connect.
            this->onConnect();
            return true;
        } else {
            std::cerr << "Timeout getting the value of PV " << name() << "\n";
            return false;
        }
    }


    void EpicsPVA::onConnect() {

    }

    long EpicsPVA::sec() {
      if (!getComplete()) { return -1; }
      return _strct->getSubField<pvd::PVStructure>("timeStamp")->getSubField<pvd::PVLong>("secondsPastEpoch")->get();
    }
    int EpicsPVA::nsec() {
      if (!getComplete()) { return -1; }
      return _strct->getSubField<pvd::PVStructure>("timeStamp")->getSubField<pvd::PVInt>("nanoseconds")->get();
    }

    size_t EpicsPVA::nelem() {
        if (!getComplete()) { return -1; }
        return _strct->getSubField<pvd::PVArray>("value")->getLength();
    }
}
