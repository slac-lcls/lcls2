#pragma once

#include <thread>
#include <atomic>
#include "psdaq/service/Collection.hh"
#include "psdaq/tpr/Client.hh"

namespace Drp {

struct TprParameters;

class TprApp : public CollectionApp
{
public:
    TprApp(TprParameters& para);
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo() override;
    void handleConnect(const nlohmann::json& msg) override;
    void handleDisconnect(const nlohmann::json& msg) override;
    void _disconnect();
    void _worker();
private:
    TprParameters& m_para;
    unsigned m_group;
    std::thread m_workerThread;
    std::atomic<bool> m_terminate;
};

}

