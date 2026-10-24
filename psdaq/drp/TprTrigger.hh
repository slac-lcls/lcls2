/**
 * @file
 * @brief TprApp, a collection application that programs TPR trigger outputs for a readout group.
 */
#pragma once

#include <thread>
#include <atomic>
#include "psdaq/service/Collection.hh"
#include "psdaq/tpr/Client.hh"

namespace Drp {

struct TprParameters;

/** CollectionApp (role tpr) that, after connect, programs the TPR channel and outputs given in its parameters to trigger on the partition readout group, and keeps a worker thread until disconnect. */
class TprApp : public CollectionApp
{
public:
    /** Register with the collection as a tpr with the alias from para. */
    TprApp(TprParameters& para);
    /** Unsubscribe from the partition and stop the worker thread. */
    void handleReset(const nlohmann::json& msg) override;
private:
    nlohmann::json connectionInfo(const nlohmann::json& msg) override;
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

