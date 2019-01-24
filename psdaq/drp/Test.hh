#pragma once

#include "psdaq/service/Collection.hh"

struct Parameters;
class Detector;

class DrpApp : public CollectionApp
{
public:
    DrpApp(Parameters* para);
    void handleConnect(const json& msg) override;
    void handleReset(const json& msg) override;
private:
    void parseConnectionParams(const json& msg);
    Parameters* m_para;
};
