#include <iostream>
#include "PGPReader.hh"
#include "Collector.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "DrpApp.hh"

using namespace Pds::Eb;

DrpApp::DrpApp(Parameters* para) :
    CollectionApp(para->collect_host, para->partition, "drp")
{
    m_para = para;
    std::cout<<"Constructor"<<std::endl;
}

void DrpApp::handleConnect(const json &msg)
{
    int numWorkers = 2;
    int numEntries = 8192;
    int laneMask = 0xf;

    // these parameters must agree with the server side
    size_t maxSize = sizeof(MyDgram);
    m_para->tPrms = { /* .addrs         = */ { },
                     /* .ports         = */ { },
                     /* .ifAddr        = */ nullptr,
                     /* .port          = */ { },
                     /* .id            = */ 0,
                     /* .builders      = */ 0,
                     /* .duration      = */ BATCH_DURATION,
                     /* .maxBatches    = */ MAX_BATCHES,
                     /* .maxEntries    = */ MAX_ENTRIES,
                     /* .maxInputSize  = */ maxSize,
                     /* .maxResultSize = */ maxSize,
                     /* .core          = */ { 11 + 0,
                                            12 },
                     /* .verbose       = */ 0 };

    m_para->mPrms = { /* .addrs         = */ { },
                     /* .ports         = */ { },
                     /* .id            = */ 0,
                     /* .maxEvents     = */ 8,    //mon_buf_cnt,
                     /* .maxEvSize     = */ 1024, //mon_buf_size,
                     /* .maxTrSize     = */ 1024, //mon_trSize,
                     /* .verbose       = */ 0 };

    parseConnectionParams(msg["body"]);

    // should move into constructor
    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    Detector* det = f.create(m_para->detectorType);
    if (det == nullptr) {
        std::cout<< "Error !! Could not create Detector object\n";
    }

    MemPool pool(numWorkers, numEntries);
    PGPReader pgpReader(pool, det, laneMask, numWorkers);
    std::thread pgpThread(&PGPReader::run, std::ref(pgpReader));
    Pds::Eb::TebContributor ebCtrb(m_para->tPrms);
    Pds::Eb::MebContributor* meb = nullptr;
    if (m_para->mPrms.addrs.size() != 0) {
        meb = new Pds::Eb::MebContributor(m_para->mPrms);
    }

    // start performance monitor thread
    std::thread monitor_thread(monitor_func, std::ref(pgpReader.get_counters()),
                               std::ref(pool), std::ref(ebCtrb));

    // reply to collection with connect status
    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
    setState(State::connect);

    // spend all time here blocking listening to PGP
    collector(pool, *m_para, ebCtrb, meb);

    pgpThread.join();
}

void DrpApp::handleReset(const json &msg)
{
    setState(State::reset);
}

void DrpApp::parseConnectionParams(const json& body)
{
    std::string id = std::to_string(getId());
    m_para->tPrms.id = body["drp"][id]["drp_id"];
    const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_MEBS + MAX_MEBS;
    const unsigned tebPortBase = TEB_PORT_BASE + numPorts * m_para->partition;
    const unsigned drpPortBase = DRP_PORT_BASE + numPorts * m_para->partition;
    const unsigned mebPortBase = MEB_PORT_BASE + numPorts * m_para->partition;

    m_para->tPrms.port = std::to_string(drpPortBase + m_para->tPrms.id);

    uint64_t builders = 0;
    for (auto it : body["teb"].items()) {
        unsigned tebId = it.value()["teb_id"];
        // FIXME infiniband -> nic_ip
        std::string address = it.value()["connect_info"]["infiniband"];
        std::cout << "TEB: " << tebId << "  " << address << '\n';
        builders |= 1ul << tebId;
        m_para->tPrms.addrs.push_back(address);
        m_para->tPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
    m_para->tPrms.builders = builders;

    if (body.find("meb") != body.end()) {
        for (auto it : body["meb"].items()) {
            unsigned mebId = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["nic_ip"];
            std::cout << "MEB: " << mebId << "  " << address << '\n';
            m_para->mPrms.addrs.push_back(address);
            m_para->mPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
        }
    }
}
