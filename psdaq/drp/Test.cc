#include <iostream>
#include "PGPReader.hh"
#include "Collector.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "Test.hh"

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
    parseConnectionParams(msg["body"]);

    // should move into constructor
    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    Detector* det = f.create(m_para->detectorType);

    MemPool pool(numWorkers, numEntries);
    PGPReader pgpReader(pool, det, laneMask, numWorkers);
    std::thread pgpThread(&PGPReader::run, std::ref(pgpReader));
    Pds::Eb::TebContributor ebCtrb(m_para->tPrms);
    Pds::Eb::MebContributor* meb = nullptr;
    if (m_para->mPrms.addrs.size() != 0) {
        meb = new Pds::Eb::MebContributor(m_para->mPrms);
    }

    // reply to collection with connect status
    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
    setState(State::connect);

    // collector(pool, m_para.get(), ebCtrb, meb);
    pgpThread.join();

    /*
    // make connections and reply
    MemPool pool(num_workers, num_entries);
    // TODO: This should be moved to configure when the lane_mask is known.
    PGPReader pgp_reader(pool, lane_mask, num_workers);

    // spend all time here blocking listening to PGP
    collector(pool, para, ebCtrb, meb);
    pgp_thread.join();
    for (int i = 0; i < num_workers; i++) {
        worker_threads[i].join();
    */
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
        std::string address = it.value()["connect_info"]["nic_ip"];
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

#include <getopt.h>

int main(int argc, char* argv[])
{
    Parameters para;
    int c;
    while((c = getopt(argc, argv, "p:o:D:C:")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.output_dir = optarg;
                break;
            case 'D':
                para.detectorType = optarg;
                break;
            case 'C':
                para.collect_host = optarg;
                break;
            default:
                exit(1);
        }
    }
    DrpApp app(&para);
    app.run();
}
