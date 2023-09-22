#include "TprTrigger.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <getopt.h>
#include "drp.hh"
#include "psdaq/service/kwargs.hh"
#include "psalg/utils/SysLog.hh"

using json = nlohmann::json;
using logging = psalg::SysLog;

using namespace Pds::Tpr;

namespace Drp {

struct TprParameters : public Parameters
{
  unsigned output  = 0;
  unsigned channel = 10;
  unsigned delay   = 0;
  unsigned width   = 1;
  unsigned tap     = 0;
  unsigned polarity = 0; // Falling
  int  clksel      = 1;
  int  modsel      = 0;
  int  loopout     = -1;
};


extern int optind;

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("          -t <dev>  : <tpr a/b>\n");
  printf("          -c <chan> : logic channel\n");
  printf("          -o <outp> : bit mask of outputs\n");
  printf("          -d <clks> : delay\n");
  printf("          -w <clks> : width\n");
  printf("          -D <unit> : delay tap\n");
  printf("          -P <0/1>  : polarity (falling/rising)\n");
  printf("          -K        : clksel (0=LCLS1,1=LCLS2[default]\n");
  printf("          -M        : modsel (0=clksel[default],1=LCLS1,2=LCLS2\n");
  printf("          -L        : loopout (0=false, 1=true)\n");
}


TprApp::TprApp(TprParameters& para) :
    CollectionApp(para.collectionHost, para.partition, "tpr", para.alias),
    m_para(para)
{
}

void TprApp::_disconnect()
{
    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
}

json TprApp::connectionInfo()
{
    json body = {{}};
    return body;
}

void TprApp::_worker()
{
    printf("Configuring channel %u outputs 0x%x for %s %u\n",
           m_para.channel, m_para.output, "Partition", m_group);

    Pds::Tpr::Client client(m_para.device.c_str(), m_para.channel, m_para.clksel>0, (Client::ModeSel)m_para.modsel);

    client.stop();  // Is this enough to clear the trigger fifo?
    usleep(1000000);

    if (m_para.loopout>=0)
        client.loopOut(m_para.loopout>0);

    client.reg().tpr.dump();

    auto output = m_para.output;
    for(unsigned i=0; output; i++) {
        if (output & (1<<i)) {
            client.setup(i, m_para.delay, m_para.width, m_para.polarity, m_para.tap);
            output &= ~(1<<i);
        }
    }

    client.start(TprBase::Partition((TprBase::Partition)m_group));

    client.reg().base.dump();
    for(unsigned i=0; i<4; i++) {
        usleep(1000000);
        client.reg().base.dump();
    }

    client.release();

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        sleep(1);
    }
    logging::info("Worker thread exiting\n");
}

void TprApp::handleConnect(const nlohmann::json& msg)
{
    std::string id = std::to_string(getId());
    m_group = msg["body"]["tpr"][id]["det_info"]["readout"];

    m_terminate.store(false, std::memory_order_release);

    m_workerThread = std::thread{&TprApp::_worker, this};

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void TprApp::handleDisconnect(const json& msg)
{
    _disconnect();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void TprApp::handleReset(const nlohmann::json& msg)
{
    unsubscribePartition();    // ZMQ_UNSUBSCRIBE
    _disconnect();
}

} // namespace Drp


int main(int argc, char* argv[])
{
    extern char* optarg;
    bool lUsage = false;
    Drp::TprParameters para;
    char tprid='a';
    std::string kwargs_str;
    int c;
    while((c = getopt(argc, argv, "p:C:u:k:H:c:d:D:w:o:t:K:L:M:P:hv?")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + ", " + optarg;
                break;
            case 'H':
                para.instrument = optarg;
                break;
            case 'K':                   // Was 'C'
                para.clksel = strtoul(optarg,NULL,0);
                break;
            case 'L':
                para.loopout = strtoul(optarg,NULL,0);
                break;
            case 'M':
                para.modsel = strtoul(optarg,NULL,0);
                break;
            case 'c':
                para.channel = strtoul(optarg,NULL,0);
                break;
            case 'd':
                para.delay = strtoul(optarg,NULL,0);
                break;
            case 'D':
                para.tap = strtoul(optarg,NULL,0);
                break;
            case 'w':
                para.width = strtoul(optarg,NULL,0);
                break;
            case 't':
                tprid  = optarg[0];
                if (strlen(optarg) != 1) {
                  printf("%s: option `-t' parsing error\n", argv[0]);
                  lUsage = true;
                }
                break;
            case 'o':
                para.output = strtoul(optarg,NULL,0);
                break;
            case 'P':
                para.polarity = strtoul(optarg,NULL,0);
                break;
            case 'v':
                ++para.verbose;
                break;
            case 'h':
                Drp::usage(argv[0]);
                exit(0);
            case '?':
            default:
                return 1;
        }
    }

    if (optind < argc) {
        printf("%s: invalid argument -- %s\n",argv[0], argv[optind]);
        lUsage = true;
    }

    if (lUsage) {
        Drp::usage(argv[0]);
        exit(1);
    }

    switch (para.verbose) {
        case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
        default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");

    // Check required parameters
    if (para.instrument.empty()) {
        logging::warning("-H: instrument name is missing");
    }
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        return 1;
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        return 1;
    }

    char evrdev[16];
    sprintf(evrdev,"/dev/tpr%c",tprid);
    para.device = std::string(evrdev);
    para.detName = para.alias;
    para.detSegment = 0;

    try {
        get_kwargs(kwargs_str, para.kwargs);
        for (const auto& kwargs : para.kwargs) {
            logging::critical("Unrecognized kwarg '%s=%s'\n",
                              kwargs.first.c_str(), kwargs.second.c_str());
            return 1;
        }

        Drp::TprApp(para).run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
}
