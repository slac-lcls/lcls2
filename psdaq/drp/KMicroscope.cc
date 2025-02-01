#include "KMicroscope.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include <iostream>
#include <cstdlib>
#include <string>

using logging = psalg::SysLog;
using json = nlohmann::json;

namespace Drp {

KMicroscope::KMicroscope(Parameters& para, DrpBase& drp)
    : XpmDetector(&para, &drp.pool)
{
    logging::info("KMicroscope instance created");
}

KMicroscope::~KMicroscope() {
    logging::info("KMicroscope instance destroyed");
}

void KMicroscope::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) {
    logging::debug("KMicroscope::event called");

    // Process events and populate the datagram
    // Example: Attach dummy data to the XTC
    auto& xtc = dgram.xtc;
    auto namesId = XtcData::NamesId(this->nodeId, 0);
    XtcData::DescribedData desc(xtc, bufEnd, this->namesLookup(), namesId);
    int dummyData = 42;
    memcpy(desc.data(), &dummyData, sizeof(dummyData));
    desc.set_data_length(sizeof(dummyData));
}

CustomBldApp::CustomBldApp(Parameters& para,
                           DrpBase& drp,
                           const std::string& customParam,
                           int measurementTimeMs,
                           const std::string& iniFilePath,
                           size_t batchSize)
    : BldApp(para, drp, std::make_unique<Drp::KMicroscope>(para, drp)),
      m_customParam(customParam),
      m_measurementTimeMs(measurementTimeMs)
{
    logging::info("CustomBldApp initialized with KMicroscope (customParam: %s, measurementTime: %d ms)",
                  m_customParam.c_str(), m_measurementTimeMs);
}

CustomBldApp::~CustomBldApp() {
    logging::info("Shutting down CustomBldApp...");
}

void CustomBldApp::run() {
    logging::info("Running CustomBldApp with customParam: %s and measurementTime: %d ms",
                  m_customParam.c_str(), m_measurementTimeMs);
    BldApp::run();  // Call the base class's run method
}

KMicroscopeBld::KMicroscopeBld(int measurementTimeMs,
                               const std::string& iniFilePath,
                               size_t batchSize)
    : Bld(0, 0, 0, 0, 0, 0, 0),  // These values are unused
      m_callbackHandler(measurementTimeMs, iniFilePath, batchSize), 
      m_measurementTimeMs(measurementTimeMs)
{
}

KMicroscopeBld::~KMicroscopeBld() {
    // Nothing additional to do; m_callbackHandler cleans up automatically.
}

uint64_t KMicroscopeBld::next() {
    sc_DldEvent event;
    // Busy–wait until an event is available.
    // Optionally, you could call m_callbackHandler.flushPending() here if desired.
    while (!m_callbackHandler.popEvent(event)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    // Return only the time_tag from the event.
    return event.time_tag;
}

}

int main(int argc, char* argv[]) {
    // Default values.
    std::string iniFile = "tdc_gpx3.ini";
    int measurementTimeMs = 1000; // For example, 1000 ms.
    size_t batchSize = 100;        // Default batch size.

    // Simple command line argument parsing.
    // Usage: runner -i <ini file> -t <measurement time ms> -b <batch size>
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if ((arg == "-i" || arg == "--ini") && i + 1 < argc) {
            iniFile = argv[++i];
        } else if ((arg == "-t" || arg == "--time") && i + 1 < argc) {
            measurementTimeMs = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--batch") && i + 1 < argc) {
            batchSize = static_cast<size_t>(std::atoi(argv[++i]));
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " -i <ini file> -t <measurement time ms> -b <batch size>\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "Using INI file: " << iniFile << "\n";
    std::cout << "Measurement time: " << measurementTimeMs << " ms\n";
    std::cout << "Batch size: " << batchSize << "\n";

    try {
        // Create the KMicroscopeBld instance using the provided parameters.
        Drp::KMicroscopeBld microscope(measurementTimeMs, iniFile, batchSize);

        // For demonstration, retrieve 10 pulse IDs.
        for (int i = 0; i < 10; i++) {
            uint64_t pulse = microscope.next();
            std::cout << "Pulse " << i << ": " << pulse << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*int main(int argc, char* argv[])
{
    Drp::Parameters para;
    std::string kwargs_str;
    int measurementTimeMs = 0;  // Default to 0 if `-t` is not provided
    int c;
    while((c = getopt(argc, argv, "p:o:C:b:d:D:u:P:T::k:M:v")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'b':
                para.detName = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'D':
                para.detType = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'k':
                kwargs_str = kwargs_str.empty()
                           ? optarg
                           : kwargs_str + "," + optarg;
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            case 'v':
                ++para.verbose;
                break;
            case 't': measurementTimeMs = std::stoi(optarg); break;  // Sets value only if `-t` is provided
            default:
                return 1;
        }
    }

    switch (para.verbose) {
      case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
      default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");
    if (optind < argc)
    {
        logging::error("Unrecognized argument:");
        while (optind < argc)
            logging::error("  %s ", argv[optind++]);
        return 1;
    }
    if (para.instrument.empty()) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        return 1;
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        return 1;
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        return 1;
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        return 1;
    }
    para.detName = "bld";  //para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));
    get_kwargs(kwargs_str, para.kwargs);
    for (const auto& kwargs : para.kwargs) {
        if (kwargs.first == "forceEnet")      continue;
        if (kwargs.first == "ep_fabric")      continue;
        if (kwargs.first == "ep_domain")      continue;
        if (kwargs.first == "ep_provider")    continue;
        if (kwargs.first == "sim_length")     continue;  // XpmDetector
        if (kwargs.first == "timebase")       continue;  // XpmDetector
        if (kwargs.first == "pebbleBufSize")  continue;  // DrpBase
        if (kwargs.first == "pebbleBufCount") continue;  // DrpBase
        if (kwargs.first == "batching")       continue;  // DrpBase
        if (kwargs.first == "directIO")       continue;  // DrpBase
        if (kwargs.first == "pva_addr")       continue;  // DrpBase
        if (kwargs.first == "interface")      continue;
        logging::critical("Unrecognized kwarg '%s=%s'\n",
                          kwargs.first.c_str(), kwargs.second.c_str());
        return 1;
    }
    */

    /*
    //  Add pva_addr to the environment
    if (para.kwargs.find("pva_addr")!=para.kwargs.end()) {
        const char* a = para.kwargs["pva_addr"].c_str();
        char* p = getenv("EPICS_PVA_ADDR_LIST");
        char envBuff[256];
        if (p)
            sprintf(envBuff,"EPICS_PVA_ADDR_LIST=%s %s", p, a);
        else
            sprintf(envBuff,"EPICS_PVA_ADDR_LIST=%s", a);
        logging::info("Setting env %s\n", envBuff);
        putenv(envBuff);
    }
    */

    /*para.maxTrSize = 256 * 1024;
    std::string customParam = "KMicroscope Custom Test Parameter";
    logging::info("Measurement time set to %d ms, Custom Param: %s", measurementTimeMs, customParam.c_str());

    Py_Initialize();  // Initialize Python before creating any objects

    ZmqContext zmqCtx;
    try {
        Drp::DrpBase drp(para, zmqCtx);  // Create DrpBase first
        Drp::CustomBldApp app(para, drp, customParam, measurementTimeMs);
        app.run();
        return 0;
    }
    catch (std::exception& e)  { logging::critical("%s", e.what()); }
    catch (std::string& e)     { logging::critical("%s", e.c_str()); }
    catch (char const* e)      { logging::critical("%s", e); }
    catch (...)                { logging::critical("Default exception"); }
    return EXIT_FAILURE;
    
}*/



