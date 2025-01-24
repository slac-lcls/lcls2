#include "sls/Detector.h"

#include "JungfrauDetectorId.hh"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <getopt.h>

constexpr unsigned defPort { 32410 };
constexpr unsigned defNumImages { 1 };
constexpr unsigned defBias { 200 };
constexpr double defExposureTime { 0.00001 };
constexpr double defExposurePeriod { 0.2 };
constexpr double defTriggerDelay { 0.000238 };

constexpr double tsClock { 10.e6 };

static const std::vector<sls::defs::gainMode> slsGains {
  sls::defs::DYNAMIC,
  sls::defs::FORCE_SWITCH_G1,
  sls::defs::FORCE_SWITCH_G2,
  sls::defs::FIX_G1,
  sls::defs::FIX_G2,
};

static const std::vector<sls::defs::speedLevel> slsSpeeds {
  sls::defs::FULL_SPEED,
  sls::defs::HALF_SPEED,
  sls::defs::QUARTER_SPEED,
};

using namespace Drp::Jungfrau;

static void showUsage(const char* p)
{
  std::cout << "Usage: " << p << " [-v|--version] [-h|--help]" << std::endl;
  std::cout << "-H|--host <host> [-P|--port <port>]" << std::endl;
  std::cout << "[-w|--write <filename prefix>] [-n|--number <number of images>] [-e|--exposure <exposure time (sec)>]" << std::endl;
  std::cout << "[-b|--bias <bias>] [-g|--gain <gain>] [-S|--speed <speed>] [-t|--trigger <delay>] [-r|--receiver]" << std::endl;
  std::cout << "-H|--host <host> [-P|--port <port>] -m|--mac <mac> -d|--detip <detip> -s|--sls <sls>" << std::endl;
  std::cout << " Options:" << std::endl;
  std::cout << "    -w|--write    <filename prefix>       output filename prefix" << std::endl;
  std::cout << "    -n|--number   <number of images>      number of images to be captured (default: " << defNumImages << ")" << std::endl;
  std::cout << "    -e|--exposure <exposure time>         exposure time (sec) (default: " << defExposureTime << " sec)" << std::endl;
  std::cout << "    -E|--period   <exposure period>       exposure period (sec) (default: " << defExposurePeriod << " sec)" << std::endl;
  std::cout << "    -b|--bias     <bias>                  the bias voltage to apply to the sensor (default: " << defBias << ")" << std::endl;
  std::cout << "    -g|--gain     <gain 0-4>              the gain mode of the detector (default: Dynamic - 0)" << std::endl;
  std::cout << "    -S|--speed    <speed 0-2>             the clock speed mode of the detector (default: Half - 1)" << std::endl;
  std::cout << "    -P|--port     <port>                  set the receiver udp port number (default: " << defPort << ")" << std::endl;
  std::cout << "    -H|--host     <host>                  set the receiver host ip" << std::endl;
  std::cout << "    -m|--mac      <mac>                   set the receiver mac address" << std::endl;
  std::cout << "    -d|--detip    <detip>                 set the detector ip address" << std::endl;
  std::cout << "    -s|--sls      <sls>                   set the hostname of the slsDetector interface" << std::endl;
  std::cout << "    -t|--trigger  <delay>                 the internal acquisition start delay to use when externally triggered (sec) (default: " << defTriggerDelay << ")" << std::endl;
  std::cout << "    -G|--highg0                           use high G0 gain mode (default: false)" << std::endl;
  std::cout << "    -I|--internal                         use the internal trigger for acquistion (default: false)" << std::endl;
  std::cout << "    -i|--info                             display additional info about recieved frames (default: false)" << std::endl;
  std::cout << "    -v|--version                          show file version" << std::endl;
  std::cout << "    -h|--help                             print this message and exit" << std::endl;
}

static void showVersion(const char* p)
{
  std::cout << "Version: " << p << " Ver 1.0.0" << std::endl; 
}

int main(int argc, char* argv[])
{
  const char*         strOptions  = ":vhw:n:e:E:b:g:S:P:H:m:d:s:t:GIiD:";
  const struct option loOptions[] =
  {
    {"version",     0, 0, 'v'},
    {"help",        0, 0, 'h'},
    {"write",       1, 0, 'w'},
    {"number",      1, 0, 'n'},
    {"exposure",    1, 0, 'e'},
    {"period",      1, 0, 'E'},
    {"bias",        1, 0, 'b'},
    {"gain",        1, 0, 'g'},
    {"speed",       1, 0, 'S'},
    {"port",        1, 0, 'P'},
    {"host",        1, 0, 'H'},
    {"mac",         1, 0, 'm'},
    {"detip",       1, 0, 'd'},
    {"sls",         1, 0, 's'},
    {"trigger",     1, 0, 't'},
    {"highg0",      0, 0, 'G'},
    {"internal",    0, 0, 'I'},
    {"info",        0, 0, 'i'},
    {"device",      0, 0, 'D'},
    {0,             0, 0,  0 }
  };

  unsigned port  = defPort;
  unsigned num_modules = 0;
  int device = 0;
  int numImages = defNumImages;
  unsigned bias = defBias;
  double exposureTime = defExposureTime;
  double exposurePeriod = defExposurePeriod;
  double triggerDelay = defTriggerDelay;
  bool lUsage = false;
  bool highg0 = false;
  bool internal = false;
  bool show_info = false;
  sls::defs::gainMode gain = sls::defs::DYNAMIC;
  sls::defs::speedLevel speed = sls::defs::HALF_SPEED;
  std::vector<std::string> sHost;
  std::vector<std::string> sMac;
  std::vector<std::string> sDetIp;
  std::vector<std::string> sSlsHost;
  std::string filename = "";

  int optionIndex  = 0;
  while ( int opt = getopt_long(argc, argv, strOptions, loOptions, &optionIndex ) ) {
    if ( opt == -1 ) break;

    switch(opt) {
    case 'h':
      showUsage(argv[0]);
      return 0;
    case 'v':
      showVersion(argv[0]);
      return 0;
    case 'w':
      filename = std::string(optarg) + ".data";
      break;
    case 'n':
      numImages = std::strtoul(optarg, NULL, 0);
      break;
    case 'e':
      exposureTime = std::strtod(optarg, NULL);
      break;
    case 'E':
      exposurePeriod = std::strtod(optarg, NULL);
      break;
    case 'b':
      bias = std::strtoul(optarg, NULL, 0);
      break;
    case 'g':
      {
        unsigned gain_idx = std::strtoul(optarg, NULL, 0);
        if (gain_idx < slsGains.size()) {
          gain = slsGains[gain_idx];
        } else {
          std::cerr << argv[0] << ": Unknown gain setting: " << gain_idx << std::endl;
          std::cerr << "Valid choices are:" << std::endl;
          for (const auto& slsGain : slsGains) {
            std::cerr << " - " << slsGain  << ": " << sls::ToString(slsGain) << std::endl;
          }
          lUsage = true;   
        }
      }
      break;
    case 'S':
      {
        unsigned speed_idx = std::strtoul(optarg, NULL, 0);
        if (speed_idx < slsSpeeds.size()) {
          speed = slsSpeeds[speed_idx];
        } else {
          std::cerr << argv[0] << ": Unknown speed setting: " << speed_idx << std::endl;
          std::cerr << "Valid choices are:" << std::endl;
          for (const auto& slsSpeed : slsSpeeds) {
            std::cerr << " - " << slsSpeed  << ": " << sls::ToString(slsSpeed) << std::endl;
          }
          lUsage = true;
        }
      }
      break;
    case 'H':
      sHost.push_back(optarg);
      break;
    case 'P':
      port = std::strtoul(optarg, NULL, 0);
      break;
    case 'm':
      sMac.push_back(optarg);
      break;
    case 'd':
      sDetIp.push_back(optarg);
      break;
    case 's':
      sSlsHost.push_back(optarg);
      num_modules++;
      break;
    case 't':
      triggerDelay = std::strtod(optarg, NULL);
      break;
    case 'G':
      highg0 = true;
      break;
    case 'I':
      internal = true;
      break;
    case 'i':
      show_info = true;
      break;
    case 'D':
      device = strtol(optarg, NULL, 0);
      break;
    case '?':
      if (optopt)
        std::cerr << argv[0] << ": Unknown option: " << char(optopt) << std::endl;
      else
        std::cerr << argv[0] << ": Unknown option: " << argv[optind-1] << std::endl;
      lUsage = true;
      break;
    case ':':
      std::cerr << argv[0] << ": Missing argument for " << char(optopt) << std::endl;
      lUsage = true;
      break;
    default:
      lUsage = true;
      break;
    }
  }

  if(num_modules == 0) {
    std::cerr << argv[0] << ": at least one module is required" << std::endl;
    lUsage = true;
  }

  if(sHost.size() != num_modules) {
    std::cerr << argv[0] << ": receiver hostname for each module is required" << std::endl;
    lUsage = true;
  }

  if(sMac.size() != num_modules) {
    std::cerr << argv[0] << ": receiver mac address for each module is required" << std::endl;
    lUsage = true;
  }

  if(sDetIp.size() != num_modules) {
    std::cerr << argv[0] << ": detector ip address for each module is required" << std::endl;
    lUsage = true;
  }

  if(sSlsHost.size() != num_modules) {
    std::cerr << argv[0] << ": slsDetector interface hostname for each module is required" << std::endl;
    lUsage = true;
  }

  if (optind < argc) {
    std::cerr << argv[0] << ": invalid argument -- " << argv[optind] << std::endl;
    lUsage = true;
  }

  if (lUsage) {
    showUsage(argv[0]);
    return 1;
  }

  // single Detector class can talk to multiple dets in parallel
  sls::Detector det(device);

  det.setHostname(sSlsHost);

  std::cout << "Using SlsDetector package version: " << det.getPackageVersion() << std::endl;

  auto slsSize = det.size();
  // Check if detector control hostname setup worked
  if (slsSize != (int) num_modules) {
    std::cerr << argv[0] << ": failed to initialize all modules: " << slsSize
              << " versus expected of " << num_modules << std::endl;
  } else {
    std::cout << "Detector contains " << slsSize << " modules" << std::endl;
  }

  auto slsHostnames = det.getHostname();
  auto slsTypes = det.getDetectorType();
  auto slsFirmwareVers = det.getFirmwareVersion();
  auto slsServerVers = det.getDetectorServerVersion();
  auto slsHardwareVers  = det.getHardwareVersion();
  auto slsKernelVers = det.getKernelVersion();
  auto slsModuleId = det.getModuleId();
  auto slsSerialNumber = det.getSerialNumber();

  // construct full serial number used by psana
  DetIdLookup lookup;

  for (int d = 0; d < slsSize; d++) {
    std::cout << "Module " << d << " Info:" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "  Hostname:   " << slsHostnames[d] << std::endl;
    std::cout << "  Det Type:   " << sls::ToString(slsTypes[d]) << std::endl;
    std::cout << "  Firmware:   " << std::hex << slsFirmwareVers[d] << std::endl;
    std::cout << "  Hardware:   " << slsHardwareVers[d] << std::endl;
    std::cout << "  Software:   " << slsServerVers[d] << std::endl;
    std::cout << "  Kernel:     " << slsKernelVers[d] << std::endl;
    std::cout << "  Module id:  " << slsModuleId[d] << std::endl;
    std::cout << "  Serial Num: " << std::hex << slsSerialNumber[d] << std::endl;
    DetId slacSerialNumber;
    if (lookup.has(slsHostnames[d])) {
      slacSerialNumber = DetId(lookup[slsHostnames[d]], slsModuleId[d]);
    } else {
      slacSerialNumber = DetId(slsSerialNumber[d], slsModuleId[d]);
    }
    std::cout << "  SLAC ID:    " << std::hex << slacSerialNumber.full() << std::endl;
  }

}
