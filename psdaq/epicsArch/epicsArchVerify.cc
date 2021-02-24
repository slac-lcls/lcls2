#include <getopt.h>
#include <string>
#include "PvConfigFile.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

static const int iMaxNumPv = 10000;


namespace Drp {

int verify(const std::string& sFnConfig,
           int                iDebugLevel,
           std::string&       sConfigFileWarning)
{
  PvConfigFile::TPvList vPvList;
  int iMaxDepth = 10;
  std::string sProvider = "ca";
  PvConfigFile configFile(sFnConfig, sProvider, iMaxDepth, iMaxNumPv, (iDebugLevel >= 1));
  int iFail = configFile.read(vPvList, sConfigFileWarning);

  logging::debug("PVs:");
  for (unsigned iPv = 0; iPv < vPvList.size(); iPv++)
    logging::debug("  [%3d] %-32s PV %-32s Provider '%s'", iPv,
                   vPvList[iPv].sPvDescription.c_str(),
                   vPvList[iPv].sPvName.c_str(),
                   vPvList[iPv].sProvider.c_str());

  if (iFail != 0) {
    logging::critical("configFile(%s).read() failed", sFnConfig.c_str());
    throw "Reading config file failed";
  }

  if (vPvList.empty()) {
    logging::critical("No PV is specified in the config file %s", sFnConfig.c_str());
    throw "Empty config file";
  }

  return 0;
}

} // namespace Drp


int main(int argc, char* argv[])
{
  int verbose = 0;

  int c;
  while((c = getopt(argc, argv, "v")) != EOF) {
    switch(c) {
      case 'v':  ++verbose;  break;
      default:
        printf("%s "
               "[-v] "
               "<configuration filename>\n", argv[0]);
        return 1;
    }
  }

  switch (verbose) {
    case 0:  logging::init("tst", LOG_INFO);   break;
    default: logging::init("tst", LOG_DEBUG);  break;
  }

  std::string pvCfgFile;
  if (optind < argc)
    pvCfgFile = argv[optind];
  else {
    logging::critical("A PV config filename is mandatory");
    return 1;
  }

  try {
    std::string configFileWarning;
    Drp::verify(pvCfgFile, verbose, configFileWarning);
    if (!configFileWarning.empty()) {
      printf("%s: %s\n", argv[0], configFileWarning.c_str());
    }
    return 0;
  }
  catch (std::exception& e)  { logging::critical("%s", e.what()); }
  catch (std::string& e)     { logging::critical("%s", e.c_str()); }
  catch (char const* e)      { logging::critical("%s", e); }
  catch (...)                { logging::critical("Default exception"); }
  return EXIT_FAILURE;
}
