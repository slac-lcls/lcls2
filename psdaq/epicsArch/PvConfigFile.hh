#ifndef _PV_CONFIG_FILE_H
#define _PV_CONFIG_FILE_H

#include <string>
#include <vector>
#include <set>

namespace Pds
{
  class PvConfigFile
  {
  public:
    PvConfigFile(const std::string & sFnConfig, bool bProviderType, int iMaxDepth, int iMaxNumPv,
                 bool verbose);
    ~PvConfigFile();
  public:

    typedef std::set < std::string > TPvNameSet;

    TPvNameSet  _setPvName;
    TPvNameSet  _setPvDescription;
    std::string _sFnConfig;
    bool        _bProviderType;
    int         _iMaxDepth;
    int         _iMaxNumPv;
    bool        _verbose;

    struct PvInfo
    {
      std::string sPvName;
      std::string sPvDescription;
      bool bProviderType;

        PvInfo(const std::string & sPvName1,
               const std::string & sPvDescription1,
               bool bProviderType1):sPvName(sPvName1),
                                    sPvDescription(sPvDescription1), bProviderType(bProviderType1)
      {
      }
    };

    typedef std::vector < PvInfo >      TPvList;
    typedef std::vector < std::string > TFileList;

    int read(TPvList & vPvList, std::string& sConfigFileWarning);

  private:

    int _addPv               (const std::string & sPvList, std::string & sPvDescription,
                              TPvList & vPvList, bool & bPvAdd,
                              const std::string & sFnConfig, int iLineNumber, std::string & sConfigFileWarning);
    int _readConfigFile      (const std::string & sFnConfig, TPvList & vPvList, std::string & sConfigFileWarning, int iMaxDepth);
    int _getPvDescription    (const std::string & sLine, std::string & sPvDescription);
    int _updatePvDescription (const std::string & sPvName, const std::string & sFnConfig, int iLineNumber, std::string & sPvDescription, std::string & sConfigFileWarning);

    void _trimComments       (std::string & sLine);
    static int _splitFileList(const std::string & sFileList, TFileList & vsFileList, int iMaxNumPv);

    // Class usage control: Value semantics is disabled
    PvConfigFile(const PvConfigFile &);
    PvConfigFile & operator=(const PvConfigFile &);
  };
}       // namespace Pds

#endif
