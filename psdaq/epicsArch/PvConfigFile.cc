#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <regex>
#include "psalg/utils/SysLog.hh"
#include "PvConfigFile.hh"

using logging = psalg::SysLog;

using std::string;
using std::stringstream;

namespace Drp
{

PvConfigFile::PvConfigFile(const std::string & sFnConfig,
                           const std::string & sProvider,
                           int iMaxDepth,
                           int iMaxNumPv,
                           bool verbose) :
  _sFnConfig(sFnConfig),
  _sProvider(sProvider),
  _iMaxDepth(iMaxDepth),
  _iMaxNumPv(iMaxNumPv),
  _verbose(verbose)
{
}

PvConfigFile::~PvConfigFile()
{
}

int PvConfigFile::read(TPvList & vPvList, std::string & sConfigFileWarning)
{
  _setPvDescription.clear();
  _setPvName       .clear();
  return _readConfigFile(_sFnConfig, vPvList, sConfigFileWarning, _iMaxDepth);
}

int PvConfigFile::_readConfigFile(const std::string & sFnConfig,
                                  TPvList & vPvList, std::string & sConfigFileWarning, int maxDepth)
{
  if (maxDepth <= 0) {
    logging::warning("exceeded maximum include depth (%d)", _iMaxDepth);
    logging::warning("skipping file %s", sFnConfig.c_str());
    return 1;
  }

  std::ifstream ifsConfig(sFnConfig.c_str());
  if (!ifsConfig) {
    logging::error("failed to open file %s", sFnConfig.c_str());
    return 1;     // Cannot open file
  }
  logging::debug("processing file %s", sFnConfig.c_str());

  string sFnPath;
  size_t uOffsetPathEnd = sFnConfig.find_last_of('/');
  if (uOffsetPathEnd != string::npos)
    sFnPath.assign(sFnConfig, 0, uOffsetPathEnd + 1);

  int iLineNumber = 0;

  int nErrors = 0;
  string sPvDescription;
  while (!ifsConfig.eof())
  {
    ++iLineNumber;
    string sLine;
    std::getline(ifsConfig, sLine);

    if (sLine[0] == '*' ||
      (sLine[0] == '#' && sLine.size() > 1 && sLine[1] == '*') ) // alias line
    {
      _getPvDescription(sLine, sPvDescription);
      continue;
    }

    // trim comments that begin with '#'
    _trimComments(sLine);

    if (sLine.empty())
    {
      sPvDescription.clear();
      continue;   // skip empty lines
    }
    if (sLine[0] == '<')
    {
      sLine[0] = ' ';
      PvConfigFile::TFileList vsFileLst;
      _splitFileList(sLine, vsFileLst, _iMaxNumPv);

      for (int iPvFile = 0; iPvFile < (int) vsFileLst.size(); iPvFile++)
      {
        string sFnRef = vsFileLst[iPvFile];
        if (sFnRef[0] != '/')
          sFnRef = sFnPath + sFnRef;
        int iFail = _readConfigFile(sFnRef, vPvList, sConfigFileWarning, maxDepth - 1);
        if (iFail != 0)
        {
          logging::error("Error in file \"%s\", included from \"%s\":line %d",
                         sFnRef.c_str(), sFnConfig.c_str(), iLineNumber);
          return 1;
        }
      }
      continue;
    }

    bool bAddPv;
    int iError = _addPv(sLine, sPvDescription, vPvList, bAddPv, sFnConfig, iLineNumber, sConfigFileWarning);

    if (iError != 0)
      ++nErrors;
    else if (!bAddPv)
      sPvDescription.clear();
  }

  return nErrors;
}

int PvConfigFile::_addPv(const string & sPvLine, string & sPvDescription,
                         TPvList & vPvList, bool & bPvAdd,
                         const std::string& sFnConfig, int iLineNumber, std::string& sConfigFileWarning)
{
  bPvAdd = false;

  const char sPvLineSeparators[] = " ,;\t\r\n";
  size_t uOffsetPv = sPvLine.find_first_not_of(sPvLineSeparators, 0);
  if (uOffsetPv == string::npos)
    return 0;

  if (sPvLine[uOffsetPv] == '#')
    return 0;

  if ((int) vPvList.size() >= _iMaxNumPv)
  {
    logging::error("PV number > maximal allowable value (%d)", _iMaxNumPv);
    return 1;
  }

  string sPvName;
  string sProvider = _sProvider;

  size_t uOffsetSeparator =
    sPvLine.find_first_of(sPvLineSeparators, uOffsetPv + 1);
  if (uOffsetSeparator == string::npos)
  {
    sPvName   = sPvLine.substr(uOffsetPv, string::npos);
  }
  else
  {
    sPvName   = sPvLine.substr(uOffsetPv, uOffsetSeparator - uOffsetPv);
    size_t uOffsetInterval =
      sPvLine.find_first_not_of(sPvLineSeparators, uOffsetSeparator + 1);
    if (uOffsetInterval != string::npos)
    {
      if      (sPvLine.find("ca",  uOffsetInterval) != string::npos)  sProvider = "ca";
      else if (sPvLine.find("pva", uOffsetInterval) != string::npos)  sProvider = "pva";
      else
      {
        char strMessage[256];
        snprintf(strMessage, sizeof(strMessage), "PV %s: Unrecognized provider '%s' (must be 'ca' or 'pva')",
                 sPvName.c_str(), &sPvLine.c_str()[uOffsetInterval]);
        logging::error("%s, in file \"%s\":line %d", strMessage, sFnConfig.c_str(), iLineNumber);
        sConfigFileWarning = strMessage;
        return 0;
      }
    }
  }

  if ( _setPvName.find(sPvName) != _setPvName.end() )
  {
    char strMessage[256];
    snprintf(strMessage, sizeof(strMessage), "Duplicated PV name \"%s\"", sPvName.c_str());
    logging::warning("%s, in file \"%s\":line %d", strMessage, sFnConfig.c_str(), iLineNumber);
    sConfigFileWarning = strMessage;
    return 0;
  }

  if ( _setPvDescription.find(sPvName) != _setPvDescription.end() )
  {
    char strMessage[256];
    snprintf(strMessage, sizeof(strMessage), "PV name \"%s\" was used as another PV's alias", sPvName.c_str());
    logging::error("%s, in file \"%s\":line %d", strMessage, sFnConfig.c_str(), iLineNumber);
    sConfigFileWarning = strMessage;
    return 1;
  }

  string sPvDescriptionUpdate = sPvDescription;
  int iError = _updatePvDescription(sPvName, sFnConfig, iLineNumber, sPvDescriptionUpdate, sConfigFileWarning);

  if ( _setPvName.find(sPvDescriptionUpdate) != _setPvName.end() )
  {
    char strMessage[256];
    snprintf(strMessage, sizeof(strMessage), "Alias %s was used as another PV's name", sPvDescriptionUpdate.c_str());
    logging::error("%s, in file \"%s\":line %d", strMessage, sFnConfig.c_str(), iLineNumber);
    sConfigFileWarning = strMessage;
    return 1;
  }

  _setPvName.insert(sPvName);

  vPvList.push_back(PvConfigFile::PvInfo(sPvName, sPvDescriptionUpdate, sProvider));
  bPvAdd = true;

  return iError;
}

int PvConfigFile::_splitFileList(const std::string & sFileList,
                                 PvConfigFile::TFileList & vsFileList, int iMaxNumPv)
{
  static const char sFileListSeparators[] = " ,;\t\r\n";
  size_t uOffsetStart = sFileList.find_first_not_of(sFileListSeparators, 0);
  while (uOffsetStart != string::npos)
  {
    if (sFileList[uOffsetStart] == '#')
      break;      // skip the remaining characters

    size_t uOffsetEnd =
      sFileList.find_first_of(sFileListSeparators, uOffsetStart + 1);

    if (uOffsetEnd == string::npos)
    {
      if ((int) vsFileList.size() < iMaxNumPv)
        vsFileList.push_back(sFileList.substr(uOffsetStart, string::npos));

      break;
    }

    if ((int) vsFileList.size() < iMaxNumPv)
      vsFileList.push_back(sFileList.substr(uOffsetStart, uOffsetEnd - uOffsetStart));
    else
      break;

    uOffsetStart = sFileList.find_first_not_of(sFileListSeparators, uOffsetEnd + 1);
  }
  return 0;
}

void PvConfigFile::_trimComments(std::string & sLine)
{
  // erase comment beginning with #
  size_t uOffsetComment = sLine.find("#");
  if (uOffsetComment != string::npos) {
    sLine.erase(uOffsetComment);
  }

  // if only whitespace remains, clear the line
  if (sLine.find_first_not_of(" \t") == string::npos) {
    sLine.clear();
  }
}

int PvConfigFile::_getPvDescription(const std::string & sLine, std::string & sPvDescription)
{
  const char sPvDecriptionSeparators[] = " \t*#";
  size_t uOffsetStart = sLine.find_first_not_of(sPvDecriptionSeparators, 0);
  if (uOffsetStart == string::npos)
  {
    sPvDescription.clear();
    return 0;
  }

  size_t uOffsetEnd = sLine.find("#", uOffsetStart+1);
  size_t uOffsetTrail = sLine.find_last_not_of(" \t#", uOffsetEnd );
  if (uOffsetTrail == string::npos)
    sPvDescription.clear();
  else
    sPvDescription = sLine.substr(uOffsetStart, uOffsetTrail - uOffsetStart + 1);

  return 0;
}

int PvConfigFile::_updatePvDescription(const std::string& sPvName, const std::string& sFnConfig, int iLineNumber, std::string& sPvDescription, std::string& sConfigFileWarning)
{
  char strMessage [256];
  char strMessage2[384];
  if (!sPvDescription.empty())
  {
    if ( _setPvDescription.find(sPvDescription) == _setPvDescription.end() )
    {
      if (std::regex_match(sPvDescription, std::regex("[a-zA-Z_][a-zA-Z_0-9]*")))
      {
        _setPvDescription.insert(sPvDescription);
        return 0;
      }
      std::smatch sm;
      std::regex_search(sPvDescription, sm, std::regex("[a-zA-Z_][a-zA-Z_0-9]*"));
      snprintf( strMessage, sizeof(strMessage), "Invalid character '%c' in alias \"%s\"",
                sPvDescription.c_str()[sm.position() != 0 ? 0 : sm.length()], sPvDescription.c_str());
      logging::error("%s, in file \"%s\":line %d\n", strMessage, sFnConfig.c_str(), iLineNumber);
      return 1;
    }
    else
    {
      snprintf( strMessage, sizeof(strMessage), "%s has duplicated alias \"%s\"", sPvName.c_str(), sPvDescription.c_str());
      // This could result in a bad alias:
      //sPvDescription  += '-' + sPvName;
    }
  }
  else
  {
    // This could result in a bad alias:
    //sPvDescription = sPvName;
    snprintf( strMessage, sizeof(strMessage), "%s is missing an alias", sPvName.c_str());
    logging::error("%s, in file \"%s\":line %d\n", strMessage, sFnConfig.c_str(), iLineNumber);
    return 1;
  }

  // This is now obsolete since the alias is no longer updated above
  //if ( _setPvDescription.find(sPvDescription) == _setPvDescription.end() )
  //{
  //  snprintf(strMessage2, sizeof(strMessage2), "%s\nUsing \"%s\"", strMessage, sPvDescription.c_str() );
  //  logging::debug("%s, in file \"%s\":line %d", strMessage2, sFnConfig.c_str(), iLineNumber);
  //  if (sConfigFileWarning.empty())
  //    sConfigFileWarning = strMessage2;
  //
  //  _setPvDescription.insert(sPvDescription);
  //  return 0;
  //}

  static const int iMaxPvSerial = 10000;
  for (int iPvSerial = 2; iPvSerial < iMaxPvSerial; ++iPvSerial)
  {
    stringstream sNumber;
    sNumber << iPvSerial;

    string sPvDesecriptionNew;
    sPvDesecriptionNew = sPvDescription + '_' + sNumber.str();

    if ( _setPvDescription.find(sPvDesecriptionNew) == _setPvDescription.end() )
    {
      sPvDescription = sPvDesecriptionNew;

      snprintf(strMessage2, sizeof(strMessage2), "%s.  Using %s.", strMessage, sPvDescription.c_str() );
      logging::debug("%s, in file \"%s\":line %d\n  Using %s", strMessage, sFnConfig.c_str(), iLineNumber, sPvDescription.c_str() );
      if (sConfigFileWarning.empty())
        sConfigFileWarning = strMessage2;

      _setPvDescription.insert(sPvDescription);
      return 0;
    }
  }

  logging::error("Cannot generate proper PV alias for %s (%s).",
                 sPvDescription.c_str(), sPvName.c_str());

  snprintf(strMessage2, sizeof(strMessage2), "%s.  No proper alias found.", strMessage);
  logging::error("%s, in file \"%s\":line %d\n  No proper alias found.", strMessage, sFnConfig.c_str(), iLineNumber);
  sConfigFileWarning = strMessage2;

  return 1;
}

}       // namespace Drp
