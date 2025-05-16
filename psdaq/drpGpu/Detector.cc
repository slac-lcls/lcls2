#include "Detector.hh"

#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;
using json = nlohmann::json;

json Drp::Gpu::Detector::connectionInfo(const json& msg)
{
  json result;              // @todo: how to handle more than one?
  for (const auto& det : m_dets) {
    result = det->connectionInfo(msg);
  }
  return result;
}

void Drp::Gpu::Detector::connectionShutdown()
{
  for (const auto& det : m_dets) {
    det->connectionShutdown();
  }
}

void Drp::Gpu::Detector::connect(const json& connect_json, const std::string& collectionId)
{
  for (const auto& det : m_dets) {
    det->nodeId = nodeId;
    det->connect(connect_json, collectionId);
  }
}

unsigned Drp::Gpu::Detector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  printf("*** Gpu::Detector configure 1\n");
  unsigned rc = 0;

  // Configure each panel in turn
  unsigned i = 0;
  for (const auto& det : m_dets) {
    printf("*** Gpu::AreaDetector configure for %u start\n", i);
    rc = det->configure(config_alias, xtc, bufEnd);
    printf("*** Gpu::AreaDetector configure for %u done: rc %d\n", i, rc);
    if (rc) {
      logging::error("Gpu::Detector::configure failed for %s\n", m_params[i].device);
      break;
    }
    ++i;
  }

  return rc;
}

unsigned Drp::Gpu::Detector::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  // Do beginRun for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->beginrun(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::Detector::beginrun failed for %s\n", m_params[i].device);
      break;
    }
  }

  return rc;
}

unsigned Drp::Gpu::Detector::beginstep(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  // Do beginStep for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->beginstep(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::Detector::beginstep failed for %s\n", m_params[i].device);
      break;
    }
  }

  return rc;
}

unsigned Drp::Gpu::Detector::enable(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  // Do Enable for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->enable(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::Detector::enable failed for %s\n", m_params[i].device);
      break;
    }
  }

  return rc;
}

unsigned Drp::Gpu::Detector::disable(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  // Do Disable for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->disable(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::Detector::disable failed for %s\n", m_params[i].device);
      break;
    }
  }

  return rc;
}

void Drp::Gpu::Detector::shutdown()
{
  for (const auto& det : m_dets) {
    det->shutdown();
  }
}
