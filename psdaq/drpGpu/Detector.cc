#include "Detector.hh"

#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;
using json = nlohmann::json;

json Drp::Gpu::Detector::connectionInfo(const json& msg)
{
  logging::debug("Gpu::Detector::connectionInfo");
  json result;              // @todo: how to handle more than one?
  for (const auto& det : m_dets) {
    result = det->connectionInfo(msg);
  }
  logging::debug("Gpu::Detector::connectionInfo: end");
  return result;
}

void Drp::Gpu::Detector::connectionShutdown()
{
  logging::debug("Gpu::Detector::connectionShutdown");
  for (const auto& det : m_dets) {
    det->connectionShutdown();
  }
  logging::debug("Gpu::Detector::connectionShutdown: end");
}

void Drp::Gpu::Detector::connect(const json& connect_json, const std::string& collectionId)
{
  logging::debug("Gpu::Detector::connect");
  for (const auto& det : m_dets) {
    det->nodeId = nodeId;
    det->connect(connect_json, collectionId);
  }
  logging::debug("Gpu::Detector::connect: end");
}

unsigned Drp::Gpu::Detector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::debug("Gpu::Detector::configure");
  unsigned rc = 0;

  // Configure each panel in turn
  // @todo: Do we really want to extend the Xtc for each panel, or does one speak for all?
  // @todo: Calling the CPU Detector's configure for each panel seems wrong as that nominally
  //        also sets up the Xtc for the CPU Detector's data, which is different from the
  //        GPU Detector's data.  Get rid of this method to force it to be overridden by the
  //        GPU Detector's implementation?
  unsigned i = 0;
  for (const auto& det : m_dets) {
    printf("*** Gpu::Detector configure for %u start\n", i);
    rc = det->configure(config_alias, xtc, bufEnd);
    printf("*** Gpu::Detector configure for %u done: rc %d, sz %u\n", i, rc, xtc.sizeofPayload());
    if (rc) {
      logging::error("Gpu::Detector::configure failed for %s\n", m_params[i].device);
      break;
    }
    ++i;
  }

  logging::debug("Gpu::Detector::configure: end");
  return rc;
}

unsigned Drp::Gpu::Detector::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  logging::debug("Gpu::Detector::beginrun");
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

  logging::debug("Gpu::Detector::beginrun: end");
  return rc;
}

unsigned Drp::Gpu::Detector::beginstep(Xtc& xtc, const void* bufEnd, const json& stepInfo)
{
  logging::debug("Gpu::Detector::beginstep");
  // Do beginStep for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->beginstep(xtc, bufEnd, stepInfo);
    if (rc) {
      logging::error("Gpu::Detector::beginstep failed for %s\n", m_params[i].device);
      break;
    }
  }

  logging::debug("Gpu::Detector::beginstep: end");
  return rc;
}

unsigned Drp::Gpu::Detector::enable(Xtc& xtc, const void* bufEnd, const json& info)
{
  logging::debug("Gpu::Detector::enable");
  // Do Enable for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->enable(xtc, bufEnd, info);
    if (rc) {
      logging::error("Gpu::Detector::enable failed for %s\n", m_params[i].device);
      break;
    }
  }

  logging::debug("Gpu::Detector::enable: end");
  return rc;
}

unsigned Drp::Gpu::Detector::disable(Xtc& xtc, const void* bufEnd, const json& info)
{
  logging::debug("Gpu::Detector::disable");
  // Do Disable for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->disable(xtc, bufEnd, info);
    if (rc) {
      logging::error("Gpu::Detector::disable failed for %s\n", m_params[i].device);
      break;
    }
  }

  logging::debug("Gpu::Detector::disable: end");
  return rc;
}

void Drp::Gpu::Detector::shutdown()
{
  for (const auto& det : m_dets) {
    det->shutdown();
  }
}
