#include "Detector.hh"

#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp::Gpu;
using json = nlohmann::json;

void Drp::Gpu::Detector::rawEvent(Xtc& xtc, const void* bufEnd)
{
  // The raw block is the detector's own array, so it is described by the Names this
  // Detector declared under EventNamesIndex during configure() -- typed and shaped,
  // so that offline sees what the CPU DRP would have written.
  NamesId namesId(nodeId, EventNamesIndex);

  // The Xtc header is built in the CPU's pebble buffer while the payload itself is on
  // the GPU, so bufEnd is set up by the caller to let the allocate below succeed even
  // though the pebble buffer is smaller than header plus data.
  CreateData data(xtc, bufEnd, namesLookup(), namesId);

  unsigned shape[MaxRank] = { 0 };
  auto rank = rawShape(shape);
  if (rank == 0)  shape[0] = rawSize();  // Offered none: describe it as flat bytes
  data.set_array_shape(0, shape);        // Index 0: the Detector's sole raw array
}

json Drp::Gpu::Detector::connectionInfo(const json& msg)
{
  logging::debug("Gpu::Detector::connectionInfo");
  json result{};
  if (m_det) {
    result = m_det->connectionInfo(msg);
  }
  logging::debug("Gpu::Detector::connectionInfo: end");
  return result;
}

void Drp::Gpu::Detector::connectionShutdown()
{
  logging::debug("Gpu::Detector::connectionShutdown");
  if (m_det) {
    m_det->connectionShutdown();
  }
  logging::debug("Gpu::Detector::connectionShutdown: end");
}

void Drp::Gpu::Detector::connect(const json& connect_json, const std::string& collectionId)
{
  logging::debug("Gpu::Detector::connect: this %p, &det %p, det %p", this, &m_det, m_det);
  if (m_det) {
    m_det->nodeId = nodeId;
    m_det->connect(connect_json, collectionId);
  }
  logging::debug("Gpu::Detector::connect: end this %p, &det %p, det %p", this, &m_det, m_det);
}

unsigned Drp::Gpu::Detector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::debug("Gpu::Detector::configure: this %p, &det %p, det %p", this, &m_det, m_det);
  unsigned rc{0};

  // Do configure
  //printf("*** Gpu::Detector::configure start: this %p &det %p, det %p\n", this, &m_det, m_det);
  if (m_det) {
    rc = m_det->configure(config_alias, xtc, bufEnd);
    //printf("*** Gpu::Detector configure done: rc %d, sz %u\n", rc, xtc.sizeofPayload());
    if (rc) {
      logging::error("Gpu::Detector::configure failed for %s\n", m_para->device);
    }
  }

  logging::debug("Gpu::Detector::configure: end");
  return rc;
}

unsigned Drp::Gpu::Detector::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  logging::debug("Gpu::Detector::beginrun");
  unsigned rc{0};

  // Do beginRun
  if (m_det) {
    rc = m_det->beginrun(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::Detector::beginrun failed for %s\n", m_para->device);
    }
  }

  logging::debug("Gpu::Detector::beginrun: end");
  return rc;
}

unsigned Drp::Gpu::Detector::beginstep(Xtc& xtc, const void* bufEnd, const json& stepInfo)
{
  logging::debug("Gpu::Detector::beginstep");
  unsigned rc{0};

  // Do beginStep
  if (m_det) {
    rc = m_det->beginstep(xtc, bufEnd, stepInfo);
    if (rc) {
      logging::error("Gpu::Detector::beginstep failed for %s\n", m_para->device);
    }
  }

  logging::debug("Gpu::Detector::beginstep: end");
  return rc;
}

unsigned Drp::Gpu::Detector::enable(Xtc& xtc, const void* bufEnd, const json& info)
{
  logging::debug("Gpu::Detector::enable");
  unsigned rc{0};

  // Do Enable
  if (m_det) {
    rc = m_det->enable(xtc, bufEnd, info);
    if (rc) {
      logging::error("Gpu::Detector::enable failed for %s\n", m_para->device);
    }
  }

  logging::debug("Gpu::Detector::enable: end");
  return rc;
}

unsigned Drp::Gpu::Detector::disable(Xtc& xtc, const void* bufEnd, const json& info)
{
  logging::debug("Gpu::Detector::disable");
  unsigned rc{0};

  // Do Disable
  if (m_det) {
    rc = m_det->disable(xtc, bufEnd, info);
    if (rc) {
      logging::error("Gpu::Detector::disable failed for %s\n", m_para->device);
    }
  }

  logging::debug("Gpu::Detector::disable: end");
  return rc;
}

void Drp::Gpu::Detector::shutdown()
{
  if (m_det) {
    m_det->shutdown();
  }
}
