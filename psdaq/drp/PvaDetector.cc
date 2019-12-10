#include "PvaDetector.hh"

#include <cassert>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include "DataDriver.h"
#include "RunInfoDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"
#include <getopt.h>
#include <Python.h>


using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

static const XtcData::Name::DataType xtype[] = {
    XtcData::Name::UINT8 , // pvBoolean
    XtcData::Name::INT8  , // pvByte
    XtcData::Name::INT16 , // pvShort
    XtcData::Name::INT32 , // pvInt
    XtcData::Name::INT64 , // pvLong
    XtcData::Name::UINT8 , // pvUByte
    XtcData::Name::UINT16, // pvUShort
    XtcData::Name::UINT32, // pvUInt
    XtcData::Name::UINT64, // pvULong
    XtcData::Name::FLOAT , // pvFloat
    XtcData::Name::DOUBLE, // pvDouble
    XtcData::Name::CHARSTR, // pvString
};

void PvaMonitor::printStructure()
{
    const pvd::StructureConstPtr& structure = _strct->getStructure();
    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    for (unsigned i=0; i<names.size(); i++) {
      logging::info("%s: FieldName:  %s  FieldType:  %s",
                    name().c_str(), names[i].c_str(), pvd::TypeFunc::name(fields[i]->getType()));
    }
}

XtcData::VarDef PvaMonitor::get(size_t& payloadSize)
{
    payloadSize = 0;
    XtcData::VarDef vd;
    const pvd::StructureConstPtr& structure = _strct->getStructure();
    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    for (unsigned i=0; i<fields.size(); i++) {
        if (names[i] != "value")  continue;
        std::string fullName(name() + "." + names[i]);
        switch (fields[i]->getType()) {
            case pvd::scalar: {
                const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
                XtcData::Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type));
                payloadSize += XtcData::Name::get_element_size(type);
                logging::info("name: %s  type: %d", fullName.c_str(), type);
                switch (scalar->getScalarType()) {
                    case pvd::pvInt:    getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<int32_t >(data, length); };  break;
                    case pvd::pvLong:   getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<int64_t >(data, length); };  break;
                    case pvd::pvUInt:   getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<uint32_t>(data, length); };  break;
                    case pvd::pvULong:  getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<uint64_t>(data, length); };  break;
                    case pvd::pvFloat:  getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<float   >(data, length); };  break;
                    case pvd::pvDouble: getData = [&](void* data, size_t& length) -> size_t { return _getDatumT<double  >(data, length); };  break;
                    default: {
                        logging::critical("%s: Unsupported Scalar type %d",
                                          fullName.c_str(),
                                          scalar->getScalarType());
                        throw "Unsupported scalar type";
                        break;
                    }
                }
                break;
            }
            case pvd::scalarArray: {
                const pvd::ScalarArray* array = static_cast<const pvd::ScalarArray*>(fields[i].get());
                XtcData::Name::DataType type = xtype[array->getElementType()];
                size_t length = _strct->getSubField<pvd::PVArray>(names[i].c_str())->getLength();
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type, 1));
                payloadSize += length * XtcData::Name::get_element_size(type);
                logging::info("name: %s  type: %d  length: %zd", fullName.c_str(), type, length);
                switch (array->getElementType()) {
                    case pvd::pvInt:    getData = [&](void* data, size_t& length) -> size_t { return _getDataT<int32_t >(data, length); };  break;
                    case pvd::pvLong:   getData = [&](void* data, size_t& length) -> size_t { return _getDataT<int64_t >(data, length); };  break;
                    case pvd::pvUInt:   getData = [&](void* data, size_t& length) -> size_t { return _getDataT<uint32_t>(data, length); };  break;
                    case pvd::pvULong:  getData = [&](void* data, size_t& length) -> size_t { return _getDataT<uint64_t>(data, length); };  break;
                    case pvd::pvFloat:  getData = [&](void* data, size_t& length) -> size_t { return _getDataT<float   >(data, length); };  break;
                    case pvd::pvDouble: getData = [&](void* data, size_t& length) -> size_t { return _getDataT<double  >(data, length); };  break;
                    default: {
                        logging::critical("%s: Unsupported ScalarArray type %d",
                                          fullName.c_str(),
                                          array->getElementType());
                        throw "Unsupported ScalarArray type";
                        break;
                    }
                }
                break;
            }
            default: {
                logging::critical("%s: Unsupported field type '%s'",
                                  fullName.c_str(),
                                  pvd::TypeFunc::name(fields[i]->getType()));
                throw "Unsupported field type";
                break;
            }
        }
    }
    return vd;
}

void PvaMonitor::updated()
{
    //std::cout<<"updated\n";

    m_app.process(*this);
}


class Pgp
{
public:
    Pgp(MemPool& pool, unsigned nodeId, uint32_t envMask) :
        m_pool(pool), m_nodeId(nodeId), m_envMask(envMask), m_available(0), m_current(0)
    {
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<4; i++) {
            dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
        }
        dmaSetMaskBytes(pool.fd(), mask);
    }

    Pds::EbDgram* next(uint32_t& evtIndex);
private:
    Pds::EbDgram* _handle(Pds::TimingHeader* timingHeader, uint32_t& evtIndex);
    MemPool& m_pool;
    unsigned m_nodeId;
    uint32_t m_envMask;
    int32_t m_available;
    int32_t m_current;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
};

Pds::EbDgram* Pgp::_handle(Pds::TimingHeader* timingHeader, uint32_t& evtIndex)
{
    int32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    if (unsigned(size) > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d\n", size, m_pool.dmaSize());
        exit(-1);
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    evtIndex = evtCounter & (m_pool.nbuffers() - 1);
    PGPEvent* event = &m_pool.pgpEvents[evtIndex];

    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    // move the control bits from the pulseId into the
    // top 8 bits of env.
    unsigned control = timingHeader->timing_control();
    timingHeader->env = (timingHeader->env&0xffffff)|(control<<24);

    // make new dgram in the pebble
    Pds::EbDgram* dgram = new(m_pool.pebble[evtIndex]) Pds::EbDgram(*timingHeader, XtcData::Src(m_nodeId), m_envMask);

    return dgram;
}

Pds::EbDgram* Pgp::next(uint32_t& evtIndex)
{
    // get new buffers
    if (m_current == m_available) {
        m_current = 0;
        auto start = std::chrono::steady_clock::now();
        while (true) {
            m_available = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
            if (m_available > 0) {
                break;
            }

            // wait for a total of 10 ms otherwise timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > 10) {
                // printf("pgp timeout\n");
                return nullptr;
            }
        }
    }

    Pds::TimingHeader* timingHeader = (Pds::TimingHeader*)m_pool.dmaBuffers[dmaIndex[m_current]];

    Pds::EbDgram* dgram = _handle(timingHeader, evtIndex);
    m_current++;
    return dgram;
}

PvaApp::PvaApp(Parameters& para, const std::string& pvName) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_pvName(pvName),
    m_inputQueue(m_drp.pool.nbuffers()),
    m_swept(false),
    m_terminate(false)
{
    logging::info("Ready for transitions");
}

void PvaApp::_shutdown()
{
    m_exporter.reset();

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_pvaMonitor.reset();
    m_drp.shutdown();
    m_namesLookup.clear();   // erase all elements
}

json PvaApp::connectionInfo()
{
    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo); // Revisit: Should be in det_info
    return body;
}
void PvaApp::_error(const std::string& which, const nlohmann::json& msg, const std::string& errorMsg)
{
    json body = json({});
    body["err_info"] = errorMsg;
    json answer = createMsg(which, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvaApp::handleConnect(const nlohmann::json& msg)
{
    std::string errorMsg = m_drp.connect(msg, getId());
    if (!errorMsg.empty()) {
        logging::error("Error in DrpBase::connect");
        logging::error("%s", errorMsg.c_str());
        _error("connect", msg, errorMsg);
        return;
    }

    m_pvaMonitor = std::make_unique<PvaMonitor>(m_pvName.c_str(), *this);

    auto start = std::chrono::steady_clock::now();
    while(true) {                       // Revisit: Time this out
        if (m_pvaMonitor->connected()) {
            m_pvaMonitor->printStructure();
            break;
        }
        usleep(100000);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > 5000) {
            logging::error("Failed to connect with %s", m_pvName.c_str());
            _error("connect", msg, "Failed to connect with " + m_pvName);
            return;
        }
    }

    _connectPgp(msg, std::to_string(getId()));

    m_unconfigure = false;

    json body = json({});
    json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvaApp::handleDisconnect(const json& msg)
{
    _shutdown();

    json body = json({});
    reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void PvaApp::handlePhase1(const json& msg)
{
    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in PvaApp", key.c_str());

    if (key == "configure") {
        if (m_unconfigure) {
            _shutdown();
            m_unconfigure = false;
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            _error(key, msg, errorMsg);
            logging::error("%s", errorMsg.c_str());
        }
        else {
            m_exporter = std::make_shared<MetricExporter>();
            if (m_drp.exposer()) {
                m_drp.exposer()->RegisterCollectable(m_exporter);
            }

            m_swept.store(false, std::memory_order_release);
            m_terminate.store(false, std::memory_order_release);

            m_workerThread = std::thread{&PvaApp::_worker, this, m_exporter};
        }
    }
    else if (key == "unconfigure") {
        // Delay unconfiguration until after phase 2 of unconfigure has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        std::string errorMsg = m_drp.beginrun(phase1Info, m_runInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
    }

    json answer = createMsg(key, msg["header"]["msg_id"], getId(), body);
    reply(answer);
}

void PvaApp::handleReset(const nlohmann::json& msg)
{
    _shutdown();
}

void PvaApp::_connectPgp(const json& json, const std::string& collectionId)
{
    // FIXME not sure what the size should be since for this DRP we expect no PGP payload
    int length = 0;
    int links = m_para.laneMask;

    int fd = open(m_para.device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para.device.c_str());
    }

    int readoutGroup = json["body"]["drp"][collectionId]["det_info"]["readout"];
    uint32_t v = ((readoutGroup&0xf)<<0) |
                  ((length&0xffffff)<<4) |
                  (links<<28);
    dmaWriteRegister(fd, 0x00a00000, v);
    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    logging::info("Configured readout group [%u], length [%u], links [%x]: [%x](%x)",
           readoutGroup, length, links, v, w);
    for (unsigned i=0; i<4; i++) {
        if (links&(1<<i)) {
            // this is the threshold to assert deadtime (high water mark) for every link
            // 0x1f00 corresponds to 0x1f free buffers
            dmaWriteRegister(fd, 0x00800084+32*i, 0x1f00);
        }
    }
    close(fd);
}

void PvaApp::_worker(std::shared_ptr<MetricExporter> exporter)
{
    size_t payloadSize;
    XtcData::VarDef pvaDef = m_pvaMonitor->get(payloadSize);
    logging::debug("payloadSize %zd", payloadSize);
    if (payloadSize > m_drp.pool.pebble.bufferSize()) {
        logging::critical("Event buffer size (%zd) is too small for %s payload (%zd)",
                          m_drp.pool.pebble.bufferSize(), m_pvName, payloadSize);
        exit(-1);
    }

    Pgp pgp(m_drp.pool, m_drp.nodeId(), 0xffff0000 | uint32_t(m_para.rogMask));

    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para.partition)}};
    m_nEvents = 0;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nEvents;});
    m_nUpdates = 0;
    exporter->add("pva_update_rate", labels, MetricType::Rate,
                  [&](){return m_nUpdates;});
    m_nMissed = 0;
    exporter->add("pva_miss_count", labels, MetricType::Counter,
                  [&](){return m_nMissed;});
    m_nEmpty = 0;
    exporter->add("pva_empty_count", labels, MetricType::Counter,
                  [&](){return m_nEmpty;});
    m_nTooOld = 0;
    exporter->add("pva_tooOld_count", labels, MetricType::Counter,
                  [&](){return m_nTooOld;});

    exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
                  [&](){return m_inputQueue.guess_size();});

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        // Drain PGP to avoid inducing backpressure
        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index);
        if (dgram) {
            XtcData::TransitionId::Value service = dgram->service();
            if ((service == XtcData::TransitionId::L1Accept) ||
                (service == XtcData::TransitionId::SlowUpdate)) {
                m_inputQueue.push(index);
            }
            else {
                // Construct the transition in its own buffer from the PGP Dgram
                Pds::EbDgram* trDgram = m_drp.pool.transitionDgram();
                *trDgram = *dgram;

                switch (service) {
                    case XtcData::TransitionId::Configure: {
                        logging::info("PVA configure");

                        XtcData::Alg pvaAlg("pvaAlg", 1, 2, 3);
                        XtcData::NamesId pvaNamesId(m_drp.nodeId(), PvaNamesIndex);
                        XtcData::Names& pvaNames = *new(trDgram->xtc) XtcData::Names("pva", pvaAlg,
                                                                                     "pva", "pva1234", pvaNamesId);
                        pvaNames.add(trDgram->xtc, pvaDef);
                        m_namesLookup[pvaNamesId] = XtcData::NameIndex(pvaNames);

                        m_drp.runInfoSupport(trDgram->xtc, m_namesLookup);
                        break;
                    }
                    case XtcData::TransitionId::BeginRun: {
                        if (m_runInfo.runNumber > 0) {
                            m_drp.runInfoData(trDgram->xtc, m_namesLookup, m_runInfo);
                        }
                        break;
                    }
                    case XtcData::TransitionId::Disable: { // Sweep out L1As
                        m_inputQueue.push(index);
                        std::unique_lock<std::mutex> lock(_lock);
                        std::chrono::milliseconds tmo(100);
                        _cv.wait_for(lock, tmo, [this] { return m_swept.load(std::memory_order_relaxed); });
                        if (!m_swept.load(std::memory_order_relaxed)) { // If timed out
                            while (true) { // Post everything still on the queue
                                uint32_t idx;
                                if (!m_inputQueue.try_pop(idx)) {
                                    break;
                                }
                                Pds::EbDgram* dg = (Pds::EbDgram*)m_drp.pool.pebble[idx];
                                _sendToTeb(*dg, idx);
                                m_nEvents++;
                            }
                        }
                        break;
                    }
                    default: {              // Handle other transitions
                        break;
                    }
                }

                // Make sure the transition didn't get too big
                size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                if (size > m_para.maxTrSize) {
                    logging::critical("Transition: buffer size (%zd) too small for Dgram (%zd)", m_para.maxTrSize, size);
                    exit(-1);
                }

                _sendToTeb(*dgram, index);
                m_nEvents++;
            }
        }
    }
    logging::info("Worker thread finished");
}

void PvaApp::process(const PvaMonitor& pva)
{
    unsigned seconds = pva.getScalarAs<unsigned>("timeStamp.secondsPastEpoch");
    unsigned nanoseconds = pva.getScalarAs<unsigned>("timeStamp.nanoseconds");
    XtcData::TimeStamp timestamp(seconds - (20*365+5)*24*3600, // Convert from 1/1/70 to 1/1/90 epoch with 5 leap years
                                 nanoseconds);
    ++m_nUpdates;
    bool retried = false;
    while (true) {
        uint32_t index;
        if (!m_inputQueue.peek(index)) {
            retried = true;
            if (m_terminate.load(std::memory_order_relaxed)) {
                return;                 // Return b/c index is not valid
            }
            continue;
        }
        if (retried) {
            ++m_nMissed;      // Count number of times a PV had to wait for PGP
            retried = false;
        }

        Pds::EbDgram* dgram = (Pds::EbDgram*)m_drp.pool.pebble[index];
        if (dgram->service() == XtcData::TransitionId::Disable) {
            uint32_t idx;
            m_inputQueue.try_pop(idx);  // Actually consume the element
            assert(idx == index);

            std::lock_guard<std::mutex> lock(_lock);
            m_swept.store(true, std::memory_order_release);
            _cv.notify_one();
            break;
        }
        else if (timestamp == dgram->time) {
            uint32_t idx;
            m_inputQueue.try_pop(idx);  // Actually consume the element
            assert(idx == index);

            logging::debug("PV matches PGP!!  "
                           "TimeStamp PV %d.%09d | PGP %d.%09d\n",
                           timestamp.seconds(), timestamp.nanoseconds(),
                           dgram->time.seconds(), dgram->time.nanoseconds());

            XtcData::NamesId namesId(m_drp.nodeId(), PvaNamesIndex);
            XtcData::DescribedData desc(dgram->xtc, m_namesLookup, namesId);
            size_t length;
            size_t size = pva.getData(desc.data(), length);
            desc.set_data_length(size);
            unsigned shape[] = { unsigned(length) };
            desc.set_array_shape(0, shape);
            //size_t sz = (sizeof(*dgram) + dgram->xtc.sizeofPayload()) >> 2;
            //uint32_t* payload = (uint32_t*)dgram->xtc.payload();
            //printf("sz = %zd, size = %zd, extent = %d, szofPyld = %d, pyldIdx = %ld\n", sz, size, dgram->xtc.extent, dgram->xtc.sizeofPayload(), payload - (uint32_t*)dgram);
            //uint32_t* buf = (uint32_t*)dgram;
            //for (unsigned i = 0; i < sz; ++i) {
            //  if (&buf[i] == (uint32_t*)dgram)        printf(  "dgram:   ");
            //  if (&buf[i] == (uint32_t*)payload)      printf("\npayload: ");
            //  if (&buf[i] == (uint32_t*)desc.data())  printf("\ndata:    ");
            //  printf("%08x ", buf[i]);
            //}
            //printf("\n");

            _sendToTeb(*dgram, index);
            m_nEvents++;
            break;
        }
        // No PVA data for PGP timestamp so forward empty event
        else if (timestamp > dgram->time) {
            uint32_t idx;
            m_inputQueue.try_pop(idx);  // Actually consume the element
            assert(idx == index);

            // No PVA data so mark event as damaged
            dgram->xtc.damage.increase(XtcData::Damage::MissingData);

            ++m_nEmpty;
            logging::debug("No PV data!!      "
                           "TimeStamp PV %d.%09d | PGP %d.%09d\n",
                           timestamp.seconds(), timestamp.nanoseconds(),
                           dgram->time.seconds(), dgram->time.nanoseconds());
            _sendToTeb(*dgram, index);
            m_nEvents++;
            // Keep processing PGP events until a match is found
        }
        // The PVA timestamp is older than the earliest PGP event, so skip
        else {
            ++m_nTooOld;
            logging::debug("PV too old!!      "
                           "TimeStamp PV %d.%09d | PGP %d.%09d\n",
                           timestamp.seconds(), timestamp.nanoseconds(),
                           dgram->time.seconds(), dgram->time.nanoseconds());
            break;
        }
    }
}

void PvaApp::_sendToTeb(Pds::EbDgram& dgram, uint32_t index)
{
    void* buffer = m_drp.tebContributor().allocate(&dgram, (void*)((uintptr_t)index));
    if (buffer) { // else timed out
        PGPEvent* event = &m_drp.pool.pgpEvents[index];
        event->l3InpBuf = buffer;
        Pds::EbDgram* l3InpDg = new(buffer) Pds::EbDgram(dgram);
        if (dgram.isEvent()) {
            if (m_drp.triggerPrimitive()) {// else this DRP doesn't provide input
                m_drp.triggerPrimitive()->event(m_drp.pool, index, dgram.xtc, l3InpDg->xtc); // Produce
            }
        }
        m_drp.tebContributor().process(l3InpDg);
    }
}

} // namespace Drp


int main(int argc, char* argv[])
{
    Drp::Parameters para;
    para.partition = -1;
    para.laneMask = 0x1;
    para.detName = "pva";               // Revisit: Should come from alias?
    para.detSegment = 0;
    para.verbose = 0;
    char *instrument = NULL;
    int c;
    while((c = getopt(argc, argv, "p:o:C:d:u:P:T::v")) != EOF) {
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
            case 'd':
                para.device = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            case 'P':
                instrument = optarg;
                break;
            case 'T':
                para.trgDetName = optarg ? optarg : "trigger";
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                exit(1);
        }
    }

    switch (para.verbose) {
        case 0:  logging::init(instrument, LOG_INFO);     break;
        default: logging::init(instrument, LOG_DEBUG);    break;
    }
    logging::info("logging configured");
    if (!instrument) {
        logging::warning("-P: instrument name is missing");
    }
    // Check required parameters
    if (para.partition == unsigned(-1)) {
        logging::critical("-p: partition is mandatory");
        exit(1);
    }
    if (para.device.empty()) {
        logging::critical("-d: device is mandatory");
        exit(1);
    }
    if (para.alias.empty()) {
        logging::critical("-u: alias is mandatory");
        exit(1);
    }

    // Alias must be of form <detName>_<detSegment>
    size_t found = para.alias.rfind('_');
    if ((found == std::string::npos) || !isdigit(para.alias.back())) {
        logging::critical("-u: alias must have _N suffix");
        exit(1);
    }
    para.detName = para.alias.substr(0, found);
    para.detSegment = std::stoi(para.alias.substr(found+1, para.alias.size()));

    std::string pvName;
    if (optind < argc)
        pvName = argv[optind];
    else {
        logging::critical("A PV name is mandatory");
        exit(1);
    }

    para.maxTrSize = 256 * 1024;

    Py_Initialize(); // for use by configuration
    Drp::PvaApp app(para, pvName);
    app.run();
    app.handleReset(json({}));
    Py_Finalize(); // for use by configuration
}
