#include "PvaDetector.hh"

#include <getopt.h>
#include <cassert>
#include <chrono>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <Python.h>
#include "DataDriver.h"
#include "RunInfoDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psalg/utils/SysLog.hh"

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
    XtcData::VarDef vd;
    const pvd::StructureConstPtr& structure = _strct->getStructure();
    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    unsigned i;
    for (i=0; i<fields.size(); i++) {
        if (names[i] == "value")  break;
    }
    std::string fullName(name() + "." + names[i]);
    switch (fields[i]->getType()) {
        case pvd::scalar: {
            const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
            XtcData::Name::DataType type = xtype[scalar->getScalarType()];
            vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type));
            payloadSize = XtcData::Name::get_element_size(type);
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
                }
            }
            break;
        }
        case pvd::scalarArray: {
            const pvd::ScalarArray* array = static_cast<const pvd::ScalarArray*>(fields[i].get());
            XtcData::Name::DataType type = xtype[array->getElementType()];
            size_t length = _strct->getSubField<pvd::PVArray>(names[i].c_str())->getLength();
            vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type, 1));
            payloadSize = length * XtcData::Name::get_element_size(type);
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
                }
            }
            break;
        }
        default: {
            logging::critical("%s: Unsupported field type '%s'",
                              fullName.c_str(),
                              pvd::TypeFunc::name(fields[i]->getType()));
            throw "Unsupported field type";
        }
    }

    return vd;
}

void PvaMonitor::updated()
{
    m_pvaDetector.process(*this);
}


class Pgp
{
public:
    Pgp(const Parameters& para, DrpBase& drp, const bool& running) :
        m_para(para), m_pool(drp.pool), m_tebContributor(drp.tebContributor()), m_running(running),
        m_available(0), m_current(0), m_lastComplete(0)
    {
        m_nodeId = drp.nodeId();
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<4; i++) {
            if (para.laneMask & (1 << i)) {
                logging::info("setting lane  %d", i);
                dmaAddMaskBytes((uint8_t*)mask, dmaDest(i, 0));
            }
        }
        dmaSetMaskBytes(drp.pool.fd(), mask);
    }

    Pds::EbDgram* next(uint32_t& evtIndex, uint64_t& bytes);
private:
    Pds::EbDgram* _handle(uint32_t& evtIndex, uint64_t& bytes);
    const Parameters& m_para;
    MemPool& m_pool;
    Pds::Eb::TebContributor& m_tebContributor;
    static const int MAX_RET_CNT_C = 100;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    const bool& m_running;
    int32_t m_available;
    int32_t m_current;
    uint32_t m_lastComplete;
    XtcData::TransitionId::Value m_lastTid;
    uint32_t m_lastData[6];
    unsigned m_nodeId;
};

Pds::EbDgram* Pgp::_handle(uint32_t& current, uint64_t& bytes)
{
    int32_t size = dmaRet[m_current];
    uint32_t index = dmaIndex[m_current];
    uint32_t lane = (dest[m_current] >> 8) & 7;
    bytes += size;
    if (unsigned(size) > m_pool.dmaSize()) {
        logging::critical("DMA overflowed buffer: %d vs %d\n", size, m_pool.dmaSize());
        exit(-1);
    }

    const uint32_t* data = (uint32_t*)m_pool.dmaBuffers[index];
    uint32_t evtCounter = data[5] & 0xffffff;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    current = evtCounter & bufferMask;
    PGPEvent* event = &m_pool.pgpEvents[current];

    DmaBuffer* buffer = &event->buffers[lane];
    buffer->size = size;
    buffer->index = index;
    event->mask |= (1 << lane);

    logging::debug("PGPReader  lane %d  size %d  hdr %016lx.%016lx.%08x",
                   lane, size,
                   reinterpret_cast<const uint64_t*>(data)[0],
                   reinterpret_cast<const uint64_t*>(data)[1],
                   reinterpret_cast<const uint32_t*>(data)[4]);

    const Pds::TimingHeader* timingHeader = reinterpret_cast<const Pds::TimingHeader*>(data);
    // Revisit: Check bit 7 in pulseId for error
    bool error = timingHeader->control() & (1 << 7);
    if (error) {
        logging::error("Error bit in pulseId is set");
    }
    XtcData::TransitionId::Value transitionId = timingHeader->service();
    if (transitionId != XtcData::TransitionId::L1Accept) {
        logging::debug("PGPReader  saw %s transition @ %u.%09u (%014lx)",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId());
    }
    if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
        logging::critical("%PGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
               RED_ON, m_lastComplete, evtCounter, evtCounter - m_lastComplete, XtcData::TransitionId::name(transitionId), RED_OFF);
        logging::critical("data: %08x %08x %08x %08x %08x %08x",
               data[0], data[1], data[2], data[3], data[4], data[5]);

        logging::critical("lastTid %s", XtcData::TransitionId::name(m_lastTid));
        logging::critical("lastData: %08x %08x %08x %08x %08x %08x",
               m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5]);

        throw "Jump in event counter";

        for (unsigned e=m_lastComplete+1; e<evtCounter; e++) {
            PGPEvent* brokenEvent = &m_pool.pgpEvents[e & bufferMask];
            logging::error("broken event:  %08x", brokenEvent->mask);
            brokenEvent->mask = 0;

        }
    }
    m_lastComplete = evtCounter;
    m_lastTid = transitionId;
    memcpy(m_lastData, data, 24);

    event->l3InpBuf = m_tebContributor.allocate(*timingHeader, (void*)((uintptr_t)current));

    // make new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    Pds::EbDgram* dgram = new(m_pool.pebble[current]) Pds::EbDgram(*timingHeader, XtcData::Src(m_nodeId), m_para.rogMask);

    return dgram;
}

Pds::EbDgram* Pgp::next(uint32_t& evtIndex, uint64_t& bytes)
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
                if (m_running)  logging::debug("pgp timeout");
                return nullptr;
            }
        }
    }

    Pds::EbDgram* dgram = _handle(evtIndex, bytes);
    m_current++;
    return dgram;
}


PvaDetector::PvaDetector(Parameters& para, const std::string& pvName, DrpBase& drp) :
    XpmDetector(&para, &drp.pool),
    m_pvName(pvName),
    m_drp(drp),
    m_inputQueue(drp.pool.nbuffers()),
    m_terminate(false),
    m_running(false)
{
}

unsigned PvaDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc)
{
    logging::info("PVA configure");

    m_exporter = std::make_shared<MetricExporter>();
    if (m_drp.exposer()) {
        m_drp.exposer()->RegisterCollectable(m_exporter);
    }

    m_pvaMonitor = std::make_unique<PvaMonitor>(m_pvName.c_str(), *this);

    auto start = std::chrono::steady_clock::now();
    while(true) {
        if (m_pvaMonitor->connected()) {
            m_pvaMonitor->printStructure();
            break;
        }
        usleep(100000);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > 5000) {
            logging::error("Failed to connect with %s", m_pvaMonitor->name().c_str());
            return 1;
        }
    }

    XtcData::Alg pvaAlg("pvaAlg", 1, 2, 3);
    XtcData::NamesId pvaNamesId(nodeId, PvaNamesIndex);
    XtcData::Names& pvaNames = *new(xtc) XtcData::Names("pva", pvaAlg,
                                                        "pva", "pva1234", pvaNamesId);
    size_t payloadSize;
    XtcData::VarDef pvaDef = m_pvaMonitor->get(payloadSize);
    logging::debug("payloadSize %zd", payloadSize);
    if (payloadSize > m_pool->pebble.bufferSize()) {
        logging::error("Event buffer size (%zd) is too small for %s payload (%zd)",
                       m_pool->pebble.bufferSize(), m_pvaMonitor->name().c_str(), payloadSize);
        return 1;
    }

    pvaNames.add(xtc, pvaDef);
    m_namesLookup[pvaNamesId] = XtcData::NameIndex(pvaNames);

    m_workerThread = std::thread{&PvaDetector::_worker, this};

    return 0;
}

void PvaDetector::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    XtcData::NamesId namesId(nodeId, PvaNamesIndex);
    XtcData::DescribedData desc(dgram.xtc, m_namesLookup, namesId);
    size_t length;
    size_t size = m_pvaMonitor->getData(desc.data(), length);
    desc.set_data_length(size);
    unsigned shape[] = { unsigned(length) };
    desc.set_array_shape(0, shape);
    //size_t sz = (sizeof(dgram) + dgram.xtc.sizeofPayload()) >> 2;
    //uint32_t* payload = (uint32_t*)dgram.xtc.payload();
    //printf("sz = %zd, size = %zd, extent = %d, szofPyld = %d, pyldIdx = %ld\n", sz, size, dgram.xtc.extent, dgram.xtc.sizeofPayload(), payload - (uint32_t*)dgram);
    //uint32_t* buf = (uint32_t*)&dgram;
    //for (unsigned i = 0; i < sz; ++i) {
    //  if (&buf[i] == (uint32_t*)&dgram)       printf(  "dgram:   ");
    //  if (&buf[i] == (uint32_t*)payload)      printf("\npayload: ");
    //  if (&buf[i] == (uint32_t*)desc.data())  printf("\ndata:    ");
    //  printf("%08x ", buf[i]);
    //}
    //printf("\n");
}

void PvaDetector::shutdown()
{
    m_exporter.reset();

    m_terminate.store(true, std::memory_order_release);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    m_pvaMonitor.reset();
    m_namesLookup.clear();   // erase all elements
}

void PvaDetector::_worker()
{
    // setup monitoring
    std::map<std::string, std::string> labels{{"partition", std::to_string(m_para->partition)},
                                              {"PV", m_pvaMonitor->name()}};
    m_nEvents = 0;
    m_exporter->add("drp_event_rate", labels, MetricType::Rate,
                    [&](){return m_nEvents;});
    uint64_t bytes = 0L;
    m_exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                    [&](){return bytes;});
    m_nUpdates = 0;
    m_exporter->add("pva_update_rate", labels, MetricType::Rate,
                    [&](){return m_nUpdates;});
    m_nMatch = 0;
    m_exporter->add("pva_match_count", labels, MetricType::Counter,
                    [&](){return m_nMatch;});
    m_nEmpty = 0;
    m_exporter->add("pva_empty_count", labels, MetricType::Counter,
                    [&](){return m_nEmpty;});
    m_nMissed = 0;
    m_exporter->add("pva_miss_count", labels, MetricType::Counter,
                    [&](){return m_nMissed;});
    m_nTooOld = 0;
    m_exporter->add("pva_tooOld_count", labels, MetricType::Counter,
                    [&](){return m_nTooOld;});
    m_nTimedOut = 0;
    m_exporter->add("pva_timeout_count", labels, MetricType::Counter,
                    [&](){return m_nTimedOut;});

    m_exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
                    [&](){return m_inputQueue.guess_size();});

    Pgp pgp(*m_para, m_drp, m_running);

    m_terminate.store(false, std::memory_order_release);

    auto t0 = Pds::fast_monotonic_clock::now();
    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }

        uint32_t index;
        Pds::EbDgram* dgram = pgp.next(index, bytes);
        if (dgram) {
            XtcData::TransitionId::Value service = dgram->service();
            if ((service == XtcData::TransitionId::L1Accept) ||
                (service == XtcData::TransitionId::SlowUpdate)) {
                m_inputQueue.push(index);

                // Run the timeout routine once in a while to sweep out older
                // events.  If the PV is updating, _timeout() never finds
                // anything to do.  Delay avoids queue head contention.
                using ms_t = std::chrono::milliseconds;
                auto  t1   = Pds::fast_monotonic_clock::now();
                const unsigned msTmo = 100;
                if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > msTmo)
                {
                    const unsigned nsTmo = msTmo * 1000000;
                    XtcData::TimeStamp timestamp(dgram->time.value() - nsTmo);
                    _timeout(timestamp);

                    t0 = Pds::fast_monotonic_clock::now();
                }
            }
            else {
                // Since the Transition Dgram's XTC was already created on
                // phase1 of the transition, fix up the Dgram header with the
                // real one while taking care not to touch the XTC
                // Revisit: Delay this until EbReceiver time?
                Pds::EbDgram* trDgram = m_pool->transitionDgram();
                memcpy(trDgram, dgram, sizeof(*dgram) - sizeof(dgram->xtc));

                if (service == XtcData::TransitionId::Enable) {
                    m_running = true;
                }
                else if (service == XtcData::TransitionId::Disable) { // Sweep out L1As
                    m_running = false;
                    logging::debug("Sweeping out L1Accepts and SlowUpdates");
                    _timeout(dgram->time);
                }

                _sendToTeb(*dgram, index);
            }
        }
    }
    logging::info("Worker thread finished");
}

void PvaDetector::process(const PvaMonitor& pva)
{
    // Prevent _timeout() from interfering
    std::lock_guard<std::mutex> lock(m_lock);

    unsigned seconds = pva.getScalarAs<unsigned>("timeStamp.secondsPastEpoch");
    unsigned nanoseconds = pva.getScalarAs<unsigned>("timeStamp.nanoseconds");
    // Convert timestamp from 1/1/70 to 1/1/90 epoch (5 leap years)
    XtcData::TimeStamp timestamp(seconds - (20*365+5)*24*3600, nanoseconds);

    if (m_running) {
        ++m_nUpdates;
        logging::debug("%s updated @ %u.%09u", pva.name().c_str(), seconds, nanoseconds);
    }

    while (true) {
        uint32_t index;
        if (!m_inputQueue.peek(index)) {
            if (m_running) {
                ++m_nMissed;
            }
            return;
        }

        Pds::EbDgram* dgram = (Pds::EbDgram*)m_pool->pebble[index];
        if (timestamp == dgram->time) {
            uint32_t idx;
            m_inputQueue.try_pop(idx);  // Actually consume the element
            assert(idx == index);

            ++m_nMatch;
            logging::debug("PV matches PGP!!  "
                           "TimeStamps: PV %u.%09u == PGP %u.%09u\n",
                           timestamp.seconds(), timestamp.nanoseconds(),
                           dgram->time.seconds(), dgram->time.nanoseconds());

            PGPEvent* pgpEvent = &m_pool->pgpEvents[index];
            event(*dgram, pgpEvent);

            _sendToTeb(*dgram, index);
            break;
        }
        // The PV is newer than the event, so forward empty event with damage
        else if (timestamp > dgram->time) {
            uint32_t idx;
            m_inputQueue.try_pop(idx);  // Actually consume the element
            assert(idx == index);

            if (dgram->service() != XtcData::TransitionId::SlowUpdate) {
                // No PVA data so mark event as damaged
                dgram->xtc.damage.increase(XtcData::Damage::MissingData);

                ++m_nEmpty;
                logging::debug("No PV data!!      "
                               "TimeStamps: PV %u.%09u > PGP %u.%09u\n",
                               timestamp.seconds(), timestamp.nanoseconds(),
                               dgram->time.seconds(), dgram->time.nanoseconds());
            }
            _sendToTeb(*dgram, index);
            // Keep processing PGP events until a match is found
        }
        // The PV is older than the event, so go await an update
        else {
            if (dgram->service() != XtcData::TransitionId::SlowUpdate) {
                ++m_nTooOld;
                logging::debug("PV too old!!      "
                               "TimeStamps: PV %u.%09u < PGP %u.%09u\n",
                               timestamp.seconds(), timestamp.nanoseconds(),
                               dgram->time.seconds(), dgram->time.nanoseconds());
            }
            break;
        }
    }
}

void PvaDetector::_timeout(const XtcData::TimeStamp& timestamp)
{
    // Prevent handling of newer events from interfering
    std::lock_guard<std::mutex> lock(m_lock);

    while (true) {
        uint32_t index;
        if (!m_inputQueue.peek(index)) {
            break;
        }

        Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(m_pool->pebble[index]);
        if (dgram->time > timestamp) {
            break;                  // dgram is newer than the timeout timestamp
        }

        uint32_t idx;
        m_inputQueue.try_pop(idx);  // Actually consume the element
        assert(idx == index);

        if (dgram->service() != XtcData::TransitionId::SlowUpdate) {
            // No PVA data so mark event as damaged
            dgram->xtc.damage.increase(XtcData::Damage::TimedOut);

            ++m_nTimedOut;
            logging::debug("Event timed out!! "
                           "TimeStamps: timeout %u.%09u > PGP %u.%09u\n",
                           timestamp.seconds(), timestamp.nanoseconds(),
                           dgram->time.seconds(), dgram->time.nanoseconds());
        }
        _sendToTeb(*dgram, index);
    }
}

void PvaDetector::_sendToTeb(Pds::EbDgram& dgram, uint32_t index)
{
    m_nEvents++;

    // Make sure the datagram didn't get too big
    const size_t size = sizeof(dgram) + dgram.xtc.sizeofPayload();
    const size_t maxSize = ((dgram.service() == XtcData::TransitionId::L1Accept) ||
                            (dgram.service() == XtcData::TransitionId::SlowUpdate))
                         ? m_pool->bufferSize()
                         : m_para->maxTrSize;
    if (size > maxSize) {
        logging::critical("%s Dgram of size %zd overflowed buffer of size %zd", XtcData::TransitionId::name(dgram.service()), size, maxSize);
        exit(-1);
    }

    PGPEvent* event = &m_pool->pgpEvents[index];
    if (event->l3InpBuf) { // else timed out
        Pds::EbDgram* l3InpDg = new(event->l3InpBuf) Pds::EbDgram(dgram);
        if (l3InpDg->isEvent()) {
            if (m_drp.triggerPrimitive()) { // else this DRP doesn't provide input
                m_drp.triggerPrimitive()->event(*m_pool, index, dgram.xtc, l3InpDg->xtc); // Produce
            }
        }
        m_drp.tebContributor().process(l3InpDg);
    }
}


PvaApp::PvaApp(Parameters& para, const std::string& pvName) :
    CollectionApp(para.collectionHost, para.partition, "drp", para.alias),
    m_drp(para, context()),
    m_para(para),
    m_det(std::make_unique<PvaDetector>(m_para, pvName, m_drp))
{
    if (m_det == nullptr) {
        logging::critical("Error !! Could not create Detector object");
        throw "Could not create Detector object";
    }
    if (m_para.outputDir.empty()) {
        logging::info("output dir: n/a");
    } else {
        logging::info("output dir: %s", m_para.outputDir.c_str());
    }
    logging::info("Ready for transitions");
}

void PvaApp::_shutdown()
{
    m_drp.shutdown();        // TebContributor must be shut down befoe the worker
    m_det->shutdown();
}

json PvaApp::connectionInfo()
{
    std::string ip = getNicIp();
    logging::debug("nic ip  %s", ip.c_str());
    json body = {{"connect_info", {{"nic_ip", ip}}}};
    json info = m_det->connectionInfo();
    body["connect_info"].update(info);
    json bufInfo = m_drp.connectionInfo();
    body["connect_info"].update(bufInfo);
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

    m_det->nodeId = m_drp.nodeId();
    m_det->connect(msg, std::to_string(getId()));

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
    std::string key = msg["header"]["key"];
    logging::debug("handlePhase1 for %s in PvaDetectorApp", key.c_str());

    XtcData::Xtc& xtc = m_det->transitionXtc();
    XtcData::TypeId tid(XtcData::TypeId::Parent, 0);
    xtc.src = XtcData::Src(m_det->nodeId); // set the src field for the event builders
    xtc.damage = 0;
    xtc.contains = tid;
    xtc.extent = sizeof(XtcData::Xtc);

    json phase1Info{ "" };
    if (msg.find("body") != msg.end()) {
        if (msg["body"].find("phase1Info") != msg["body"].end()) {
            phase1Info = msg["body"]["phase1Info"];
        }
    }

    json body = json({});

    if (key == "configure") {
        if (m_unconfigure) {
            _shutdown();
            m_unconfigure = false;
        }

        std::string errorMsg = m_drp.configure(msg);
        if (!errorMsg.empty()) {
            errorMsg = "Phase 1 error: " + errorMsg;
            logging::error("%s", errorMsg.c_str());
            _error(key, msg, errorMsg);
            return;
        }
        else {
            std::string config_alias = msg["body"]["config_alias"];
            unsigned error = m_det->configure(config_alias, xtc);
            if (error) {
                std::string errorMsg = "Phase 1 error in Detector::configure";
                logging::error("%s", errorMsg.c_str());
                _error(key, msg, errorMsg);
                return;
            }
            else {
                m_drp.runInfoSupport(xtc, m_det->namesLookup());
            }
        }
    }
    else if (key == "unconfigure") {
        // Delay unconfiguration until after phase 2 of unconfigure has completed
        m_unconfigure = true;
    }
    else if (key == "beginrun") {
        RunInfo runInfo;
        std::string errorMsg = m_drp.beginrun(phase1Info, runInfo);
        if (!errorMsg.empty()) {
            body["err_info"] = errorMsg;
            logging::error("%s", errorMsg.c_str());
        }
        else if (runInfo.runNumber > 0) {
            m_drp.runInfoData(xtc, m_det->namesLookup(), runInfo);
        }
    }
    else if (key == "endrun") {
        std::string errorMsg = m_drp.endrun(phase1Info);
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

} // namespace Drp


void get_kwargs(Drp::Parameters& para, const std::string& kwargs_str) {
    std::istringstream ss(kwargs_str);
    std::string kwarg;
    std::string::size_type pos = 0;
    while (getline(ss, kwarg, ',')) {
        pos = kwarg.find("=", pos);
        if (!pos) {
          logging::critical("Keyword argument with no equal sign");
            throw "error: keyword argument with no equal sign: "+kwargs_str;
        }
        std::string key = kwarg.substr(0,pos);
        std::string value = kwarg.substr(pos+1,kwarg.length());
        //cout << kwarg << " " << key << " " << value << endl;
        para.kwargs[key] = value;
    }
}

int main(int argc, char* argv[])
{
    Drp::Parameters para;
    para.partition = -1;
    para.laneMask = 0x1;
    para.detSegment = 0;
    std::string kwargs_str;
    para.verbose = 0;
    int c;
    while((c = getopt(argc, argv, "p:o:C:d:u:k:P:T::M:v")) != EOF) {
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
            case 'k':
                kwargs_str = std::string(optarg);
                break;
            case 'P':
                para.instrument = optarg;
                break;
            case 'T':
                para.trgDetName = optarg ? optarg : "trigger";
                break;
            case 'M':
                para.prometheusDir = optarg;
                break;
            case 'v':
                ++para.verbose;
                break;
            default:
                exit(1);
        }
    }

    switch (para.verbose) {
        case 0:  logging::init(para.instrument.c_str(), LOG_INFO);   break;
        default: logging::init(para.instrument.c_str(), LOG_DEBUG);  break;
    }
    logging::info("logging configured");
    if (para.instrument.empty()) {
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

    get_kwargs(para, kwargs_str);

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
