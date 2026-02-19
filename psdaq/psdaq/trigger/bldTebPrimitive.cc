#include "TriggerPrimitive.hh"
#include "BldTebData.hh"
#include "EBeamTebData.hh"
#include "GasDetTebData.hh"
#include "GmdTebData.hh"
#include "PhaseCavityTebData.hh"
#include "XGmdTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/NamesIter.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "psalg/utils/SysLog.hh"

#include <cstdint>
#include <stdio.h>

using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Pds {
    namespace Trg {
        //
        //  Use a factory pattern to do all the association conditions at configure
        //
        class BldTebDataFactory {
        public:
            ~BldTebDataFactory() {}
            virtual BldTebData::BldSource source() = 0;
            virtual void create(BldTebData&, XtcData::DescData&) = 0;
        };

        class GmdTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::gmd_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] GmdTebData& dat = *new(tdat.gmd()) 
                    GmdTebData( desc.get_value<float_t>("milliJoulesPerPulse") );
            }
        };

        class XGmdTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::xgmd_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] XGmdTebData& dat = *new(tdat.xgmd()) 
                    XGmdTebData( desc.get_value<float_t>("milliJoulesPerPulse"),
                                 desc.get_value<float_t>("POSY") );
            }
        };

        class PhaseCavityTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::pcav_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] PhaseCavityTebData& dat = *new(tdat.pcav()) 
                    PhaseCavityTebData( desc.get_value<double_t>("fitTime1"),
                                        desc.get_value<double_t>("fitTime2"),
                                        desc.get_value<double_t>("charge1"),
                                        desc.get_value<double_t>("charge2") );
            }
        };

        class PhaseCavitysTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::pcavs_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] PhaseCavityTebData& dat = *new(tdat.pcavs()) 
                    PhaseCavityTebData( desc.get_value<double_t>("fitTime1"),
                                        desc.get_value<double_t>("fitTime2"),
                                        desc.get_value<double_t>("charge1"),
                                        desc.get_value<double_t>("charge2") );
            }
        };

        class GasDetTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::gasdet_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] GasDetTebData& dat = *new(tdat.gasdet()) 
                    GasDetTebData( desc.get_value<double_t>("f11ENRC"),
                                   desc.get_value<double_t>("f12ENRC"),
                                   desc.get_value<double_t>("f21ENRC"),
                                   desc.get_value<double_t>("f22ENRC"),
                                   desc.get_value<double_t>("f63ENRC"),
                                   desc.get_value<double_t>("f64ENRC") );
            }
        };

        class EBeamTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::ebeam_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] EBeamTebData& dat = *new(tdat.ebeam()) 
                    EBeamTebData( desc.get_value<double_t>("ebeamL3Energy") );
            }
        };

        class EBeamsTebDataFactory : public BldTebDataFactory {
        public:
            BldTebData::BldSource source() override { return BldTebData::ebeams_; }
            void create(BldTebData& tdat, XtcData::DescData& desc) override {
                [[maybe_unused]] EBeamTebData& dat = *new(tdat.ebeams()) 
                    EBeamTebData( desc.get_value<double_t>("ebeamL3Energy") );
            }
        };

        //  This holds all the associations
        typedef std::unordered_map<unsigned,BldTebDataFactory*> TebDataFactory;

        class BldTebPrimitive : public TriggerPrimitive {
        public:
            int    configure(const json& configureMsg,
                             const json& connectMsg,
                             size_t      collectionId) override;
            void   configure(const XtcData::Xtc& xtc, const void* bufEnd) override;
            void   event(const Drp::MemPool& pool,
                         uint32_t            idx,
                         const XtcData::Xtc& ctrb,
                         XtcData::Xtc&       xtc,
                         const void*         bufEnd) override;
            size_t size() const  { return BldTebData::sizeof_(); }
        private:
            uint64_t             m_sources;
            XtcData::NamesLookup m_namesLookup;
            TebDataFactory       m_factory;
        };

        class BldDataIterator : public XtcData::XtcIterator
        {
        public:
            BldDataIterator(XtcData::Xtc&         xtc, 
                            const void*           bufEnd,
                            XtcData::Xtc&         dst,
                            const void*           dstBufEnd,
                            uint64_t              sources,
                            XtcData::NamesLookup& namesLookup,
                            TebDataFactory&       factory);
        
            virtual int process(XtcData::Xtc* xtc, const void* bufEnd);
        private:
            XtcData::NamesLookup& _namesLookup;
            TebDataFactory&       _tebFactory;
            std::unordered_map<unsigned,XtcData::ShapesData*> _shapesData;
            uint64_t              _sources;
            BldTebData*           _tebData;
        };

    };

};


using namespace Pds::Trg;
using namespace XtcData;

BldDataIterator::BldDataIterator(Xtc&                  xtc, 
                                 const void*           bufEnd,
                                 Xtc&                  dest,
                                 const void*           destBufEnd,
                                 uint64_t              sources,
                                 XtcData::NamesLookup& namesLookup,
                                 TebDataFactory&       factory) : 
    XtcIterator (&xtc, bufEnd),
    _namesLookup(namesLookup),
    _tebFactory (factory),
    _sources    (0)
{
    iterate(); 

    _tebData = new((void*)dest.next()) BldTebData(_sources);
    logging::info("BldDataIterator tebData [%x] at %p",_sources,_tebData);

    for(const auto& [key, value] : _shapesData) {
        DescData desc(*value,_namesLookup[key]);
        logging::info("DescData(%p,namesLookup[%x]) src %u", value, key, _tebFactory[key]->source());
        _tebFactory[key]->create(*_tebData,desc);
    }

    dest.alloc( _tebData->offset_(BldTebData::NSOURCES), destBufEnd );
}

int BldDataIterator::process(Xtc* xtc, const void* bufEnd)
{
    switch (xtc->contains.id()) {
    case (TypeId::Parent): {
        iterate(xtc, bufEnd);
        break;
    }
    case (TypeId::ShapesData): {
        ShapesData* psd = (ShapesData*)xtc;
        unsigned namesId = psd->namesId();
        logging::info("BldDataIterator shapesData[%x] = %p", namesId,psd);
        _shapesData[namesId] = psd;
        if (_tebFactory.find(namesId)!=_tebFactory.end())
            _sources |= 1ULL<<_tebFactory[namesId]->source();
        break;
    }
    default:
        break;
    }
    return 1;
}

int Pds::Trg::BldTebPrimitive::configure(const json& configureMsg,
                                         const json& connectMsg,
                                         size_t      collectionId)
{
    logging::info("BldTebPrimitive::configure json");
    return 0;
}

void Pds::Trg::BldTebPrimitive::configure(const Xtc& xtc, const void* bufEnd)
{
    logging::info("BldTebPrimitive::configure xtc %p  contains %x  size %u  bufEnd %p", 
                     &xtc, xtc.contains.value(), xtc.sizeofPayload(), bufEnd);

    // Need to cache the name lookup, detName -> namesId
    NamesIter iter;
    iter.iterate(const_cast<Xtc*>(&xtc), bufEnd);
    m_namesLookup = iter.namesLookup();

    //  Build the mask of data sources
    m_sources = 0;
    for(const auto& [key, value] : m_namesLookup) {  // value is NameIndex
        Names& names = const_cast<NameIndex&>(value).names();
        BldTebData::BldSource src = BldTebData::lookup(names.detName());
        if (src < BldTebData::NSOURCES) {
            m_sources |= (1<<src);

            BldTebDataFactory* gen=0;
            if (strcmp(names.detName(),"gmd")==0)
                gen = new GmdTebDataFactory;
            else if (strcmp(names.detName(),"xgmd")==0)
                gen = new XGmdTebDataFactory;
            else if (strcmp(names.detName(),"pcav")==0)
                gen = new PhaseCavityTebDataFactory;
            else if (strcmp(names.detName(),"pcavs")==0)
                gen = new PhaseCavitysTebDataFactory;
            else if (strcmp(names.detName(),"gasdet")==0)
                gen = new GasDetTebDataFactory;
            else if (strcmp(names.detName(),"ebeam")==0)
                gen = new EBeamTebDataFactory;
            else if (strcmp(names.detName(),"ebeams")==0)
                gen = new EBeamsTebDataFactory;

            if (!gen) {
                logging::critical("No TebDataFactory found for %s", names.detName());
                abort();
            }

            m_factory[names.namesId()] = gen;
        }

        // print out
        printf("[%x] = id [%x] detName [%s]\n",
               key, names.namesId().value(), names.detName());
        for(unsigned i=0; i<names.num(); i++) {
            Name& name = names.get(i);
            printf("    %s [%s] rank %u\n",
                   name.name(), name.str_type(), name.rank());
        }
    }
}

void Pds::Trg::BldTebPrimitive::event(const Drp::MemPool& pool,
                                      uint32_t            idx,
                                      const Xtc&          ctrb,
                                      Xtc&                xtc,
                                      const void*         bufEnd)
{
    logging::info("BldTebPrimitive::event xtc %p  bufEnd %p", &xtc, bufEnd);
    if (!m_sources)
        return;

    //  Build the result within the iterator
    BldDataIterator iter(const_cast<Xtc&>(ctrb), ctrb.payload()+ctrb.sizeofPayload(), 
                         xtc, bufEnd, m_sources, m_namesLookup, m_factory);

}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_bld()
{
    return new Pds::Trg::BldTebPrimitive;
}
