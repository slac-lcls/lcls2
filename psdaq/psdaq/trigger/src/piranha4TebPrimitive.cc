#include "TriggerPrimitive.hh"
#include "Piranha4TTTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/NamesIter.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "psalg/utils/SysLog.hh"

#include <cstdint>
#include <stdio.h>

using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Pds {
    namespace Trg {
        class Piranha4TebPrimitive : public TriggerPrimitive {
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
            size_t size() const  { return sizeof(Piranha4TTTebData); }
        private:
            XtcData::NamesLookup m_namesLookup;
            unsigned m_src;
        };

        class MyIterator : public XtcData::XtcIterator {
        public:
            MyIterator(XtcData::Xtc&,
                       const void*,
                       unsigned);
        public:
            XtcData::ShapesData* shapesdata() const { return m_shapesdata; }
            int process(XtcData::Xtc*, const void*);
        private:
            unsigned             m_source;
            XtcData::ShapesData* m_shapesdata;
        };

        class DumpIterator : public XtcData::XtcIterator {
        public:
            DumpIterator(char* root, unsigned indent=0) : 
                m_root(root), m_indent(indent) {}
        public:
            int process(XtcData::Xtc* xtc, const void* bufEnd);
        private:
            char*    m_root;
            unsigned m_indent;
        };

    };

};


using namespace Pds::Trg;
using namespace XtcData;

int DumpIterator::process(Xtc* xtc, const void* bufEnd)
{
    printf("   [%u] xtc 0x%lx  typeid 0x%x  src 0x%x  extent 0x%x\n",
           m_indent, (char*)xtc - m_root, xtc->contains.value(), xtc->src.value(), xtc->extent);
    switch(xtc->contains.id()) {
    case (TypeId::Parent):
    case (TypeId::ShapesData): {
        DumpIterator iter(m_root, m_indent+1);
        iter.iterate(xtc, bufEnd);
        break;
    }
    default:
        break; 
    }
    return 1;
}

MyIterator::MyIterator(Xtc&                  xtc, 
                       const void*           bufEnd,
                       unsigned              source) :
    XtcIterator (&xtc, bufEnd),
    m_source    (source),
    m_shapesdata(0)
{
    iterate();
}

int MyIterator::process(Xtc* xtc, const void* bufEnd)
{
    switch(xtc->contains.id()) {
    case (TypeId::Parent): {
        iterate(xtc, bufEnd);
        break;
    }
    case (TypeId::ShapesData): {
        ShapesData* shapesdata = (ShapesData*)xtc;
        if (shapesdata->namesId()==m_source) {  // Found it!
            m_shapesdata = shapesdata;
            return 0;
        }
        break;
    }
    default:
        break; 
    }
    return 1;
}

int Pds::Trg::Piranha4TebPrimitive::configure(const json& configureMsg,
                                              const json& connectMsg,
                                              size_t      collectionId)
{
    logging::info("Piranha4TebPrimitive::configure json");
    return 0;
}

void Pds::Trg::Piranha4TebPrimitive::configure(const Xtc& xtc, const void* bufEnd)
{
    logging::info("BldTebPrimitive::configure xtc %p  contains %x  size %u  bufEnd %p", 
                     &xtc, xtc.contains.value(), xtc.sizeofPayload(), bufEnd);

    DumpIterator dump((char*)&xtc);
    dump.iterate(const_cast<Xtc*>(&xtc), bufEnd);

    // Need to cache the name lookup, detName -> namesId
    NamesIter iter;
    iter.iterate(const_cast<Xtc*>(&xtc), bufEnd);
    m_namesLookup = iter.namesLookup();

    m_src = -1U;
    for(const auto& [key, value] : m_namesLookup) {  // value is NameIndex
        Names& names = const_cast<NameIndex&>(value).names();
        if (strcmp(names.alg().name(),"ttfex")==0) {
            m_src = key;
            break;
        }
    }
}

void Pds::Trg::Piranha4TebPrimitive::event(const Drp::MemPool& pool,
                                           uint32_t            idx,
                                           const Xtc&          ctrb,
                                           Xtc&                xtc,
                                           const void*         bufEnd)
{
    MyIterator iter(const_cast<Xtc&>(ctrb), ctrb.payload()+ctrb.sizeofPayload(), m_src);

    new(xtc.alloc(sizeof(Piranha4TTTebData), bufEnd))
        Piranha4TTTebData(iter.shapesdata() ? (double*)iter.shapesdata()->data().payload() : 0);
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_piranha4()
{
    return new Pds::Trg::Piranha4TebPrimitive;
}
