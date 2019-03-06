/*
 * Test access to xtc data for LCLS2
 */
#include <fcntl.h> // O_RDONLY
#include <stdio.h> // for  sprintf, printf( "%lf\n", accum );
#include <iostream> // for cout, puts etc.
//#include <vector>
//#include <stdlib.h>
//#include <fstream>
//#include <stdlib.h>
#include <unistd.h> // close
#include <stdint.h>  // uint8_t, uint32_t, etc.

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
//#include "xtcdata/xtc/NamesIter.hh"
#include "xtcdata/xtc/ConfigIter.hh"
#include "xtcdata/xtc/DataIter.hh"
#include "xtcdata/xtc/NamesLookup.hh"

//using namespace psalgos;
//using namespace psalg;
using namespace std; 

using namespace XtcData;
//using std::string;

//-----------------------------

class TestXtcIterator : public XtcData::XtcIterator
{
public:
    enum { Stop, Continue };
    TestXtcIterator(XtcData::Xtc* xtc) : XtcData::XtcIterator(xtc) {} // {iterate();}
    TestXtcIterator() : XtcData::XtcIterator() {}

    int process(XtcData::Xtc* xtc) {
        TypeId::Type type = xtc->contains.id(); 
	printf("    TestXtcIterator TypeId::%-12s id: %-4d Xtc* pointer: %p\n", TypeId::name(type), type, xtc);

	switch (xtc->contains.id()) {
	case (TypeId::Parent): {iterate(xtc); break;}
	case (TypeId::Names): {break;}
        case (TypeId::ShapesData): {iterate(xtc); break;}
        case (TypeId::Data): {break;}
        case (TypeId::Shapes): {break;}
	default: {
	    printf("    WARNING: UNKNOWN DATA TYPE !!! \n");
	    break;
	    }
	}

        //==============
	// iterate(xtc);
        //==============

        return Continue;
    }
};

//-----------------------------

// void dump(const char* transition, Names& names, DescData& descdata) {
void dump(const char* comment, DescData& descdata) {
 
    static const char* _names[] = {"Configuration", "Data"};

  //NameIndex& nameIndex   = descdata.nameindex();
    ShapesData& shapesData = descdata.shapesdata();
    NamesId& namesId       = shapesData.namesId();
    Names& names           = descdata.nameindex().names();

    unsigned trans_ind = namesId.namesId();
    const char* transition  = _names[trans_ind];
    printf("transition: %d  0/1 = config/data transition: %s\n", trans_ind, transition);

    printf("Names:: detName: %s  detType: %s  detId: %s  segment: %d alg.name: %s\n",
	   names.detName(), names.detType(), names.detId(), names.segment(), names.alg().name());

    printf("------ %d Names for %s ---------\n", names.num(), comment);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("%02d name: %-32s rank: %d type: %d\n", i, name.name(), name.rank(), name.type());
    }
    printf("------ Values for %s ---------\n", comment);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        if (name.type()==Name::INT64 and name.rank()==0) {
	  printf("%02d name %-32s value %ld\n", i, name.name(), descdata.get_value<int64_t>(name.name()));
        }
    }

    printf("============= THE END OF DUMP FOR %s =============\n\n", comment);
}

//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------

int file_descriptor(int argc, char* argv[]) {

    const char* fname = "/reg/neh/home/cpo/git/lcls2/psana/psana/dgramPort/jungfrau.xtc2";
    std::cout << "xtc file name: " << fname << '\n';

    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", fname);
        exit(2);
    }
    
    return fd;
}

//-----------------------------

int test_xtc_content(int argc, char* argv[]) {

    int fd = file_descriptor(argc, argv);
    XtcFileIterator it_fdg(fd, 0x4000000);
    //Dgram* dg = it_fdg.next();
    Dgram* dg;

    unsigned ndgreq=1000000;
    unsigned ndg=0;
    while ((dg = it_fdg.next())) {
        ndg++;
	printf("%04d ---- datagram ----\n", ndg);
        if (ndg>=ndgreq) break;
        TestXtcIterator iter(&(dg->xtc));
        iter.iterate();
    }
    return 0;
}

//-----------------------------

int test_all(int argc, char* argv[]) {

    int fd = file_descriptor(argc, argv);
    XtcFileIterator xfi(fd, 0x4000000);

    // get data out of the 1-st datagram configure transition
    Dgram* dg = xfi.next();

    ConfigIter configo(&(dg->xtc));
    NamesLookup& namesLookup = configo.namesLookup();

    //DESC_SHAPE(desc_shape, configo, namesLookup);
    DescData& desc_shape = configo.desc_shape();
    dump("Config shape", desc_shape);

    //DESC_VALUE(desc_value, configo, namesLookup);
    DescData& desc_value = configo.desc_value();
    dump("Config values", desc_value);

    unsigned neventreq=2;
    unsigned nevent=0;
    while ((dg = xfi.next())) {
        if (nevent>=neventreq) break;
        nevent++;

        DataIter datao(&(dg->xtc));
 
        printf("evt:%04d ==== transition: %s of type: %d time %d.%09d, pulseId %lux, env %ux, "
               "payloadSize %d extent %d\n", nevent,
               TransitionId::name(dg->seq.service()), dg->seq.type(), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload(), dg->xtc.extent);

        //DESC_VALUE(desc_data, datao, namesLookup);
        DescData& desc_data = datao.desc_value(namesLookup);
        dump("Data values", desc_data);
    }

    ::close(fd);
    return 0;
}

//-----------------------------

int main (int argc, char* argv[]) {
  //return test_xtc_content(argc, argv);
  return test_all(argc, argv);
}

//-----------------------------
