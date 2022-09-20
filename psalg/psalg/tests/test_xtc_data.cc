/*
 * Test access to xtc data for LCLS2
 */
#include <fcntl.h> // O_RDONLY
#include <stdio.h> // sprintf, printf( "%lf\n", accum );
#include <iostream> // cout, puts etc.
#include <sstream> // stringstream
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

#include "psalg/calib/NDArray.hh"

//using namespace psalgos;
//using namespace psalg;
using namespace std;

using namespace XtcData;
//using std::string;

//-------------------

void print_hline(const unsigned nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------

class TestXtcIterator : public XtcData::XtcIterator
{
public:
    enum {Stop, Continue};
    TestXtcIterator(XtcData::Xtc* xtc, const void* bufEnd) : XtcData::XtcIterator(xtc, bufEnd) {} // {iterate();}
    TestXtcIterator() : XtcData::XtcIterator() {}

    int process(XtcData::Xtc* xtc, const void* bufEnd) {
        TypeId::Type type = xtc->contains.id();
        printf("    TestXtcIterator TypeId::%-12s id: %-4d Xtc* pointer: %p\n", TypeId::name(type), type, xtc);

        switch (xtc->contains.id()) {
            case (TypeId::Parent): {iterate(xtc, bufEnd); break;}
            case (TypeId::Names): {break;}
            case (TypeId::ShapesData): {iterate(xtc, bufEnd); break;}
            case (TypeId::Data): {break;}
            case (TypeId::Shapes): {break;}
            default: {
                printf("    WARNING: UNKNOWN DATA TYPE !!! \n");
                break;
            }
        }

        //==============
        // iterate(xtc, bufEnd);
        //==============

        return Continue;
    }
};

//-----------------------------

// void dump(const char* transition, Names& names, DescData& descdata) {
void dump(const char* comment, DescData& descdata) {

    static const char* _names[] = {"Configuration", "Data"};

  //NameIndex& nameIndex   = descdata.nameindex();
    Names& names           = descdata.nameindex().names();
    ShapesData& shapesData = descdata.shapesdata();
    NamesId& namesId       = shapesData.namesId();

    unsigned trans_ind = namesId.namesId();
    const char* transition  = _names[trans_ind];
    printf("transition: %d  0/1 = config/data transition: %s\n", trans_ind, transition);

    printf("Names:: detName: %s  detType: %s  detId: %s  segment: %d alg.name: %s\n",
	   names.detName(), names.detType(), names.detId(), names.segment(), names.alg().name());

    printf("------ %d Names for %s ---------\n", names.num(), comment);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("%02d name: %-32s rank: %d type: %d %s\n", i, name.name(), name.rank(), name.type(), name.str_type());
    }

    printf("------ Values for %s ---------\n", comment);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);

        if(name.rank()==0) {
          if(name.type()==Name::INT64)
	    printf("%02d name %-32s value %ld\n", i, name.name(), descdata.get_value<int64_t>(name.name()));
          else if(name.type()==Name::UINT16)
	    printf("%02d name %-32s value %d\n", i, name.name(), descdata.get_value<uint16_t>(name.name()));
          else
	    printf("%02d name %-32s not-implemented in dump(...) scalar type: %s\n", i, name.name(), name.str_type());
        }
        else if(name.rank() > 0) {
          ////enum DataType { UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT, DOUBLE, CHARSTR, ENUMVAL, ENUMDICT};
          if(name.type()==Name::UINT16) {
	    psalg::NDArray<uint16_t> nda(descdata.get_array<uint16_t>(i));
	    std::stringstream ss; ss << nda;
	    printf("%02d name %-32s Array %s\n", i, name.name(), ss.str().c_str());
	  }
          else if(name.type()==Name::INT64) {
	    psalg::NDArray<int64_t> nda(descdata.get_array<int64_t>(i));
	    std::stringstream ss; ss << nda;
	    printf("%02d name %-32s Array %s\n", i, name.name(), ss.str().c_str());
	  }
          else if(name.type()==Name::INT16) {
	    psalg::NDArray<int16_t> nda(descdata.get_array<int16_t>(i));
	    std::stringstream ss; ss << nda;
	    printf("%02d name %-32s Array %s\n", i, name.name(), ss.str().c_str());
	  }
          else if(name.type()==Name::FLOAT) {
	    psalg::NDArray<float> nda(descdata.get_array<float>(i));
	    std::stringstream ss; ss << nda;
	    printf("%02d name %-32s Array %s\n", i, name.name(), ss.str().c_str());
	  }
          else if(name.type()==Name::DOUBLE) {
	    psalg::NDArray<double> nda(descdata.get_array<double>(i));
	    std::stringstream ss; ss << nda;
	    printf("%02d name %-32s Array %s\n", i, name.name(), ss.str().c_str());
	  }
          else if(name.type()==Name::UINT8) {
	    psalg::NDArray<uint8_t> nda(descdata.get_array<uint8_t>(i));
	    std::stringstream ss; ss << nda;
	    printf("%02d name %-32s Array %s\n", i, name.name(), ss.str().c_str());
	  }
          else
	    printf("%02d name %-32s not-implemented in dump(...) type: Array<%s>\n", i, name.name(), name.str_type());
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

//-------------------

int file_descriptor(const char* fname="data-xpptut15-r0430-e000010-jungfrau.xtc2",
                    const char* dname="/reg/g/psdm/detector/data2_test/xtc/") {
  //const char* path = "/reg/neh/home/cpo/git/lcls2/psana/psana/dgramPort/jungfrau.xtc2";
  //const char* path = "/reg/g/psdm/detector/data2_test/xtc/data-cxid9114-r0089-e000010-cspad.xtc2";
  //const char* path = "/reg/g/psdm/detector/data2_test/xtc/data-xpptut15-r0430-e000010-jungfrau.xtc2";
  string path(dname); path += fname;
    std::cout << "xtc file name: " << path.c_str() << '\n';
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", path.c_str());
        exit(2);
    }
    return fd;
}

//-----------------------------

int test_xtc_content(const char* fname="data-xpptut15-r0430-e000010-jungfrau.xtc2") {

    int fd = file_descriptor(fname);
    XtcFileIterator it_fdg(fd, 0x4000000);
    Dgram* dg = it_fdg.next();
    const void* bufEnd = ((char*)dg) + it_fdg.size();

    unsigned ndgreq=1000000;
    unsigned ndg=0;
    while (dg) {
        ndg++;
	printf("%04d ---- datagram ----\n", ndg);
        if (ndg>=ndgreq) break;
        TestXtcIterator iter(&(dg->xtc), bufEnd);
        iter.iterate();
        dg = it_fdg.next();
    }
    return 0;
}

//-----------------------------

int test_all(const char* fname="data-xpptut15-r0430-e000010-jungfrau.xtc2") {

    int fd = file_descriptor(fname);
    XtcFileIterator xfi(fd, 0x4000000);

    // get data out of the 1-st datagram configure transition
    Dgram* dg = xfi.next();
    const void* bufEnd = ((char*)dg) + xfi.size();

    ConfigIter configo(&(dg->xtc), bufEnd);
    NamesLookup& namesLookup = configo.namesLookup();

    //DESC_SHAPE(desc_shape, configo, namesLookup);
    DescData& desc_shape = configo.desc_shape();
    dump("Config shape", desc_shape);

    printf("\nXXX point C sizeofPayload: %d\n", dg->xtc.sizeofPayload());

    ////DESC_VALUE(desc_value, configo, namesLookup);
    DescData& desc_value = configo.desc_value();
    printf("\nXXX point C2\n");
    dump("Config values", desc_value);

    printf("\nXXX point D\n");

    unsigned neventreq=5;
    unsigned nevent=0;
    while ((dg = xfi.next())) {
        if (nevent>=neventreq) break;
        nevent++;

        DataIter datao(&(dg->xtc), bufEnd);

        printf("evt:%04d ==== transition: %s of type: %d time %d.%09d, env %ux, "
               "payloadSize %d extent %d\n", nevent,
               TransitionId::name(dg->service()), dg->type(), dg->time.seconds(),
               dg->time.nanoseconds(),
               dg->env, dg->xtc.sizeofPayload(), dg->xtc.extent);

        //DESC_VALUE(desc_data, datao, namesLookup);
        DescData& desc_data = datao.desc_value(namesLookup);
        dump("Data values", desc_data);
    }

    ::close(fd);
    return 0;
}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_xtc_data <test-number>\n  where test-number";
  if (tname == "" || tname=="0")  ss << "\n   0 - test_xtc_content(jungfrau)  - demo of ...";
  if (tname == "" || tname=="1")  ss << "\n   1 - test_all(jungfrau)          - demo of ...";

  if (tname == "" || tname=="10") ss << "\n  10 - test_xtc_content(cspad)     - demo of ...";
  if (tname == "" || tname=="11") ss << "\n  11 - test_all(cspad)             - demo of ...";

  if (tname == "" || tname=="20") ss << "\n  20 - test_xtc_content(pnccd)     - demo of ...";
  if (tname == "" || tname=="21") ss << "\n  21 - test_all(pnccd)             - demo of ...";

  if (tname == "" || tname=="30") ss << "\n  30 - test_xtc_content(opal1k)    - demo of ...";
  if (tname == "" || tname=="31") ss << "\n  31 - test_all(opal1k)            - demo of ...";

  if (tname == "" || tname=="40") ss << "\n  40 - test_xtc_content(opal4k)    - demo of ...";
  if (tname == "" || tname=="41") ss << "\n  41 - test_all(opal4k)            - demo of ...";

  if (tname == "" || tname=="50") ss << "\n  50 - test_xtc_content(opal8k)    - demo of ...";
  if (tname == "" || tname=="51") ss << "\n  51 - test_all(opal8k)            - demo of ...";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char **argv) {

  print_hline(80,'_');
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  //printf("%H:%M:%S.%f");           // set level and time format

  print_hline(80,'_');
  std::string tname((argc>1) ? argv[1] : "0");
  cout << usage(tname);

  if      (tname== "0") test_xtc_content("data-xpptut15-r0430-e000010-jungfrau.xtc2");
  else if (tname== "1") test_all();

  else if (tname=="10") test_xtc_content("data-cxid9114-r0089-e000010-cspad.xtc2");
  else if (tname=="11") test_all("data-cxid9114-r0089-e000010-cspad.xtc2");

  else if (tname=="20") test_xtc_content("data-xpptut15-r0450-e000010-pnccd.xtc2");
  else if (tname=="21") test_all("data-xpptut15-r0450-e000010-pnccd.xtc2");

  else if (tname=="30") test_xtc_content("data-xppl1316-r0193-e000010-opal1k.xtc2");  // opal1k
  else if (tname=="31") test_all("data-xppl1316-r0193-e000010-opal1k.xtc2");

  else if (tname=="40") test_xtc_content("data-mfxm5116-r0020-e000010-opal4k.xtc2"); // opal4k
  else if (tname=="41") test_all("data-mfxm5116-r0020-e000010-opal4k.xtc2");

  else if (tname=="50") test_xtc_content("data-meca6013-r0009-e000010-opal8k.xtc2");  // opal8k
  else if (tname=="51") test_all("data-meca6013-r0009-e000010-opal8k.xtc2");

  else printf("WARNING: Undefined test name: %s", tname.c_str());

  print_hline(80,'_');
  cout << usage();
  return EXIT_SUCCESS;
}

//-------------------

//int main (int argc, char* argv[]) {
//  return test_all(argc, argv);
//}

//-----------------------------
