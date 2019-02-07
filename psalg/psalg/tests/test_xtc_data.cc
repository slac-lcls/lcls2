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

//using namespace psalgos;
//using namespace psalg;
using namespace std; 

using namespace XtcData;
//using std::string;


class MyXtcIterator : public XtcIterator
{
public:
    enum { Stop, Continue };
    MyXtcIterator(Xtc* xtc) : XtcIterator(xtc)
    {
    }

    int process(Xtc* xtc)
    {
        TypeId::Type type = xtc->contains.id();
	cout << "YYYY TypeId::" << TypeId::name(type) << '\n';

        switch (type) {
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            Alg& alg = names.alg();
	    printf("*** DetName: %s, DetType: %s, Alg: %s, Version: 0x%6.6x, Names:\n",
                   names.detName(), names.detType(), alg.name(), alg.version());

	    cout << "number of names: " << names.num() << '\n';

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("%2d Name: %s Type: %d Rank: %d\n", i, name.name(), name.type(), name.rank());
            }
            break;
        }
        case (TypeId::ShapesData): {
	    //ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            // unsigned namesId = shapesdata.shapes().namesId();
            // DescData descdata(shapesdata, _namesVec[namesId]);
            // Names& names = descdata.nameindex().names();
            //iterate(xtc);
            break;
        }
        case (TypeId::Parent): {break;}
        case (TypeId::Shapes): {break;}
        case (TypeId::Data):   {break;}
        default:{cout << "YYYY TypeId::default ????? type = " << type << " \n"; break; }
        }

	iterate(xtc); 
        return Continue;
    }
};


int main (int argc, char* argv[]) {

    const char* fname = "/reg/neh/home/cpo/git/lcls2/psana/psana/dgramPort/jungfrau.xtc2";
    std::cout << "xtc file name: " << fname << '\n';

    unsigned neventreq=10;

    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", fname);
        exit(2);
    }

    XtcFileIterator itdg(fd, 0x4000000);
    Dgram* dg;
    unsigned nevent=0;
    while ((dg = itdg.next())) {
        if (nevent>=neventreq) break;
        nevent++;
        printf("evt:%04d ==== %s transition: time %d.%09d, pulseId 0x%lux, env 0x%ux, "
               "payloadSize %d extent %d\n", nevent,
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload(),dg->xtc.extent);
        MyXtcIterator iter(&(dg->xtc));
        iter.iterate();
    }

    ::close(fd);

  return 0;
}

