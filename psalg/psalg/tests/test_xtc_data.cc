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
#include "xtcdata/xtc/NamesIter.hh"
#include "xtcdata/xtc/NamesLookup.hh"

//using namespace psalgos;
//using namespace psalg;
using namespace std; 

using namespace XtcData;
//using std::string;

//-----------------------------

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
        case (TypeId::Parent): {
	    iterate(xtc); 
            break;
        }
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
	  //_shapesData = (ShapesData*)xtc;
 
            ShapesData* tmp = (ShapesData*)xtc;
            _shapesData[tmp->namesId().namesId()] = tmp;
 
            // lookup the index of the names we are supposed to use
            NamesId  namesId = tmp->namesId();
	    cout << "YYYYYYYYYY namesId."
                 << "  level:"   << namesId.level()
                 << "  value:"   << namesId.value()
                 << "  namesId:" << namesId.namesId() << '\n';
            NamesIter namesIter(xtc);
            //namesIter.iterate();
            //NamesLookup& namesLookup = namesIter.namesLookup();
            //DescData descdata(shapesdata, namesLookup[namesId]);
            //Names& names = descdata.nameindex().names();

            //for (unsigned i = 0; i < names.num(); i++) {
            //    Name& name = names.get(i);
            //    cout << " " << name.name();
	    //}
            //cout << '\n';

            break;
        }
        case (TypeId::Shapes): {break;}
        case (TypeId::Data):   {break;}
        default:{cout << "YYYY TypeId::default ????? type = " << type << " \n"; break;}
        }

	//cout << "XXXX In MyXtcIterator just before exit\n";
        return Continue;
    }

    ShapesData& config() {return *_shapesData[0];}
    ShapesData& event()  {return *_shapesData[1];}

private:
    ShapesData* _shapesData[2];
};

//-----------------------------

void dump(const char* transition, Names& names, DescData& descdata) {
    printf("------ Names for %s transition ---------\n",transition);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("rank %d type %d name %s\n", name.rank(), name.type(), name.name());
    }
    printf("------ Values for %s transition ---------\n",transition);
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        if (name.type()==Name::INT64 and name.rank()==0) {
            printf("Name %s has value %ld\n", name.name(), descdata.get_value<int64_t>(name.name()));
        }
    }
    printf("\n\n");
}

//-----------------------------

int main (int argc, char* argv[]) {

    const char* fname = "/reg/neh/home/cpo/git/lcls2/psana/psana/dgramPort/jungfrau.xtc2";
    std::cout << "xtc file name: " << fname << '\n';

    unsigned neventreq=3;

    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", fname);
        exit(2);
    }

    XtcFileIterator itdg(fd, 0x4000000);

    Dgram* dg = itdg.next();

    NamesIter& namesIter = *new NamesIter(&(dg->xtc));
    namesIter.iterate();
    NamesLookup& namesLookup = namesIter.namesLookup();
    
    // get data out of the configure transition
    MyXtcIterator dgiter(&(dg->xtc));
    dgiter.iterate();
    NamesId& namesId = dgiter.config().namesId();
    DescData descdata(dgiter.config(), namesLookup[namesId]);
    Names& names = descdata.nameindex().names();

    cout << "ZZZZ the 1st dg - Configure\n";
    dump("Configure", names, descdata);

    unsigned nevent=0;
    while ((dg = itdg.next())) {
        if (nevent>=neventreq) break;
        nevent++;

        MyXtcIterator iter(&(dg->xtc));
        iter.iterate();
 
	cout << "XXXXXXXXXXX dg->seq.isEvent(): " << dg->seq.isEvent() << '\n';

        printf("evt:%04d ==== %s transition: time %d.%09d, pulseId %lux, env %ux, "
               "payloadSize %d extent %d\n", nevent,
               TransitionId::name(dg->seq.service()), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload(),dg->xtc.extent);

        NamesId& namesId = iter.event().namesId();
        DescData descdata(iter.event(), namesLookup[namesId]);
        Names& names = descdata.nameindex().names();
        dump("Event",names,descdata);
    }

    ::close(fd);
    return 0;
}

//-----------------------------
