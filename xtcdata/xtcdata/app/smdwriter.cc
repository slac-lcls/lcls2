
#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <array>

// additions from xtc writer
#include <type_traits>
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/Smd.hh"

using namespace XtcData;
using std::string;
using namespace std;

#define BUFSIZE 0x4000000

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter() : XtcIterator()
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();
        printf("%d: %s rank %d, type %d\n", i, name.name(), data_rank, data_type);

        switch(name.type()){
        case(Name::UINT8):{
            if(data_rank > 0){
                Array<uint8_t> arrT = descdata.get_array<uint8_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(Name::UINT16):{
            if(data_rank > 0){
                Array<uint16_t> arrT = descdata.get_array<uint16_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(Name::UINT32):{
            if(data_rank > 0){
                Array<uint32_t> arrT = descdata.get_array<uint32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(Name::UINT64):{
            if(data_rank > 0){
                Array<uint64_t> arrT = descdata.get_array<uint64_t>(i);
                printf("%s: %llu, %llu, %llu\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %llu\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(Name::INT8):{
            if(data_rank > 0){
                Array<int8_t> arrT = descdata.get_array<int8_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(Name::INT16):{
            if(data_rank > 0){
                Array<int16_t> arrT = descdata.get_array<int16_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(Name::INT32):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::INT64):{
            if(data_rank > 0){
                Array<int64_t> arrT = descdata.get_array<int64_t>(i);
                printf("%s: %lld, %lld, %lld\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %lld\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(Name::FLOAT):{
            if(data_rank > 0){
                Array<float> arrT = descdata.get_array<float>(i);
                printf("%s: %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1]);
                    }
            else{
                printf("%s: %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(Name::DOUBLE):{
            if(data_rank > 0){
                Array<double> arrT = descdata.get_array<double>(i);
                printf("%s: %f, %f, %f\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        case(Name::CHARSTR):{
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(i);
                printf("%s: \"%s\"\n",name.name(),arrT.data());
                    }
            else{
                printf("%s: string with no rank?!?\n",name.name());
            }
            break;
        }

        case(Name::ENUMVAL):{
            if(data_rank > 0){
                Array<int32_t> arrT = descdata.get_array<int32_t>(i);
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::ENUMDICT):{
            if(data_rank > 0){
                printf("%s: enumdict with rank?!?\n", name.name());
            } else{
                printf("%s: %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }
        }
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
        printf("*** DetName: %s, Segment %d, DetType: %s, Alg: %s, Version: 0x%6.6x, Names:\n",
                   names.detName(), names.segment(), names.detType(),
                   alg.name(), alg.version());

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("Name: %s Type: %d Rank: %d\n",name.name(),name.type(), name.rank());
            }

            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this xtc
            // may not have a _namesLookup
            if (_namesLookup.count(namesId)<0) break;
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
        printf("Found %d names\n",names.num());
            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                get_value(i, name, descdata);
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    NamesLookup _namesLookup;
};

void save(Dgram& dg, FILE* xtcFile) {
    if (fwrite(&dg, sizeof(dg) + dg.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
}

void show(Dgram& dg) {
    printf("%s transition: time %d.%09d, env 0x%u, "
           "payloadSize %d extent %d\n",
           TransitionId::name(dg.service()), dg.time.seconds(),
           dg.time.nanoseconds(),
           dg.env, dg.xtc.sizeofPayload(),dg.xtc.extent);
    DebugIter dbgiter;
    dbgiter.iterate(&(dg.xtc));
}

void usage(char* progname)
{
  fprintf(stderr, "Usage: %s -f <filename> [-h]\n", progname);
}

#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
    /*
    * The smdwriter reads an xtc file, extracts
    * payload size for each event datagram,
    * then writes out (fseek) offset in smd.xtc2 file.
    */
    int c;
    int writeTs = 0;
    char* tsname = 0;
    char* xtcname = 0;
    int parseErr = 0;
    size_t n_events = 0;
    int n_mod = 0;
    char outname[MAX_FNAME_LEN];
    strncpy(outname, "smd.xtc2", MAX_FNAME_LEN);

    while ((c = getopt(argc, argv, "ht:n:m:f:o:")) != -1) {
    switch (c) {
      case 'h':
        usage(argv[0]);
        exit(0);
      case 't':
        writeTs = 1;
        tsname = optarg;
        break;
      case 'n':
        n_events = stoi(optarg);
        break;
      case 'm':
        n_mod = stoi(optarg);
        break;
      case 'f':
        xtcname = optarg;
        break;
      case 'o':
        strncpy(outname, optarg, MAX_FNAME_LEN);
        break;
      default:
        parseErr++;
    }
    }

    cout << "n_mod=" << n_mod << endl;
    if (!xtcname) {
    usage(argv[0]);
    exit(2);
    }

    // Read input xtc file
    int fd = open(xtcname, O_RDONLY);
    if (fd < 0) {
    fprintf(stderr, "Unable to open file '%s'\n", xtcname);
    exit(2);
    }

    XtcFileIterator iter(fd, BUFSIZE);
    Dgram* dgIn;

    // Prepare output smd.xtc2 file
    FILE* xtcFile = fopen(outname, "w");
    if (!xtcFile) {
    printf("Error opening output xtc file.\n");
    return -1;
    }

    // Read timestamp
    array<time_t, 500> sec_arr = {};
    array<long, 500> nsec_arr = {};
    sec_arr[0] = 0;
    nsec_arr[0] = 0;
    if (tsname != 0) {
    ifstream tsfile(tsname);
    time_t sec = 0;
    long nsec = 0;
    int i = 1;
    while (tsfile >> sec >> nsec) {
        sec_arr[i] = sec;
        nsec_arr[i] = nsec;
        cout << sec_arr[i] << " " << nsec_arr[i] << endl;
        i++;
    }
    printf("found %d timestamps", i);
    }

    // Writing out data
    void* buf = malloc(BUFSIZE);
    unsigned eventId = 0;
    uint64_t nowOffset = 0;
    uint64_t nowDgramSize = 0;
    unsigned nodeId=512; // choose a nodeId that the DAQ will not use.  this is checked in addNames()
    NamesLookup namesLookup;
    NamesId namesId(nodeId, 0);

    printf("\nStart writing offsets.\n");

    Smd smd;
    Dgram* dgOut;
    while ((dgIn = iter.next())) {
        nowDgramSize = (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload());
        dgOut = smd.generate(dgIn, buf, nowOffset, nowDgramSize, namesLookup, namesId);

        save(*dgOut, xtcFile);
        eventId++;
        nowOffset += nowDgramSize;
        if (n_events > 0) {
            if (eventId - 1 >= n_events) {
                cout << "Stop writing. The option -n (no. of events) was set to " << n_events << endl;
                break;
            }
        }


        if (n_mod > 0) {
            if (eventId == 1) {
                cout << eventId << " sleep...3s " << endl;
                sleep(3);
            }else if(eventId % n_mod == 0) {
                cout << eventId << " sleep...1s " << endl;
                sleep(1);
            }
        }
    }// end while((dgIn...

  cout << "Finished writing smd for " << eventId << " events with size (B): " << nowOffset << endl;
  fclose(xtcFile);
  ::close(fd);
  free(buf);

  return 0;

}
