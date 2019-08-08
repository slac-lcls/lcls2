
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

// additions from xtc writer
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"

#include "xtcdata/xtc/XtcIterator.hh"

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
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<uint64_t>(i));
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
                printf("%s: %d, %d, %d\n",name.name(),arrT.data()[0],arrT.data()[1], arrT.data()[2]);
                    }
            else{
                printf("%s: %d\n",name.name(),descdata.get_value<int64_t>(i));
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


class SmdDef:public VarDef
{
public:
  enum index
    {
      intOffset,
      intDgramSize
    };

   SmdDef()
   {
     NameVec.push_back({"intOffset", Name::UINT64});
     NameVec.push_back({"intDgramSize", Name::UINT64});
   }
} SmdDef;

void addNames(Xtc& parent, NamesLookup& namesLookup, unsigned nodeId)
{
    Alg alg("offsetAlg",0,0,0);
    NamesId namesId(nodeId,0);
    Names& offsetNames = *new(parent) Names("info", alg, "offset", "", namesId);
    offsetNames.add(parent,SmdDef);
    namesLookup[namesId] = NameIndex(offsetNames);
}

void save(Dgram& dg, FILE* xtcFile) {
    if (fwrite(&dg, sizeof(dg) + dg.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
}

void show(Dgram& dg) {
    printf("%s transition: time %d.%09d, pulseId 0x%lu, env 0x%lu, "
           "payloadSize %d extent %d\n",
           TransitionId::name(dg.seq.service()), dg.seq.stamp().seconds(),
           dg.seq.stamp().nanoseconds(), dg.seq.pulseId().value(),
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
    char outname[MAX_FNAME_LEN];
    strncpy(outname, "smd.xtc2", MAX_FNAME_LEN);

    while ((c = getopt(argc, argv, "ht:n:f:o:")) != -1) {
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
    unsigned eventL1Id = 0;
    unsigned eventUpdateId = 0;
    unsigned eventId = 0;
    uint64_t nowOffset = 0;
    uint64_t nowDgramSize = 0;
    struct timeval tv;
    uint64_t pulseId = 0;
    unsigned nodeId=0;
    NamesLookup namesLookup;

    printf("\nStart writing offsets.\n"); 
    
    while ((dgIn = iter.next())) {
        if (dgIn->seq.service() == TransitionId::Configure) {
            nowOffset += (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload()); // add up offset before appending info name.
            addNames(dgIn->xtc, namesLookup, nodeId);
            save(*dgIn, xtcFile);
        } else if (dgIn->seq.service() == TransitionId::SlowUpdate || dgIn->seq.service() == TransitionId::Enable || dgIn->seq.service() == TransitionId::Disable) {
            save(*dgIn, xtcFile);
            eventUpdateId++;
            nowOffset += (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload());    
        } else if (dgIn->seq.service() == TransitionId::L1Accept) {
            Dgram& dgOut = *(Dgram*)buf;
            TypeId tid(TypeId::Parent, 0);
            dgOut.xtc.contains = tid;
            dgOut.xtc.damage = 0;
            dgOut.xtc.extent = sizeof(Xtc);

            if (writeTs != 0) {
                if (tsname != 0) {
                    dgOut.seq = Sequence(TimeStamp(sec_arr[eventL1Id], nsec_arr[eventL1Id]), PulseId(pulseId));
                } else {
                    gettimeofday(&tv, NULL);
                    dgOut.seq = Sequence(TimeStamp(tv.tv_sec, tv.tv_usec), PulseId(pulseId));
                    cout << "warning: new timestamp (not from bigdata xtc) " << tv.tv_sec << " " << tv.tv_usec << endl;
                }
            } else {
                dgOut.seq = dgIn->seq;
            }

            NamesId namesId(nodeId,0);
            CreateData smd(dgOut.xtc, namesLookup, namesId);
            smd.set_value(SmdDef::intOffset, nowOffset);
            nowDgramSize = (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload());
            smd.set_value(SmdDef::intDgramSize, nowDgramSize);

            if (nowOffset < 0) {
                cout << "Error offset value (offset=" << nowOffset << ")" << endl;
                return -1;
            }
            if (nowDgramSize <= 0) {
                cout << "Error size value (size=" << nowDgramSize << ")" << endl;
                return -1;
            }

            save(dgOut, xtcFile);
            eventL1Id++;
            nowOffset += (uint64_t)(sizeof(*dgIn) + dgIn->xtc.sizeofPayload());    

            if (n_events > 0) {
                if (eventL1Id - 1 >= n_events) {
                    cout << "Stop writing. The option -n (no. of events) was set to " << n_events << endl;
                    break;
                }
            }
        }// end if (dgIn->
        
        eventId++;
    }// end while((dgIn...

  cout << "Finished writing smd for " << eventL1Id - 1 << " L1 events and " << eventUpdateId << " update events. Big data file has " << eventId << " events with size (B): " << nowOffset << endl;
  fclose(xtcFile);
  ::close(fd);
  
  return 0;

}
