#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>

using namespace XtcData;

#define BUFSIZE 0x4000000

class LaserDef: public VarDef {
public:
    enum index { laserIndex };
    LaserDef()  {NameVec.push_back({"laserOn",Name::UINT32});}
} LaserDef;

class EBeamDef: public VarDef {
public:
    enum index { ebeamIndex };
    EBeamDef()  {NameVec.push_back({"energy",Name::FLOAT});}
} EBeamDef;

class HsdDef: public VarDef {
public:
    enum index { hsdIndex };
    HsdDef()  {NameVec.push_back({"waveform",Name::UINT16,1});}
} HsdDef;

class CspadDef: public VarDef {
public:
    enum index { cspadIndex };
    CspadDef()  {NameVec.push_back({"arrayRaw",Name::UINT16,2});}
} CspadDef;

void addCspad(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId,
              unsigned value)
{ 
    CreateData fex(parent, namesLookup, namesId);
    unsigned shape[MaxRank] = {50,100};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(CspadDef::cspadIndex,shape);
    unsigned factors[4];
    for (unsigned k=0; k<4; k++)
      factors[k] = (value + k) % 4;
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            if (i < shape[0] / 2) {
              if (j < shape[1] / 2)
                arrayT(i,j) = value * factors[0];
              else
                arrayT(i,j) = value * factors[1];
            } else {
              if (j < shape[1] / 2)
                arrayT(i,j) = value * factors[2];
              else
                arrayT(i,j) = value * factors[3];
            }
        }
    };
}

void addLaser(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId,
             unsigned value)
{ 
    CreateData fex(parent, namesLookup, namesId);
    fex.set_value(LaserDef::laserIndex, (uint32_t)value);
}

void addEBeam(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId,
              float value)
{ 
    CreateData fex(parent, namesLookup, namesId);
    fex.set_value(EBeamDef::ebeamIndex, value);
}

void addHsd(Xtc& parent, NamesLookup& namesLookup, NamesId& namesId,
            unsigned value)
{ 
    CreateData fex(parent, namesLookup, namesId);
    unsigned shape[MaxRank] = {50};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(HsdDef::hsdIndex,shape);
    for(unsigned i=0; i<shape[0]; i++){
        arrayT(i) = value;
    };
}

void addCspadNames(Xtc& xtc, NamesLookup& namesLookup, NamesId& namesId,
                   unsigned segment) {
    Alg cspadAlg("raw",2,3,42);
    Names& cspadNames = *new(xtc) Names("xppcspad", cspadAlg, "cspad", "serialnum1234", namesId, segment);
    cspadNames.add(xtc, CspadDef);
    namesLookup[namesId] = NameIndex(cspadNames);
}

void addLaserNames(Xtc& xtc, NamesLookup& namesLookup, NamesId& namesId) {
    Alg laserAlg("raw",2,3,42);
    unsigned segment = 0;
    Names& laserNames = *new(xtc) Names("xpplaser", laserAlg, "laser", "serialnum1234", namesId, 0);
    laserNames.add(xtc, LaserDef);
    namesLookup[namesId] = NameIndex(laserNames);
}

void addEBeamNames(Xtc& xtc, NamesLookup& namesLookup, NamesId& namesId) {
    Alg ebeamAlg("raw",2,3,42);
    unsigned segment = 0;
    Names& ebeamNames = *new(xtc) Names("EBeam", ebeamAlg, "ebeam", "serialnum1234", namesId, 0);
    ebeamNames.add(xtc, EBeamDef);
    namesLookup[namesId] = NameIndex(ebeamNames);
}

void addHsdNames(Xtc& xtc, NamesLookup& namesLookup, NamesId& namesId) {
    Alg hsdAlg("raw",2,3,42);
    unsigned segment = 0;
    Names& hsdNames = *new(xtc) Names("xpphsd", hsdAlg, "hsd", "serialnum1234", namesId, 0);
    hsdNames.add(xtc, HsdDef);
    namesLookup[namesId] = NameIndex(hsdNames);
}

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s [-f <filename> -n <numEvents> -t -h]\n", progname);
}

Dgram& createTransition(TransitionId::Value transId) {
    TypeId tid(TypeId::Parent, 0);
    uint64_t pulseId = 0;
    uint32_t env = 0;
    struct timeval tv;
    void* buf = malloc(BUFSIZE);
    gettimeofday(&tv, NULL);
    Sequence seq(Sequence::Event, transId, TimeStamp(tv.tv_sec, tv.tv_usec * 1000), PulseId(pulseId,0));
    return *new(buf) Dgram(Transition(seq, env), Xtc(tid));
}

void save(Dgram& dg, FILE* xtcFile) {
    if (fwrite(&dg, sizeof(dg) + dg.xtc.sizeofPayload(), 1, xtcFile) != 1) {
        printf("Error writing to output xtc file.\n");
    }
}

#define MAX_FNAME_LEN 256

int main(int argc, char* argv[])
{
    int c;
    int parseErr = 0;
    unsigned nevents = 2;
    char xtcname[MAX_FNAME_LEN];
    strncpy(xtcname, "ami.xtc2", MAX_FNAME_LEN);

    while ((c = getopt(argc, argv, "htf:n:")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 'n':
                nevents = atoi(optarg);
                break;
            case 'f':
                strncpy(xtcname, optarg, MAX_FNAME_LEN);
                break;
            default:
                parseErr++;
        }
    }

    FILE* xtcFile = fopen(xtcname, "w");
    if (!xtcFile) {
        printf("Error opening output xtc file.\n");
        return -1;
    }

    struct timeval tv;
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    uint64_t pulseId = 0;

    Dgram& config = createTransition(TransitionId::Configure);

    unsigned nodeId = 1;
    NamesLookup namesLookup;
    unsigned nSegments=2;

    NamesId namesIdCspad[] = {NamesId(nodeId,0+10*0), NamesId(nodeId,0+10*1)};
    for (unsigned iseg=0; iseg<nSegments; iseg++) {
        addCspadNames(config.xtc, namesLookup, namesIdCspad[iseg], iseg);
    }

    NamesId namesIdLaser(nodeId,1);
    addLaserNames(config.xtc, namesLookup, namesIdLaser);

    NamesId namesIdEBeam(nodeId,2);
    addEBeamNames(config.xtc, namesLookup, namesIdEBeam);

    NamesId namesIdHsd(nodeId,3);
    addHsdNames(config.xtc, namesLookup, namesIdHsd);

    save(config,xtcFile);

    void* buf = malloc(BUFSIZE);
    for (int i = 0; i < nevents; i++) {
        gettimeofday(&tv, NULL);
        Sequence seq(Sequence::Event, TransitionId::L1Accept, TimeStamp(tv.tv_sec, tv.tv_usec * 1000), PulseId(pulseId,0));
        Dgram& dgram = *new(buf) Dgram(Transition(seq, env), Xtc(tid));

        for (unsigned iseg=0; iseg<nSegments; iseg++) {
            addCspad(dgram.xtc, namesLookup, namesIdCspad[iseg], i);
        }
        addLaser(dgram.xtc, namesLookup, namesIdLaser, i%2);
        addEBeam(dgram.xtc, namesLookup, namesIdEBeam, (float)i);
        if (i%2==0) {
            addHsd(dgram.xtc, namesLookup, namesIdHsd, i);
        }

        save(dgram,xtcFile);
     }

    fclose(xtcFile);

    return 0;
}
