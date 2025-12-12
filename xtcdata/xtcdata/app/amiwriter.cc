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

class ConfigDef: public VarDef {
public:
    enum index { configIndex };
    ConfigDef()  {NameVec.push_back({"fakeValue",Name::CHARSTR,1});}
} ConfigDef;

class RunInfoDef:public VarDef {
public:
    enum index {expIndex, runIndex};
    RunInfoDef() {
        NameVec.push_back({"expt",Name::CHARSTR,1});
        NameVec.push_back({"runnum",Name::UINT32});
    }
} RunInfoDef;

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

class ScanDef: public VarDef {
public:
    enum index { scanIndex };
    ScanDef()  {NameVec.push_back({"motor",Name::FLOAT});}
} ScanDef;

class CspadDef: public VarDef {
public:
    enum index { cspadIndex };
    CspadDef()  {NameVec.push_back({"arrayRaw",Name::UINT16,2});}
} CspadDef;

void addCspad(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
              unsigned value)
{
    CreateData fex(parent, bufEnd, namesLookup, namesId);
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

void addConfig(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId, const char* value)
{
    CreateData config(parent, bufEnd, namesLookup, namesId);
    config.set_string(ConfigDef::configIndex, value);
}

void addLaser(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
             unsigned value)
{
    CreateData fex(parent, bufEnd, namesLookup, namesId);
    fex.set_value(LaserDef::laserIndex, (uint32_t)value);
}

void addEBeam(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
              float value)
{
    CreateData fex(parent, bufEnd, namesLookup, namesId);
    fex.set_value(EBeamDef::ebeamIndex, value);
}

void addHsd(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
            unsigned value)
{
    CreateData fex(parent, bufEnd, namesLookup, namesId);
    unsigned shape[MaxRank] = {50};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(HsdDef::hsdIndex,shape);
    for(unsigned i=0; i<shape[0]; i++){
        arrayT(i) = value;
    };
}

void addScan(Xtc& parent, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
             float value)
{
    CreateData scan(parent, bufEnd, namesLookup, namesId);
    scan.set_value(ScanDef::scanIndex, value);
}

void addRunInfo(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
                const char* exp, uint32_t run) {
    CreateData runinfo(xtc, bufEnd, namesLookup, namesId);
    runinfo.set_string(RunInfoDef::expIndex, exp);
    runinfo.set_value(RunInfoDef::runIndex, run);
}

void addRunInfoNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg runInfoAlg("runinfo",0,0,1);
    Names& runInfoNames = *new(xtc, bufEnd) Names(bufEnd, "runinfo", runInfoAlg, "runinfo", "", namesId, 0);
    runInfoNames.add(xtc, bufEnd, RunInfoDef);
    namesLookup[namesId] = NameIndex(runInfoNames);
}

void addCspadNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
                   unsigned segment) {
    Alg cspadAlg("raw",2,3,42);
    Names& cspadNames = *new(xtc, bufEnd) Names(bufEnd, "xppcspad", cspadAlg, "cspad", "serialnum1234", namesId, segment);
    cspadNames.add(xtc, bufEnd, CspadDef);
    namesLookup[namesId] = NameIndex(cspadNames);
}

void addCspadConfigNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId,
                         unsigned segment) {
    Alg cspadAlg("fakeConfig",0,0,1);
    Names& cspadNames = *new(xtc, bufEnd) Names(bufEnd, "xppcspad", cspadAlg, "cspad", "serialnum1234", namesId, segment);
    cspadNames.add(xtc, bufEnd, ConfigDef);
    namesLookup[namesId] = NameIndex(cspadNames);
}

void addLaserNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg laserAlg("raw",2,3,42);
    Names& laserNames = *new(xtc, bufEnd) Names(bufEnd, "xpplaser", laserAlg, "laser", "serialnum1234", namesId, 0);
    laserNames.add(xtc, bufEnd, LaserDef);
    namesLookup[namesId] = NameIndex(laserNames);
}

void addLaserConfigNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg laserAlg("fakeConfig",0,0,1);
    Names& laserNames = *new(xtc, bufEnd) Names(bufEnd, "xpplaser", laserAlg, "laser", "serialnum1234", namesId, 0);
    laserNames.add(xtc, bufEnd, ConfigDef);
    namesLookup[namesId] = NameIndex(laserNames);
}

void addEBeamNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg ebeamAlg("raw",2,3,42);
    Names& ebeamNames = *new(xtc, bufEnd) Names(bufEnd, "EBeam", ebeamAlg, "ebeam", "serialnum1234", namesId, 0);
    ebeamNames.add(xtc, bufEnd, EBeamDef);
    namesLookup[namesId] = NameIndex(ebeamNames);
}

void addEBeamConfigNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg ebeamAlg("fakeConfig",0,0,1);
    Names& ebeamNames = *new(xtc, bufEnd) Names(bufEnd, "EBeam", ebeamAlg, "ebeam", "serialnum1234", namesId, 0);
    ebeamNames.add(xtc, bufEnd, ConfigDef);
    namesLookup[namesId] = NameIndex(ebeamNames);
}

void addHsdNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg hsdAlg("raw",2,3,42);
    Names& hsdNames = *new(xtc, bufEnd) Names(bufEnd, "xpphsd", hsdAlg, "hsd", "serialnum1234", namesId, 0);
    hsdNames.add(xtc, bufEnd, HsdDef);
    namesLookup[namesId] = NameIndex(hsdNames);
}

void addScanNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg scanAlg("raw",2,0,0);
    Names& scanNames = *new(xtc, bufEnd) Names(bufEnd, "scan", scanAlg, "scan", "serialnum1234", namesId, 0);
    scanNames.add(xtc, bufEnd, ScanDef);
    namesLookup[namesId] = NameIndex(scanNames);
}

void addHsdConfigNames(Xtc& xtc, const void* bufEnd, NamesLookup& namesLookup, NamesId& namesId) {
    Alg hsdAlg("fakeConfig",0,0,1);
    Names& hsdNames = *new(xtc, bufEnd) Names(bufEnd, "xpphsd", hsdAlg, "hsd", "serialnum1234", namesId, 0);
    hsdNames.add(xtc, bufEnd, ConfigDef);
    namesLookup[namesId] = NameIndex(hsdNames);
}

Dgram& createTransition(TransitionId::Value transId, bool counting_timestamps,
                        unsigned& timestamp_val, void** bufEnd) {
    TypeId tid(TypeId::Parent, 0);
    uint64_t pulseId = 0;
    uint32_t env = 0;
    struct timeval tv;
    void* buf = malloc(BUFSIZE);
    *bufEnd = ((char*)buf) + BUFSIZE;
    if (counting_timestamps) {
        tv.tv_sec = 0;
        tv.tv_usec = timestamp_val;
        timestamp_val++;
    } else {
        gettimeofday(&tv, NULL);
        // convert to ns for the Timestamp
        tv.tv_usec *= 1000;
    }
    Transition tr(Dgram::Event, transId, TimeStamp(tv.tv_sec, tv.tv_usec), env);
    return *new(buf) Dgram(tr, Xtc(tid));
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
    unsigned nmotorsteps = 1;
    unsigned timestamp_period = 0; // time in us
    char xtcname[MAX_FNAME_LEN];
    strncpy(xtcname, "ami.xtc2", MAX_FNAME_LEN);
    unsigned starting_segment = 0;
    // this is used to create uniform timestamps across files
    // so we can do offline event-building.
    bool counting_timestamps = false;
    bool add_fake_configs = false;
    auto usage = [](const char* progname) {
        fprintf(stderr, "Usage: %s [-f <filename> -n <numEvents> -t -c -p <period (us)> -h]\n", progname);
    };

    while ((c = getopt(argc, argv, "hf:n:m:s:tcp:")) != -1) {
        switch (c) {
            case 'h':
                usage(argv[0]);
                exit(0);
            case 'n':
                nevents = atoi(optarg);
                break;
            case 'm':
                nmotorsteps = atoi(optarg);
                break;
            case 'f':
                strncpy(xtcname, optarg, MAX_FNAME_LEN);
                break;
            case 's':
                starting_segment = atoi(optarg);
                break;
            case 't':
                counting_timestamps = true;
                break;
            case 'p':
                timestamp_period = atoi(optarg);
                break;
            case 'c':
                add_fake_configs = true;
                break;
            default:
                parseErr++;
        }
    }

    if (parseErr) {
        usage(argv[0]);
        return -1;
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
    unsigned timestamp_val = 0;
    void* bufEnd;

    Dgram& config = createTransition(TransitionId::Configure,
                                     counting_timestamps,
                                     timestamp_val,
                                     &bufEnd);

    unsigned nodeId = 1;
    NamesLookup namesLookup;
    unsigned nSegments=2;
    unsigned segmentIndex = starting_segment;

    NamesId namesIdRunInfo(nodeId,segmentIndex++);
    addRunInfoNames(config.xtc, bufEnd, namesLookup, namesIdRunInfo);

    // only add epics and scan info to the first stream
    NamesId namesIdScan(nodeId,segmentIndex++);
    if (starting_segment==0) {
        addScanNames(config.xtc, bufEnd, namesLookup, namesIdScan);
    }

    NamesId namesIdCspad[] = {NamesId(nodeId,segmentIndex++), NamesId(nodeId,segmentIndex++)};
    for (unsigned iseg=0; iseg<nSegments; iseg++) {
        addCspadNames(config.xtc, bufEnd, namesLookup, namesIdCspad[iseg], iseg);
    }

    NamesId namesIdLaser(nodeId,segmentIndex++);
    addLaserNames(config.xtc, bufEnd, namesLookup, namesIdLaser);

    NamesId namesIdEBeam(nodeId,segmentIndex++);
    addEBeamNames(config.xtc, bufEnd, namesLookup, namesIdEBeam);

    NamesId namesIdHsd(nodeId,segmentIndex++);
    addHsdNames(config.xtc, bufEnd, namesLookup, namesIdHsd);

    if (add_fake_configs) {
        NamesId namesIdCspadConfig[] = {NamesId(nodeId,segmentIndex++), NamesId(nodeId,segmentIndex++)};
        for (unsigned iseg=0; iseg<nSegments; iseg++) {
            addCspadConfigNames(config.xtc, bufEnd, namesLookup, namesIdCspadConfig[iseg], iseg);
            addConfig(config.xtc, bufEnd, namesLookup, namesIdCspadConfig[iseg], "I am a cspad!");
        }

        NamesId namesIdLaserConfig(nodeId,segmentIndex++);
        addLaserConfigNames(config.xtc, bufEnd, namesLookup, namesIdLaserConfig);
        addConfig(config.xtc, bufEnd, namesLookup, namesIdLaserConfig, "I am a laser!");

        NamesId namesIdEBeamConfig(nodeId,segmentIndex++);
        addEBeamConfigNames(config.xtc, bufEnd, namesLookup, namesIdEBeamConfig);
        addConfig(config.xtc, bufEnd, namesLookup, namesIdEBeamConfig, "I am an ebeam!");

        NamesId namesIdHsdConfig(nodeId,segmentIndex++);
        addHsdConfigNames(config.xtc, bufEnd, namesLookup, namesIdHsdConfig);
        addConfig(config.xtc, bufEnd, namesLookup, namesIdHsdConfig, "I am an hsd!");
    }

    save(config,xtcFile);

    Dgram& beginRunTr = createTransition(TransitionId::BeginRun,
                                         counting_timestamps,
                                         timestamp_val,
                                         &bufEnd);
    addRunInfo(beginRunTr.xtc, bufEnd, namesLookup, namesIdRunInfo, "xpptut15", 15);
    save(beginRunTr, xtcFile);

    for (unsigned istep=0; istep<nmotorsteps; istep++) {

        Dgram& beginStepTr = createTransition(TransitionId::BeginStep,
                                              counting_timestamps,
                                              timestamp_val,
                                              &bufEnd);
        if (starting_segment==0) addScan(beginStepTr.xtc, bufEnd, namesLookup, namesIdScan, istep);
        save(beginStepTr, xtcFile);

        void* buf = malloc(BUFSIZE);
        for (int i = 0; i < nevents; i++) {
            if (counting_timestamps) {
                tv.tv_sec = 0;
                tv.tv_usec = timestamp_val;
                timestamp_val++;
            } else {
              usleep(timestamp_period);
              gettimeofday(&tv, NULL);
              // convert to ns for the Timestamp
              tv.tv_usec *= 1000;
            }
            Transition tr(Dgram::Event, TransitionId::L1Accept, TimeStamp(tv.tv_sec, tv.tv_usec), env);
            Dgram& dgram = *new(buf) Dgram(tr, Xtc(tid));

            for (unsigned iseg=0; iseg<nSegments; iseg++) {
                addCspad(dgram.xtc, bufEnd, namesLookup, namesIdCspad[iseg], i);
            }
            addLaser(dgram.xtc, bufEnd, namesLookup, namesIdLaser, i%2);
            addEBeam(dgram.xtc, bufEnd, namesLookup, namesIdEBeam, (float)i);
            if (i%2==0) {
                addHsd(dgram.xtc, bufEnd, namesLookup, namesIdHsd, i);
            }

            save(dgram,xtcFile);
        } // events


        Dgram& disableTr = createTransition(TransitionId::Disable,
                                            counting_timestamps,
                                            timestamp_val,
                                            &bufEnd);
        save(disableTr, xtcFile);

        Dgram& endStepTr = createTransition(TransitionId::EndStep,
                                            counting_timestamps,
                                            timestamp_val,
                                            &bufEnd);
        save(endStepTr, xtcFile);
    } // steps

    // make an EndRun
    Dgram& endRunTr = createTransition(TransitionId::EndRun,
                                       counting_timestamps,
                                       timestamp_val,
                                       &bufEnd);
    save(endRunTr, xtcFile);

    fclose(xtcFile);

    return 0;
}
