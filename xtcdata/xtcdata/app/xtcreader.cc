#include <fcntl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <cinttypes>

#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"

using namespace XtcData;
using std::string;

template<typename T> static void _dump(const char* name,  Array<T> arrT, unsigned numWords, unsigned* shape, unsigned rank, const char* fmt)
{
    unsigned maxWords = 1;
    printf("'%s' ", name);
    printf("(shape:");
    for (unsigned w = 0; w < rank; w++) {
        printf(" %d",shape[w]);
        maxWords *= shape[w];
    }
    printf("): ");
    if (maxWords < numWords)
        numWords = maxWords;
    for (unsigned w = 0; w < numWords; ++w) {
        printf(fmt, arrT.data()[w]);
    }
    printf("\n");
}

class DebugIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    DebugIter(unsigned numWords) : XtcIterator(), _numWords(numWords)
    {
    }

    void get_value(int i, Name& name, DescData& descdata){
        int data_rank = name.rank();
        int data_type = name.type();
        printf("%d: '%s' rank %d, type %d\n", i, name.name(), data_rank, data_type);

        switch(name.type()){
        case(Name::UINT8):{
            if(data_rank > 0){
                _dump<uint8_t>(name.name(), descdata.get_array<uint8_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint8_t>(i));
            }
            break;
        }

        case(Name::UINT16):{
            if(data_rank > 0){
                _dump<uint16_t>(name.name(), descdata.get_array<uint16_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint16_t>(i));
            }
            break;
        }

        case(Name::UINT32):{
            if(data_rank > 0){
                _dump<uint32_t>(name.name(), descdata.get_array<uint32_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<uint32_t>(i));
            }
            break;
        }

        case(Name::UINT64):{
            if(data_rank > 0){
                _dump<uint64_t>(name.name(), descdata.get_array<uint64_t>(i), _numWords, descdata.shape(name), name.rank(), " %ld");
            }
            else{
                printf("'%s': %llu\n",name.name(),descdata.get_value<uint64_t>(i));
            }
            break;
        }

        case(Name::INT8):{
            if(data_rank > 0){
                _dump<int8_t>(name.name(), descdata.get_array<int8_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int8_t>(i));
            }
            break;
        }

        case(Name::INT16):{
            if(data_rank > 0){
                _dump<int16_t>(name.name(), descdata.get_array<int16_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int16_t>(i));
            }
            break;
        }

        case(Name::INT32):{
            if(data_rank > 0){
                _dump<int32_t>(name.name(), descdata.get_array<int32_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::INT64):{
            if(data_rank > 0){
                _dump<int64_t>(name.name(), descdata.get_array<int64_t>(i), _numWords, descdata.shape(name), name.rank(), " %ld");
            }
            else{
                printf("'%s': %lld\n",name.name(),descdata.get_value<int64_t>(i));
            }
            break;
        }

        case(Name::FLOAT):{
            if(data_rank > 0){
                _dump<float>(name.name(), descdata.get_array<float>(i), _numWords, descdata.shape(name), name.rank(), " %f");
            }
            else{
                printf("'%s': %f\n",name.name(),descdata.get_value<float>(i));
            }
            break;
        }

        case(Name::DOUBLE):{
            if(data_rank > 0){
                _dump<double>(name.name(), descdata.get_array<double>(i), _numWords, descdata.shape(name), name.rank(), " %f");
            }
            else{
                printf("'%s': %f\n",name.name(),descdata.get_value<double>(i));
            }
            break;
        }

        case(Name::CHARSTR):{
            if(data_rank > 0){
                Array<char> arrT = descdata.get_array<char>(i);
                printf("'%s': \"%s\"\n",name.name(),arrT.data());
            }
            else{
                printf("'%s': string with no rank?!?\n",name.name());
            }
            break;
        }

        case(Name::ENUMVAL):{
            if(data_rank > 0){
                _dump<int32_t>(name.name(), descdata.get_array<int32_t>(i), _numWords, descdata.shape(name), name.rank(), " %d");
            }
            else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }

        case(Name::ENUMDICT):{
            if(data_rank > 0){
                printf("'%s': enumdict with rank?!?\n", name.name());
            } else{
                printf("'%s': %d\n",name.name(),descdata.get_value<int32_t>(i));
            }
            break;
        }
        }
    }

    int process(Xtc* xtc, const void* bufEnd)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc, bufEnd);
            break;
        }
        case (TypeId::Names): {
            Names& names = *(Names*)xtc;
            _namesLookup[names.namesId()] = NameIndex(names);
            Alg& alg = names.alg();
	    printf("*** DetName: %s, Segment %d, DetType: %s, DetId: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
                   names.detName(), names.segment(), names.detType(), names.detId(),
                   alg.name(), alg.version(), (int)names.namesId());

            for (unsigned i = 0; i < names.num(); i++) {
                Name& name = names.get(i);
                printf("Name: '%s' Type: %d Rank: %d\n",name.name(),name.type(), name.rank());
            }

            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            if (_namesLookup.count(namesId)<=0) {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
                break;
            }
            DescData descdata(shapesdata, _namesLookup[namesId]);
            Names& names = descdata.nameindex().names();
            Data& data = shapesdata.data();
	    printf("Found %d names for namesid 0x%x\n",names.num(),namesId);
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
    unsigned _numWords;
};

class Stats {
public:
  Stats() : _nl1step(0),_nl1(0),_nstep(0),_ndamage(0) {}
  void update(Dgram* dg) {
    TransitionId::Value trans = dg->service();
    if (trans==TransitionId::L1Accept) {
      _nl1step++;
      _nl1++;
      if (dg->xtc.damage.value()) _ndamage++;
    }
    if (trans==TransitionId::EndStep) {
      printf("Step %d has %d L1Accepts totL1Accept %d nDamage %d\n",_nstep,_nl1step,_nl1,_ndamage);
      _nl1step=0;
      _nstep++;
    }
  }
private:
  unsigned _nl1step;
  unsigned _nl1;
  unsigned _nstep;
  unsigned _ndamage;
};

void usage(char* progname)
{
    fprintf(stderr, "Usage: %s -f <filename> [-d] [-n <nEvents>] [-w <nWords>] [-S <max dgram bytes>] [-h] [-s]\n", progname);
}

int main(int argc, char* argv[])
{
    int c;
    char* xtcname = 0;
    char* cfg_xtcname = 0;
    int parseErr = 0;
    unsigned neventreq = 0xffffffff;
    bool debugprint = false;
    unsigned numWords = 3;
    bool printTimeAsUnsignedLong = false;
    bool doStats = false;
    unsigned maxSize = 0x4000000;
    Stats stats;

    while ((c = getopt(argc, argv, "hf:n:dw:c:TsS:")) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(0);
        case 'f':
            xtcname = optarg;
            break;
        case 'n':
            neventreq = atoi(optarg);
            break;
        case 'd':
            debugprint = true;
            break;
        case 'w':
            numWords = atoi(optarg);
            break;
        case 'c':
            cfg_xtcname = optarg;
            break;
        case 's':
            doStats = true;
            break;
        case 'S':
            maxSize = strtoul(optarg,NULL,0);
            break;
        case 'T':
            printTimeAsUnsignedLong = true;
            break;
        default:
            parseErr++;
        }
    }

    if (!xtcname) {
        usage(argv[0]);
        exit(2);
    }

    int fd = open(xtcname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
        exit(2);
    }


    // Open xtc file for config if given
    int cfg_fd = -1;
    Dgram* dg;
    DebugIter dbgiter(numWords);
    if (cfg_xtcname) {
        cfg_fd = open(cfg_xtcname, O_RDONLY);
        if (cfg_fd < 0) {
            fprintf(stderr, "Unable to open config file '%s'\n", cfg_xtcname);
            exit(2);
        }

        // Use config from given config file as default
        XtcFileIterator cfg_iter(cfg_fd, maxSize);

        // Only access the first dgram (config)
        dg = cfg_iter.next();
        const void* bufEnd = ((char*)dg) + maxSize;

        // dbgiter will remember all the configs
        if (debugprint) dbgiter.iterate(&(dg->xtc), bufEnd);

    }

    XtcFileIterator iter(fd, maxSize);
    unsigned nevent=0;
    dg = iter.next();
    const void* bufEnd = ((char*)dg) + maxSize;
    while (dg) {
        if (nevent>=neventreq) break;
        nevent++;
	if (!doStats) {
	  printf("event %d, %11s transition: ",
		 nevent,
		 TransitionId::name(dg->service()));
	  if (printTimeAsUnsignedLong) {
            printf(" time %" PRIu64 ", ", dg->time.value());
	  } else {
            printf(" time 0x%8.8x.0x%8.8x, ", dg->time.seconds(), dg->time.nanoseconds());
	  }
	  printf(" env 0x%08x, payloadSize %d damage 0x%x extent %d\n",
		 dg->env, dg->xtc.sizeofPayload(),dg->xtc.damage.value(),dg->xtc.extent);
	  if (debugprint) dbgiter.iterate(&(dg->xtc), bufEnd);
	} else {
	  stats.update(dg);
	}
        dg = iter.next();
    }

    if (cfg_fd >= 0) {
        ::close(cfg_fd);
    }
    ::close(fd);
    return 0;
}
