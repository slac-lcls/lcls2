
#include "xtcdata/xtc/XtcUpdateIter.hh"

using namespace XtcData;
using std::string;

template<typename T> static void _dump(const char* name,  Array<T> arrT, unsigned numWords, unsigned* shape, unsigned rank, const char* fmt)
{
    printf("'%s' ", name);
    printf(" numWords:%u rank:%u ", numWords, rank); 
    printf("(shape:");
    for (unsigned w = 0; w < rank; w++) printf(" %d",shape[w]);
    printf("): ");
    for (unsigned w = 0; w < numWords; ++w) {
        printf(fmt, arrT.data()[w]);
    }
    printf("\n");
}

void XtcUpdateIter::get_value(int i, Name& name, DescData& descdata){
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

int XtcUpdateIter::process(Xtc* xtc)
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
    printf("*** DetName: %s, Segment %d, DetType: %s, DetId: %s, Alg: %s, Version: 0x%6.6x, namesid: 0x%x, Names:\n",
               names.detName(), names.segment(), names.detType(), names.detId(),
               alg.name(), alg.version(), (int)names.namesId());

        for (unsigned i = 0; i < names.num(); i++) {
            Name& name = names.get(i);
            printf("Name: '%s' Type: %d Rank: %d\n",name.name(),name.type(), name.rank());
        }

        unsigned namesSize = sizeof(Names) + (names.num() * sizeof(Name)); 
        printf("BEFORE sizeof(Names):%u names.num():%u sizeof(Name):%u total:%u sizeof(Xtc):%u sizeofPayload:%d\n", sizeof(Names), names.num(), sizeof(Name), namesSize, sizeof(Xtc), xtc->sizeofPayload());

        // copy Names to out buffer
        copy2buf((char*)xtc, sizeof(Xtc) + xtc->sizeofPayload());
        printf("COPY Names sizeof(Xtc):%u sizeofPayload: %u bufsize: %u\n", sizeof(Xtc), xtc->sizeofPayload(), _bufsize);

        break;
    }
    case (TypeId::ShapesData): {
        ShapesData& shapesdata = *(ShapesData*)xtc;
        // lookup the index of the names we are supposed to use
        NamesId namesId = shapesdata.namesId();
        // if this is the namesId that we want (raw.fex), copy it
        
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
        
        Alg& alg = names.alg();
    printf("*** DetName: %s, Alg: %s\n", names.detName(), alg.name());
    
    printf("Found %d names\n",names.num());
        for (unsigned i = 0; i < names.num(); i++) {
            Name& name = names.get(i);
            get_value(i, name, descdata);
        }

        // copy ShapesData to out buffer
        copy2buf((char*)xtc, sizeof(Xtc) + xtc->sizeofPayload());
        printf("COPY ShapesData sizeof(Xtc):%u sizeofPayload: %u bufsize: %u\n", sizeof(Xtc), xtc->sizeofPayload(), _bufsize);
        break;
    }
    default:
        break;
    }
    return Continue;
}

void XtcUpdateIter::copy2buf(char* in_buf, unsigned in_size){
    memcpy(_buf+get_bufsize(), in_buf, in_size);
    _bufsize += in_size;
}


void XtcUpdateIter::addNames(Xtc& xtc, char* detName, char* detType, char* detId, 
        unsigned nodeId, unsigned namesId, unsigned segment,
        char* algName, uint8_t major, uint8_t minor, uint8_t micro,
        NewDef& newdef) 
{
    Alg alg0(algName,major,minor,micro);
    NamesId namesId0(nodeId, namesId);
    Names& names0 = *new(xtc) Names(detName, alg0, detType, detId, namesId0, segment);
    names0.add(xtc, newdef);
    _namesLookup[namesId0] = NameIndex(names0);
}

void XtcUpdateIter::addData(Xtc& xtc, unsigned nodeId, unsigned namesId,
        unsigned* shape, char* data, NewDef& newdef) {
    for(unsigned i=0; i<MaxRank; i++){
        printf("shape[%u]: %u ", i, shape[i]);
    }
    printf("\n");

    uint8_t* cdata = (uint8_t*) data;
    for(unsigned i=0; i<6; i++){
        printf("data[%u]: %u", i, cdata[i]);
    }
    printf("\n");
    
    NamesId namesId0(nodeId, namesId);
    CreateData newData(xtc, _namesLookup, namesId0);

    int newIndex = 0;

    //newData.set_value(newIndex, *cdata);
    //newData.set_array_shape(newIndex, shape);

    Array<uint8_t> arrayT = newData.allocate<uint8_t>(newIndex, shape);
    Shape shp(shape);
    Name& name = _namesLookup[namesId0].names().get(newIndex);
    printf("copy to ArrayT size=%u\n", shp.size(name)); 
    memcpy(arrayT.data(), cdata, shp.size(name));
    
    //Array<uint8_t> arrayI(cdata, shape, name.rank());
    //arrayT = arrayI;
    
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            //arrayT(i,j) = 142+i*shape[1]+j;
            printf("arrayT(%u,%u)=%u\n", i, j, arrayT(i,j));
        }
    }
}
