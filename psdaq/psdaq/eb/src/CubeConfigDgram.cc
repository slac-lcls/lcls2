#include "CubeConfigDgram.hh"
#include "string.h"

using namespace Pds::Eb;

void CubeConfigDgram::resultType(char* rtype)
{
    ResultType rt = Base;
    if (strcmp(rtype,"Cube")==0)
        rt = Cube;
    else if (strcmp(rtype,"Window")==0)
        rt = Window;
    monBufNo(rt); 
}

void CubeConfigDgram::appendJson   (char*    json) {
    unsigned len = strlen(json)+1;
    memcpy( xtc.alloc(len, 0), json, len);
}

