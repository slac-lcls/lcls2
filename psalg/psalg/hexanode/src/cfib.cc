
//#include "psalg/hexanode/cfib.hh"
#include "../cfib.hh"

//namespace psalgos {

double cfib(int n) {
    int i;
    double a=0.0, b=1.0, tmp;
    for (i=0; i<n; ++i) {
        tmp = a; a = a + b; b = tmp;
    }
    return a;
}

//} //namespace psalgos
