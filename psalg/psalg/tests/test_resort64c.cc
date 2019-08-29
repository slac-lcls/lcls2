//--------------------------------------------
// Adapted on 2019-08-28 by Mikhail Dubrovin
//--------------------------------------------

//#include "hexanode/xxx.h"

// ~/lib/hexanode-lib/sort_non-LMF_from_1_detector/resort64c.h

#include "roentdek/resort64c.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//-----------------

int test01 ()
{
  cout << "ex_hexanode.test01\n";
  return 0;
}

//-----------------

int test02 ()
{
  cout << "ex_hexanode.test02\n";
  cout << "  hexanode/peak_tracker_class\n";
  double w = 0.1;
  peak_tracker_class* tracker = new peak_tracker_class(20000,-w,w,int(2.*w/0.025+1.e-6)+1);

  delete tracker;
  return 0;
}

//-----------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 
  // atoi(argv[1])==1) 
  
  test01();
  test02();

  return 0;
}

//-----------------
