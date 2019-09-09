//--------------------------------------------
// Adapted on 2019-08-28 by Mikhail Dubrovin
//--------------------------------------------

#include "roentdek/resort64c.h"
#include "psalg/hexanode/wrap_resort64c.hh"
#include "psalg/hexanode/cfib.hh"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//-----------------

void test01()
{
  cout << "ex01_resort64c.test01 ==> tests peak_tracker_class\n";
  double w = 0.1;
  peak_tracker_class* tracker = new peak_tracker_class(20000,-w,w,int(2.*w/0.025+1.e-6)+1);

  delete tracker;
}

//-----------------

void test02()
{
  cout << "ex01_resort64c.test02 ==> tests wrap_resort64c.hh/cc\n";
  test_resort();
}

//-----------------

void test03()
{
  cout << "ex01_resort64c.test03 ==> tests cfib.hh/cc\n";
  int n = 9;
  double v = cfib(9);
  //double v = psalgos::cfib(9);
  cout << "cfib(" << n << ") = " << v << '\n';
}

//-----------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 
  // atoi(argv[1])==1) 
  
  test01();
  test02();
  test03();
  return 0;
}

//-----------------
