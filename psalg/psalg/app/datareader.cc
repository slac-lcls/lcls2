// Build locally
// g++ datareader.cc -o mytest datareader

// Build in lcls2/install/bin/
// see .../psalg/psalg/CMakeLists.txt
// cd /reg/neh/home/dubrovin/LCLS/con-lcls2/lcls2
// ./build_all.sh

// Then run it (from lcls2/install/bin/datareader) as 
// datareader

//#include <stdlib.h>
//#include <strings.h>

#include <iostream> // cout, puts etc.
#include <getopt.h>
#include <stdio.h> // printf
#include <fstream> // ifstream

//#include "../include/Logger.h" // MsgLog
#include "psalg/include/Logger.h" // MsgLog
#include "xtcdata/xtc/DescData.hh" // Array
#include "psalg/include/ArrayIO.h" // ArrayIO

using namespace std;
using namespace psalg;

//-------------------

void usage(char* name) {
  std::cout << "Usage: " << name << " some stuff about arguments and options \n";
}

//-------------------

void print_hline(const uint nchars, const char c) {
    printf("%s\n", std::string(nchars,c).c_str());
}

//-------------------

void test_ArrayIO() {

  std::cout << "In test_ArrayIO\n";

  MsgLog("test_ArrayIO", DEBUG, "test Logger");


  //Array(void *data, uint32_t *shape, uint32_t rank)
  ArrayIO<float> a("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");
}

//-------------------

void test_Array() {

  std::cout << "In test_Array\n";
  //Array(void *data, uint32_t *shape, uint32_t rank)

  float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
  uint32_t sh[2] = {3,4};
  uint32_t rank = 2;
  XtcData::Array<float> a(data, sh, rank);

  for(uint32_t r=0; r<sh[0]; r++) {
    std::cout << "\nrow:" << r;
    for(uint32_t c=0; c<sh[1]; c++)
      std::cout << " " << a(r,c);
  }

  std::cout << '\n';

    //printf("%s: %i  %i  %\n",name.name(),arrT(0),arrT(1));
    // T* ptr = reinterpret_cast<T*>(data.payload() + _offset[index]);
    // Array<float> arrT = descdata.get_array<float>(i);
    // printf("%s: %f, %f\n",name.name(),arrT(0),arrT(1));
}

//-------------------

void read_data() {
    /*
    std::string fname("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");
    std::cout << fname << '\n';
    std::ifstream inf(fname.c_str());
    */
    char fname[] = "/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt";

    std::ifstream inf(fname);
 
    std::string str; 
    int counter=0;

    while(getline(inf,str)) {
      counter++; 
      if(counter<10) cout << str << '\n'; // break;
    }

    std::cout << "Number of lines in file: " << counter << '\n';

    inf.close();
}

//-------------------

int test_cli(int argc, char **argv)
{
    std::cout << "\n\nJust a test\n\n";

    char *avalue = NULL;
    char *bvalue = NULL;
    char *cvalue = NULL;
    int index;
    int c;
    extern char *optarg;
    extern int optind, optopt; //opterr;
 
    while((c = getopt (argc, argv, ":a:b:c:h")) != -1)
        switch(c)
        {
          case 'a':
            avalue = optarg;
            printf("a: avalue = %s\n",avalue);
            break;
          case 'b':
            bvalue = optarg;
            printf("b: bvalue = %s\n",bvalue);
            break;
          case 'c':
            cvalue = optarg;
            printf("c: cvalue = %s\n",cvalue);
            break;
          case 'h':
            printf("h: ");
            usage(argv[0]);
            break;
          case ':':
            printf("(:) Option -%c requires an argument.\n", optopt);
            usage(argv[0]);
            return EXIT_FAILURE;
          case '?':
            printf("?: Option -%c requires an argument.\n", optopt);
            usage(argv[0]);
            return EXIT_FAILURE;
          default:
            printf("default: You should not get here... option -%c .\n", optopt);
            abort();
        }
 
    printf("End of options: avalue = %s, bvalue = %s, cvalue = %s\n",
            avalue, bvalue, cvalue);
 
    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
    return EXIT_SUCCESS;
}

//-------------------

int main(int argc, char **argv) {
    std::cout << "\nTests\n"; 
    //test_cli(argc, argv);
                          print_hline(80,'_');
    read_data();          print_hline(80,'_');
    test_Array();         print_hline(80,'_');
    test_ArrayIO();       print_hline(80,'_');
    return EXIT_SUCCESS;
}

//-------------------
