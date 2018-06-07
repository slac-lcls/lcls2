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

#include <iostream> // for cout, puts etc.
#include <getopt.h>
#include <stdio.h>
#include <fstream>

using namespace std;

//-------------------

void usage(char* name) {
  std::cout << "Usage: " << name << " [-a <aaa>] [-b <bbb>] [-c <ccc>] [-h] <p1> [<p2> [<p3> ...]] \n";
}

//-------------------

void read_data() {
  /*
    std::string fname("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");
    std::cout << fname << '\n';
    std::ifstream inf(fname.c_str());
  */
    std::ifstream inf("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");

    std::string str; 
    int counter=0;

    while(getline(inf,str)) {
      counter++; 
      if(counter<40) cout << str << '\n'; // break;
    }

    std::cout << "Number of lines in file: " << counter << '\n';

    inf.close();
}

//-------------------

int main(int argc, char **argv) {
    std::cout << "\nTest read from text file\n"; 
    read_data(); 
    return EXIT_SUCCESS;
}

//-------------------

int main_test(int argc, char **argv)
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
