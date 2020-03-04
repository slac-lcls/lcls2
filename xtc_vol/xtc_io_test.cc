/*
 * xtc_io_test.cc
 *
 *  Created on: Jan 22, 2020
 *      Author: tonglin
 */

#include "xtc_io_api.cc"
#include "xtc_io_api_c.h"
using namespace std;

int main(int argc, char* argv[]){
    const char* cs = "//a/b/c//d3/";
    vector<string> sv = str_tok(cs, "/");
    string str = tok_to_str(sv);
    printf("Original = [%s], after being tokenized = [%s]\n", cs, str.c_str());
//    int c;
//    char* xtcname = 0;
//    int parseErr = 0;
//    unsigned neventreq = 0xffffffff;
//    bool debugprint = false;
//    printf("XTC_IO_API test\n");
//    while ((c = getopt(argc, argv, "hf:n:d")) != -1) {
//        switch (c) {
//        case 'h':
//            usage(argv[0]);
//            exit(0);
//        case 'f':
//            xtcname = optarg;
//            break;
//        case 'n':
//            neventreq = atoi(optarg);
//            break;
//        case 'd':
//            debugprint = true;
//            break;
//        default:
//            parseErr++;
//        }
//    }
//
//    if (!xtcname) {
//        usage(argv[0]);
//        exit(2);
//    }

    //int fd = open(xtcname, O_RDONLY);
//    if (fd < 0) {
//        fprintf(stderr, "Unable to open file '%s'\n", xtcname);
//        exit(2);
//    }

    //xtc_api_helper* helper = xtc_file_open(xtcname);
    //char* ds_sample_path = "/L1Accept/xpphsd_hsd_fex/1/Off:MyEnumName2";
    //char* grp_sample_path = "/L1Accept/xpphsd_hsd_fex/1";//Off:MyEnumName2
    //_target_open(helper, grp_sample_path);
    //_target_open(fd, ds_sample_path);
    //iterate_list_all(fd);
    //iterate_with_depth(fd, 1);
    return 0;
}
