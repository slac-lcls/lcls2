#include "drp.hh"
#include "DrpApp.hh"
#include <getopt.h>

int main(int argc, char* argv[])
{
    Parameters para;
    int c;
    while((c = getopt(argc, argv, "p:o:l:D:C:d:")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.output_dir = optarg;
                break;
            case 'l':
                para.laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'D':
                para.detectorType = optarg;
                break;
            case 'C':
                para.collect_host = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            default:
                exit(1);
        }
    }
    para.numWorkers = 2;
    para.numEntries = 8192;
    DrpApp app(&para);
    app.run();
}
