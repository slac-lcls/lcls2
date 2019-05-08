#include <getopt.h>
#include <Python.h>
#include "drp.hh"
#include "DrpApp.hh"

int main(int argc, char* argv[])
{
    Drp::Parameters para;
    int c;
    while((c = getopt(argc, argv, "p:o:l:D:C:d:u:")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.outputDir = optarg;
                break;
            case 'l':
                para.laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'D':
                para.detectorType = optarg;
                break;
            case 'C':
                para.collectionHost = optarg;
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'u':
                para.alias = optarg;
                break;
            default:
                exit(1);
        }
    }
    // Check required parameters
    if (para.device.empty()) {
        printf("-d: device is mandatory!\n");
        exit(1);
    }
    if (para.alias.empty()) {
        printf("-u: alias is mandatory!\n");
        exit(1);
    }

    para.nworkers = 10;
    para.batchSize = 32;
    Py_Initialize(); // for use by configuration
    Drp::DrpApp app(&para);
    app.run();
    Py_Finalize(); // for use by configuration
    app.shutdown();
    std::cout<<"end of main drp\n";
}
