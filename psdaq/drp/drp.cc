#include "drp.hh"
#include "DrpApp.hh"
#include <getopt.h>
#include <Python.h>

int main(int argc, char* argv[])
{
    Parameters para;
    int c;
    while((c = getopt(argc, argv, "p:o:l:D:C:d:u:")) != EOF) {
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

    para.numWorkers = 10;
    para.numEntries = 131072;
    printf("%d number of buffers\n", para.numEntries);
    Py_Initialize(); // for use by configuration
    DrpApp app(&para);
    app.run();
    Py_Finalize(); // for use by configuration
}
