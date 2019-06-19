#include <getopt.h>
#include <sstream>
#include <Python.h>
#include "drp.hh"
#include "DrpApp.hh"

void get_kwargs(Drp::Parameters& para, const std::string& kwargs_str) {
    std::istringstream ss(kwargs_str);
    std::string kwarg;
    std::string::size_type pos = 0;
    while (getline(ss, kwarg, ',')) {
        pos = kwarg.find("=", pos);
        if (!pos) {
            throw "drp.cc error: keyword argument with no equal sign: "+kwargs_str;
        }
        std::string key = kwarg.substr(0,pos);
        std::string value = kwarg.substr(pos+1,kwarg.length());
        //cout << kwarg << " " << key << " " << value << endl;
        para.kwargs[key] = value;
    }
}

int main(int argc, char* argv[])
{
    Drp::Parameters para;
    int c;
    para.detSegment = 0;
    std::string kwargs_str;
    while((c = getopt(argc, argv, "p:o:l:D:C:d:n:s:k:")) != EOF) {
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
            case 's':
                para.detSegment = std::stoi(optarg);
                break;
            case 'd':
                para.device = optarg;
                break;
            case 'n':
                para.detName = optarg;
                break;
            case 'k':
                kwargs_str = std::string(optarg);
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
    if (para.detName.empty()) {
        printf("-n: detector name is mandatory!\n");
        exit(1);
    }
    if (para.detName.back() == '_') {
        printf("-n: detector name must not end with underscore!\n");
        exit(1);
    }

    get_kwargs(para, kwargs_str);

    para.nworkers = 10;
    para.batchSize = 32;
    Py_Initialize(); // for use by configuration
    Drp::DrpApp app(&para);
    app.run();
    app.shutdown();
    Py_Finalize(); // for use by configuration
    std::cout<<"end of main drp\n";
}
