#include "drp.hh"
#include "DrpApp.hh"
#include <getopt.h>

int main(int argc, char* argv[])
{
    Parameters para;
    int c;
    while((c = getopt(argc, argv, "p:o:D:C:")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.output_dir = optarg;
                break;
            case 'D':
                para.detectorType = optarg;
                break;
            case 'C':
                para.collect_host = optarg;
                break;
            default:
                exit(1);
        }
    }
    DrpApp app(&para);
    app.run();
}
