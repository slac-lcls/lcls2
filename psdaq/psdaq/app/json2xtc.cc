#include <cstdio>
#include "psdaq/service/Json2Xtc.hh"

static char readBuffer[4*1024*1024];
static char writeBuffer[4*1024*1024];

int main(int argc, char **argv)
{
    if (argc <= 2) {
        printf("Usage: json2xtc INPUT.json OUTPUT.xtc\n");
        exit(0);
    }
    FILE *fp = fopen(argv[1], "r");
    if (fread(readBuffer, 1, sizeof(readBuffer), fp) <= 0) {
        fprintf(stderr, "Cannot open file!\n");
        exit(0);
    }
    fclose(fp);
    XtcData::NamesId nid(1, 0);
    unsigned len = Pds::translateJson2Xtc(readBuffer, writeBuffer, nid);
    if (len <= 0) {
        fprintf(stderr, "Parse errors, exiting.\n");
        exit(0);
    }
    fp = fopen(argv[2], "w+");
    char buf[20] = {0};   // Hack: sizeof(Dgram) - sizeof(Xtc) = 20
    if (fwrite(buf, 1, 20, fp) != 20) {
        printf("Cannot write header of jungfrau.xtc2?!?\n");
    }
    if (fwrite(writeBuffer, 1, len, fp) != len) {
        printf("Cannot write jungfrau.xtc2?!?\n");
        exit(0);
    }
    fclose(fp);
    return 0;
}
