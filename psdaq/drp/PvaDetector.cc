#include <unistd.h>
#include <iostream>
#include "psdaq/epicstools/PVBase.hh"

class PvaMonitor : public Pds_Epics::PVBase
{
public:
    PvaMonitor(const char* channelName) :
        Pds_Epics::PVBase(channelName)
    {
    }
    void printStructure()
    {
        const pvd::StructureConstPtr& structure = _strct->getStructure();
        const pvd::StringArray& names = structure->getFieldNames();
        const pvd::FieldConstPtrArray& fields = structure->getFields();
        for (unsigned i=0; i<names.size(); i++) {
            std::cout<<"FieldName:  "<<names[i]<<'\n';
            std::cout<<"FieldType:  "<<fields[i]->getType()<<'\n';
        }
    }
    void updated() override
    {
        std::cout<<"updated\n";
        unsigned seconds = getScalarAs<unsigned>("timeStamp.secondsPastEpoch");
        unsigned nanoSeconds = getScalarAs<unsigned>("timeStamp.nanoseconds");
        printf("seconds: %u nanoseconds %u\n", seconds, nanoSeconds);

        pvd::shared_vector<const uint16_t> vec;
        getVectorAs<uint16_t>(vec);
        printf("vector size %lu\n", vec.size());
    }
};

int main()
{
    PvaMonitor monitor("DAQ:LAB2:PVCAM");
    while(1) {
        if (monitor.connected()) {
            break;
        }
        usleep(100000);
    }

    monitor.printStructure();

    sleep(10000);
}
