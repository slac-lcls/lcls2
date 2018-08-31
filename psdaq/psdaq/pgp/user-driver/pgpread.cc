#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include "pgpdriver.h"
#include "xtcdata/xtc/Dgram.hh"

int main(int argc, char* argv[])
{
    
    int c;
    int device_id;
    while((c = getopt(argc, argv, "d:")) != EOF) {
        switch(c) {
            case 'd':
            device_id = std::stoi(optarg, nullptr, 16);
            break;
        }
    }

    int num_entries = 8192;
    DmaBufferPool pool(num_entries, RX_BUFFER_SIZE);
    AxisG2Device dev(device_id);
    dev.init(&pool);       
    dev.setup_lanes(0xF);

    while (true) {    
        DmaBuffer* buffer = dev.read();
        XtcData::Transition* event_header = reinterpret_cast<XtcData::Transition*>(buffer->virt);
        XtcData::TransitionId::Value transition_id = event_header->seq.service();
        printf("Size %u B | Dest %u | Transition id %d | pulse id %lu | event counter %u\n",
                buffer->size, buffer->dest, transition_id, event_header->seq.pulseId().value(), event_header->evtCounter); 
    }                                
} 
